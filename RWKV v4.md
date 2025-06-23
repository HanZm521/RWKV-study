## RWKV-V4
![image](https://rwkv.cn/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Frwkv-architecture-for-language-modeling.dd136ff1.jpg&w=3840&q=75)
# Time-mix
from RWKV-V3
```python
# Compute the W-curve = [e^(-n * e^time_decay), e^(-(n-1) * e^time_decay), ..., 1, e^(time_first)]
self.time_w = torch.cat(
    [torch.exp(self.time_decay) * self.time_curve, self.time_first], dim=-1)
w = torch.exp(self.time_w)

# Use W to mix kv and k respectively. Add K_EPS to wk to avoid divide-by-zero
wkv = TimeX.apply(w, kv, B, C, T, 0)
# RWKV_K_EPS can be removed if the CUDA kernel sets 0/0 = 0 (I will do this later)
wk = TimeX.apply(w, k, B, C, T, RWKV_K_EPS)

rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)
```
#cuda
```C
#define F4(A, B) ((float4 *)(A))[(B) >> 2]

template <typename F>
__global__ void kernel_forward(const F *__restrict__ const __w, const F *__restrict__ const __k, F *__restrict__ const x,
                               const F eps, const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int ij = (B * C) / BF;
    const int t = threadIdx.x << 2;

    __shared__ F ww[Tmax];
    __shared__ F kk[Tmax * BF];
    F4(ww, t) = F4(__w, t + T * (i % C));
    
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        F4(kk, t + Tmax * j) = F4(__k, t + T * (i + ij * j));
    }
    __syncthreads();

    float4 s[BF];
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        s[j] = {eps, eps, eps, eps};
    }
    const F *__restrict__ const w = ww + T - t - 4;
    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < BF; j++) {
            const F x = kk[u + Tmax * j];
            s[j].x += w[u + 3] * x;
            s[j].y += w[u + 2] * x;
            s[j].z += w[u + 1] * x;
            s[j].w += w[u + 0] * x;
        }
    }
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        const F *__restrict__ const k = kk + Tmax * j;
        s[j].y += w[t + 3] * k[t + 1];
        s[j].z += w[t + 2] * k[t + 1];
        s[j].z += w[t + 3] * k[t + 2];
        s[j].w += w[t + 1] * k[t + 1];
        s[j].w += w[t + 2] * k[t + 2];
        s[j].w += w[t + 3] * k[t + 3];
        F4(x, t + T * (i + ij * j)) = s[j];
    }
}
```
to RWKV-V4
```python
rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
```
#CUDA
```C
#define MIN_VALUE (-1e38)

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C,
                               const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                               F *__restrict__ const _y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c]; // time_first parameter
    F w = _w[_c]; // time_decay parameter
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    F p = 0, q = 0, o = MIN_VALUE; // p: numerator state, q: denominator state, o: max exponent for numerical stability
    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int i = 0; i < T; i++) {
        const int ii = i * C; // Stride to access k_i, v_i, y_i for the current channel

        // 1. Calculate output y[i] for current timestep i
        //    o_prev is the 'o' from the previous iteration.
        //    p_prev is the 'p' from the previous iteration.
        //    q_prev is the 'q' from the previous iteration.
        //    k[ii] is k_i for the current timestep.
        //    v[ii] is v_i for the current timestep.
        F no = max(o, u + k[ii]);      // no = max(o_prev, time_first + k_i); New max exponent for y_i calculation.
        F A = exp(o - no);            // A = exp(o_prev - no); Factor to scale down previous state (p_prev, q_prev). Corresponds to alpha_i.
        F B = exp(u + k[ii] - no);    // B = exp(time_first + k_i - no); Factor for current value v_i. Corresponds to beta_i.
        y[ii] = (A * p + B * v[ii]) / (A * q + B); // y_i = (A*p_prev + B*v_i) / (A*q_prev + B); Compute current output.

        // 2. Update recurrent state (p, q, o) for the next timestep (i+1)
        //    o_prev is still the 'o' from before the y_i calculation.
        //    p_prev is still the 'p' from before the y_i calculation.
        //    q_prev is still the 'q' from before the y_i calculation.
        no = max(w + o, k[ii]);       // o_next = max(time_decay + o_prev, k_i); New max exponent for state update.
        A = exp(w + o - no);          // A_state = exp(time_decay + o_prev - o_next); Factor to scale down previous state for next step. Corresponds to alpha_prime_i.
        B = exp(k[ii] - no);          // B_state = exp(k_i - o_next); Factor for current value's contribution to next state. Corresponds to beta_prime_i.
        p = A * p + B * v[ii];        // p_next = A_state*p_prev + B_state*v_i; Update numerator state.
        q = A * q + B;                // q_next = A_state*q_prev + B_state; Update denominator state (B_state * 1 implicitly).
        o = no;                       // o_prev becomes o_next for the next iteration.
    }
}
```
