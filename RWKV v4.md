## RWKV-V4
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

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    F p = 0, q = 0, o = MIN_VALUE;
    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int i = 0; i < T; i++) {
        const int ii = i * C;

        F no = max(o, u + k[ii]);
        F A = exp(o - no);
        F B = exp(u + k[ii] - no);
        y[ii] = (A * p + B * v[ii]) / (A * q + B);

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }
}
```
