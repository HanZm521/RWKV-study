## RWKV-v6
![image](https://rwkv.cn/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Frwkv-6-architecture.bb29d2b3.png&w=1200&q=75)
# Token Shift
for RWKV-V5:

$$\mathrm{lerp}\_{\square}(a, b)=a+(b-a) \odot \mu\_{x}$$

to RWKV-V6: 

RWKV-V6 draws on the technology of low-rank adaptation (LoRA), replacing the static parameter μ with the dynamic LoRA.

$$
\begin{array}{c}
\mathrm{lora}\_{\square}(x)=\lambda\_{\square}+\tanh \left(x A\_{\square}\right) B\_{\square} \\
\mathrm{ddlerp}\_{\square}(a, b)=a+(b-a) \odot \mathrm{lora}\_{\square}\left(a+(b-a) \odot \mu\_{x}\right)
\end{array}
$$

code:
```python
# init:
self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

# forward:
xx = self.time_shift(x) - x

xxx = x + xx * self.time_maa_x
xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
mw, mk, mv, mr, mg = xxx.unbind(dim=0)

xw = x + xx * (self.time_maa_w + mw)
xk = x + xx * (self.time_maa_k + mk)
xv = x + xx * (self.time_maa_v + mv)
xr = x + xx * (self.time_maa_r + mr)
xg = x + xx * (self.time_maa_g + mg)
```

# Time-mix

The $w\_{t}$ of RWKV-V6 is not static throughout the sequence. This is the core change of the attenuation of RWKV-V6:

$$
\begin{array}{c}
\square\_{t}=\mathrm{ddlerp}\_{\square}\left(x\_{t}, x\_{t-1}\right) W\_{\square}, \quad \square \in\{r, k, v, g\} \\
d\_{t}=\mathrm{lora}\_{d}\left(\mathrm{ddlerp}\_{d}\left(x\_{t}, x\_{t-1}\right)\right) \\
w\_{t}=\exp \left(-\exp \left(d\_{t}\right)\right)
\end{array}
$$

```python
ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
w = self.time_decay + ww
```

$$
\begin{array}{c}
w k v\_{t}=\mathrm{diag}(u) \cdot k\_{t}^{\mathrm{T}} \cdot v\_{t}+\sum\_{i=1}^{t-1} \mathrm{diag}\left(\bigodot\_{j=1}^{i-1} w\_{j}\right) \cdot k\_{i}^{\mathrm{T}} \cdot v\_{i} \in \mathbb{R}^{(D / h) \times(D / h)} \\
o\_{t}=\mathrm{concat}\left(\mathrm{SiLU}\left(g\_{t}\right) \odot \mathrm{LayerNorm}\left(r\_{t} \cdot w k v\_{t}\right)\right) W\_{o} \in \mathbb{R}^{D} \\
w k v^{\prime}=s+\mathrm{diag}(u) \cdot k^{\mathrm{T}} \cdot v \\
s^{\prime}=\mathrm{diag}(w) \cdot s+k^{\mathrm{T}} \cdot v
\end{array}
$$

Unlike RWKV-V5, the $w\_{t}$ of RWKV-V6 is not static throughout the sequence. This is the core change of RWKV-V6 attenuation: Each channel of $w\_{t}$ can change independently according to the data dependency, while previously it was a fixed learning vector.

(There is almost no difference in the actual cuda implementation.)

code:

```C
template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ _w, const F *__restrict__ _u, const F *__restrict__ _s,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _u += h*_N_;
    _s += b*H*_N_*_N_ + h*_N_*_N_ + i*_N_;

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    float state[_N_];

    __syncthreads();
    u[i] = float(_u[i]);
    __syncthreads();
    for (int j = 0; j < _N_; j++) {
        state[j] = float(_s[j]);
    }

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        w[i] = __expf(-__expf(float(_w[t])));
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F(y);
    }
}
```


