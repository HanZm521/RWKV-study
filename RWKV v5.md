## RWKV-V5
Compared with RWKV-V4, the most significant change of RWKV-V5 lies in the introduction of multi-headed matrix-valued states, namely "multi-headed matrix-valued states" in the paper.
![image](https://rwkv.cn/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Frwkv-5-6-architecture.eb7a9d99.png&w=3840&q=75)
# Time-mix
RWKV-V5 eliminates the normalization term (the denominator in the RWKV-V4 formula) and introduces the matrix value state instead of the previous vector value state.
equal:

$$
\begin{array}{c}
\square\_{t}=\mathrm{lerp}\_{\square}\left(x\_{t}, x\_{t-1}\right) W\_{\square}, \quad \square \in\{r, k, v, g\} \\
w=\exp (-\exp (\omega)) \\
w k v\_{t}=\mathrm{diag}(u) \cdot k\_{t}^{\top} \cdot v\_{t}+\sum\_{i=1}^{t-1} \mathrm{diag}(w)^{t-1-i} \cdot k\_{i}^{\top} \cdot v\_{i} \in \mathbb{R}^{(D / h) \times(D / h)} \\
o\_{t}=\mathrm{concat}\left(\mathrm{SiLU}\left(g\_{t}\right) \odot \mathrm{LayerNorm}\left(r\_{t} \cdot w k v\_{t}\right)\right) W\_{o} \in \mathbb{R}^{D}
\end{array}
$$

![image](https://github.com/HanZm521/RWKV-study/blob/main/fig/rwkv5.png?raw=true)
multi-headed:

```python
self.head_size = args.head_size_a
assert HEAD_SIZE == self.head_size # change HEAD_SIZE to match args.head_size_a
self.n_head = args.dim_att // self.head_size
```

Gate control mechanism:
```python
self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

g = F.silu(self.gate(xg))

x = self.output(x * g)
```

Diagonal parameterization:

w and u are directly stored as vectors of [H, N], rather than matrices of [H, N, N]

```C
#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    float state[_N_] = {0};

    __syncthreads();
    w[i] = _w[i];
    u[i] = float(_u[i]);
    __syncthreads();

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
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
