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
