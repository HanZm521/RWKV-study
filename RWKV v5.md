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

multi-headed:

code:
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

