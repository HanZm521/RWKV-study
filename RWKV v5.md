## RWKV-V5
Compared with RWKV-V4, the most significant change of RWKV-V5 lies in the introduction of multi-headed matrix-valued states, namely "multi-headed matrix-valued states" in the paper.
![image](https://rwkv.cn/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Frwkv-5-6-architecture.eb7a9d99.png&w=3840&q=75)
# Time-mix
equal:
$$
\begin{array}{c}
\square\_{t}=\mathrm{lerp}\_{\square}\left(x\_{t}, x\_{t-1}\right) W\_{\square}, \quad \square \in\{r, k, v, g\} \\
w=\exp (-\exp (\omega)) \\
w k v\_{t}=\mathrm{diag}(u) \cdot k\_{t}^{\top} \cdot v\_{t}+\sum\_{i=1}^{t-1} \mathrm{diag}(w)^{t-1-i} \cdot k\_{i}^{\top} \cdot v\_{i} \in \mathbb{R}^{(D / h) \times(D / h)} \\
o\_{t}=\mathrm{concat}\left(\mathrm{SiLU}\left(g\_{t}\right) \odot \mathrm{LayerNorm}\left(r\_{t} \cdot w k v\_{t}\right)\right) W\_{o} \in \mathbb{R}^{D}
\end{array}
$$
code:
