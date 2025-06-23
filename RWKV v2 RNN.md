# RWKV-V2
The RWKV-v2 version implemented the RNN mode for RWKV for the first time.
![image](https://rwkv.cn/_next/image?url=%2F_next%2Fstatic%2Fmedia%2FRWKV-v2-RNN-Architecture.36e56c99.jpg&w=1200&q=75)
## Time-mix
$$x\_{1}=\mathrm{LN}\left(x\_{1}\right)$$
```python
self.ln1 = nn.LayerNorm(config.n_embd)
x = self.ln1(x)
```
$$z_{1}=T \odot x_{0}+(1-T) \odot x_{1}$$
```python
self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():  # init to "shift half of the channels"
            ww = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd // 2):
                ww[0, 0, i] = 0
        self.time_mix = nn.Parameter(ww)

x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)
```
$$
\begin{array}{c}
k\_{1}=\exp \left(K \cdot z\_{1}\right) \\
v\_{1}=V \cdot z\_{1} \\
r\_{1}=\mathrm{sigmoid}\left(R \cdot z\_{1}\right)
\end{array}
$$
```python
## Channel-mix
