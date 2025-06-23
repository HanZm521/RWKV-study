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
self.time_shift = nn.ZeroPad2d((0, 0, 1, -1)) # This is a special fill operation used to shift the input sequence forward by one step in the time dimension (T).
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
self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)

k = self.key(x).transpose(-1, -2)
v = self.value(x).transpose(-1, -2)
r = self.receptance(x) # r = torch.sigmoid(r)
k = torch.clamp(k, max=RWKV_K_CLAMP)
k = torch.exp(k)
```
$$
\begin{array}{c}
c\_{1}=a\_{0}+X \odot k\_{1} \odot v\_{1} \\
d\_{1}=b\_{0}+X \odot k\_{1} \\
a\_{1}=W \odot a_{0}+k\_{1} \odot v\_{1} \\
b\_{1}=W \odot b_{0}+k\_{1}
\end{array}
$$
```python
############# fancy init of time_w curves ###################################
f1_begin = 3.0
f1_end = 1.2
f2_begin = 0.65
f2_end = 0.4
with torch.no_grad():  # initial time_w curves for better convergence
    decay_speed = torch.ones(attn_sz, 1)
    first_sa_layer_id = 1
    for h in range(attn_sz):
        f1 = f1_begin + (layer_id-first_sa_layer_id) / \
            (config.n_layer-1-first_sa_layer_id) * (f1_end - f1_begin)
        f2 = f2_begin + (layer_id-first_sa_layer_id) / \
            (config.n_layer-1-first_sa_layer_id) * (f2_end - f2_begin)
        if layer_id == first_sa_layer_id:
            f1 += 0.5
        if layer_id == config.n_layer-2:
            f2 = 0.4
        if layer_id == config.n_layer-1:
            f2 = 0.37
        decay_speed[h][0] = math.pow(f2, h / (attn_sz-1) * 7) * f1
self.time_decay = nn.Parameter(torch.log(decay_speed)) # will use exp(self.time_decay) to ensure time_decay > 0
self.time_curve = torch.tensor(
    [-(config.ctx_len - 2 - i) for i in range(config.ctx_len-1)]).unsqueeze(0)
self.time_curve = self.time_curve.to('cuda')
self.time_first = nn.Parameter(torch.ones(attn_sz, 1) * math.log(0.3))
#############################################################################

self.time_w = torch.cat(
    [torch.exp(self.time_decay) * self.time_curve, self.time_first], dim=-1)
w = torch.exp(self.time_w)

wkv = TimeX.apply(w, kv, B, C, T, 0)
# RWKV_K_EPS can be removed if the CUDA kernel sets 0/0 = 0 (I will do this later)
wk = TimeX.apply(w, k, B, C, T, RWKV_K_EPS)
```


## Channel-mix
