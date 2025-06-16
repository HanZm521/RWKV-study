# RWKV-V1
RWKV-V1 uses the long volume product instead of the Attention mechanism, and its architecture consists of alternating Time-mix and Channel-mix. Channel-mix is a variant of the GeGLU layer of the Transformer. Time-mix is a significant improvement of AFT.
## Time-mix
equation:
$$ \mathrm{TM}\_{t,c} = \mathrm{sigmoid}(R\_{t,c}) \cdot \sum\_{u} W\_{t,u,c} \cdot \mathrm{softmax}(K\_{u,c}) \cdot V\_{u,c} $$
code:
```python
class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_attn % config.n_head == 0
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_attn // config.n_head

        with torch.no_grad(): # initial time_w curves for better convergence
            ww = torch.ones(config.n_head, config.ctx_len)
            curve = torch.tensor([-(config.ctx_len - 1 - i) for i in range(config.ctx_len)]) # the distance
            for h in range(config.n_head):
                if h < config.n_head - 1:
                    decay_speed = math.pow(config.ctx_len, -(h+1)/(config.n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
                # print('layer', layer_id, 'head', h, 'decay_speed', round(decay_speed, 4), ww[h][:5].numpy(), '...', ww[h][-5:].numpy())
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
                
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.receptance = nn.Linear(config.n_embd, config.n_attn)

        # if config.rwkv_tiny_attn > 0:
        #     self.tiny_att = RWKV_TinyAttn(config)

        self.output = nn.Linear(config.n_attn, config.n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60) # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv * self.time_gamma[:T, :]
```
### AFT
equation:
$$Y\_{t}=\sigma\_{Q}\left(Q\_{t}\right) \odot \frac{\sum_{t^{\prime}=1}^{T} \exp \left(\mathbf{K}\_{t^{\prime}}+w\_{t, t^{\prime}}\right) \odot V\_{t^{\prime}}}{\sum\_{t^{\prime}=1}^{T} \exp \left(\mathbf{K}\_{t^{\prime}}+w\_{t, t^{\prime}}\right)}$$
##  Channel-mix
equation:
$$\mathrm{CM}\_{t, c}=\mathrm{sigmoid}\left(R\_{t, c}\right) \cdot \sum_{d} W_{c, d} \cdot \mathrm{gelu}\left(K\_{t, d}\right) \cdot V\_{t, d}$$
code:
```python
class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        
        hidden_sz = 5 * config.n_ffn // 2 # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        
        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        
        wkv = self.weight(F.mish(k) * v) # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv
```
