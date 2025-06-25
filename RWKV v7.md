## RWKV-V7
![image](https://rwkv.cn/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Frwkv-7-architecture.82025bb4.jpg&w=1200&q=75)

RWKV-V7 adopts Dynamic State Evolution. 

In simple terms, traditional attention mechanisms (such as the QKV-softmax-attention in Transformers) store multiple k, v (key and value vector pairs) and use q (query vector) to match the keys, producing the corresponding value output.

RWKV-V7 does not directly store k, v pairs. Instead, it dynamically computes and updates a state, learning the relationship between keys and values from the context. The updated state is then used to process new inputs q (denoted as r in RWKV) and generate the output.

Specifically, the RWKV-V7 model maintains an internal model $v \approx k S^{\top}$. It aims to fit a simple objective: for given vector sequences $k\_{t}$ and $v\_{t}$,the state $S$ transforms  $k\_{i}$ into $v\_{i}$, ensuring the output $v$ closely matches the target $v$. To achieve this, during inference, RWKV-V7 automatically simulates dynamic gradient descent for the L2 loss function: 

$$
L=\frac{1}{2}\left\|v-k S^{\top}\right\|^{2}
$$

This allows the model to continuously train the internal approximation $v \approx k S^{\top}$.

The gradient formula of state:

$$ 
\frac{\partial L}{\partial S}=S k^{\top} k-v^{\top} k
$$

equivalent to:

$$
S\_{t}=S\_{t-1}\left(\mathrm{diag}\left(w\_{t}\right)-k\_{t}^{\top} k\_{t} \mathrm{diag}\left(\eta\_{t}\right)\right)+v\_{t}^{\top} k\_{t} \mathrm{diag}\left(\eta\_{t}\right)
$$

Comparison of the time step formula and status update mechanism between RWKV-V7 and the historical version (RWKV-5/6):
![image](https://github.com/HanZm521/RWKV-study/blob/main/fig/rwkv7.png)

code:

```python
def ref_fwd(r, w, k, v, a, b):
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    a = a.view(B, T, H, N)
    b = b.view(B, T, H, N)
    w = torch.exp(-torch.exp(w.view(B, T, H, N)))
    out = torch.zeros((B, T, H, N), device=DEVICE)
    state = torch.zeros((B, H, N, N), device=DEVICE)
 
    for t in range(T):
        kk = k[:, t, :]
        rr = r[:, t, :]
        vv = v[:, t, :]
        aa = a[:, t, :]
        bb = b[:, t, :]
        sab = torch.einsum('bhik,bhk,bhj->bhij', state, aa, bb)
        state = state * w[: , t, :, None, :] + sab + torch.einsum('bhj,bhi->bhij', kk, vv)
        out[:, t, :] = torch.einsum('bhj,bhij->bhi', rr, state)
 
    return out.view((B, T, C))
```
