## RWKV-V3
RWKV-V3 is a short-term transitional version, which uses a more comprehensive token-shift compared to RWKV-V2 (using different trainable TimeMix factors for R/K/V in the SA and FF layers respectively):
```python
xx = self.time_shift(x)
xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
```
Furthermore, this version uses preLN instead of postLN (which is more stable and converges faster):
```python
if self.layer_id == 0:
	x = self.ln0(x)
x = x + self.att(self.ln1(x))
x = x + self.ffn(self.ln2(x))
```
