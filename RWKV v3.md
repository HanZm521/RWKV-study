## RWKV-V3
RWKV-V3 is a short-term transitional version, which uses a more comprehensive token-shift compared to RWKV-V2 (using different trainable TimeMix factors for R/K/V in the SA and FF layers respectively) :
```python
xx = self.time_shift(x)
xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
```
