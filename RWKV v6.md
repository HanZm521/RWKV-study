## RWKV-v6
![image](https://rwkv.cn/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Frwkv-6-architecture.bb29d2b3.png&w=1200&q=75)
# Token Shift
for RWKV-V5:

$$\mathrm{lerp}\_{\square}(a, b)=a+(b-a) \odot \mu\_{x}$$

to RWKV-V6:

$$
\begin{array}{c}
\mathrm{lora}\_{\square}(x)=\lambda\_{\square}+\tanh \left(x A\_{\square}\right) B\_{\square} \\
\mathrm{ddlerp}\_{\square}(a, b)=a+(b-a) \odot \mathrm{lora}\_{\square}\left(a+(b-a) \odot \mu\_{x}\right)
\end{array}
$$
