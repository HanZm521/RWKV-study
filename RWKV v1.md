# RWKV-V1
use explicit decay and Token-shift, AFT.
## Time-mix
$$
\mathrm{TM}_{t,c}=\mathrm{sigmoid}\left(R_{t,c}\right)\cdot\sum_{u}W_{t,u,c}\cdot\mathrm{softmax}_{t}\left(K_{u,c}\right)\cdot V_{u,c}
$$
