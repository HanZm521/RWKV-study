# RWKV-V1
use explicit decay and Token-shift, AFT.
## Time-mix
$TM_{t,c} =\mathrm{sigmoid}(R_{t,c})\cdot \sum_{u}^{} W_{t,u,c} \cdot \mathrm{softmax}_t(K_{u,c})\cdot V_{u,c}$

