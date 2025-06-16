# RWKV-V1
use explicit decay and Token-shift, AFT.
## Time-mix
$$ TM\_{t,c} = \mathrm{sigmoid}(R\_{t,c}) \cdot \sum\_{u=1}^{n} W\_{t,u,c} \cdot \mathrm{softmax}(K\_{u,c}) \cdot V\_{u,c} $$

