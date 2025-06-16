<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
# RWKV-V1
use explicit decay and Token-shift, AFT.
## Time-mix
$TM_{t,c} =\mathrm{sigmoid}(R_{t,c})\cdot \sum_{u}^{} W_{t,u,c} \cdot \mathrm{softmax}_t(K_{u,c})\cdot V_{u,c}$

