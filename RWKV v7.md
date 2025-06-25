## RWKV-V7
![image](https://rwkv.cn/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Frwkv-7-architecture.82025bb4.jpg&w=1200&q=75)

RWKV-V7 adopts Dynamic State Evolution. 

In simple terms, traditional attention mechanisms (such as the QKV-softmax-attention in Transformers) store multiple k, v (key and value vector pairs) and use q (query vector) to match the keys, producing the corresponding value output.

RWKV-V7 does not directly store k, v pairs. Instead, it dynamically computes and updates a state, learning the relationship between keys and values from the context. The updated state is then used to process new inputs q (denoted as r in RWKV) and generate the output.

Specifically, the RWKV-V7 model maintains an internal model $v \approx k S^{\top}$.It aims to fit a simple objective: for given vector sequences $k\_{t}$ and $v\_{t}$,the state $S$ transforms $k\_{i}$into $v\_{i}$, ensuring the output $v$ closely matches the target $v$. To achieve this, during inference, RWKV-V7 automatically simulates dynamic gradient descent for the L2 loss function: 
$$mathcal{L}=\frac{1}{2}\left\|v-k S^{\top}\right\|^{2}$$
This allows the model to continuously train the internal approximation $v \approx k S^{\top}$.
