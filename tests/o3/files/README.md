The `tp_gpu.ckpt` file has been generated with the following script:

```python
from e3nn import o3
import torch

tp = o3.FullyConnectedTensorProduct("1x0e+1x1o", "1x0e+1x1o", "1x0e+1x1o+1x2e")
tp = tp.to("cuda")
torch.save(tp, "tp_gpu.ckpt")
```

Loading this file on CPU can cause problems due to the loading of codegen.
Currently these problems are solved by passing the `map_location` explicitly.
Interestingly, if "1x2e" is removed from the `irreps_out`, it causes no problems.

