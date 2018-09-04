# Partially adaptive momentum estimation method for Keras
Keras implementation of Padam from [Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks](https://arxiv.org/abs/1806.06763).

Padam allows for much larger learning rates to be utilized, and follows generalization closely with Stochastc Gradient Descent.

# Usage
Add the `padam.py` script and import `Padam`. Apart from the other parameters which are obtained from Adam, Padam has an additional parameter - `partial`, which should be modified to lie in the range [0, 0.5]. 

```python
from padam import Padam

optimizer = Padam(lr=0.1, partial=0.125)

```

# Requirements

- Keras 2.2.0+
- Tensorflow / Theano / CNTK (Tensorflow tested)
