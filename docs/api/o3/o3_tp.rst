Tensor Product
==============

All tensor products --- denoted :math:`\otimes` --- share two key characteristics:

#. The tensor product is *bilinear*: :math:`(\alpha x_1 + x_2) \otimes y = \alpha x_1 \otimes y + x_2 \otimes y` and :math:`x \otimes (\alpha y_1 + y_2) = \alpha x \otimes y_1 + x \otimes y_2`
#. The tensor product is *equivariant*: :math:`(D x) \otimes (D y) = D (x \otimes y)` where :math:`D` is the representation of some symmetry operation from :math:`E(3)` (sorry for the very loose notation)

The class `e3nn.o3.TensorProduct` implements all possible tensor products between finite direct sums of irreducible representations (`e3nn.o3.Irreps`). While `e3nn.o3.TensorProduct` provides maximum flexibility, a number of sublcasses provide various typical special cases of the tensor product:

* `e3nn.o3.FullTensorProduct`:

.. jupyter-execute::
    :hide-code:

    from e3nn import o3


.. jupyter-execute::

    tp = o3.FullTensorProduct(
        irreps_in1='2x0e + 3x1o',
        irreps_in2='5x0e + 7x1e'
    )
    print(tp)
    tp.visualize();

The full tensor product is the "natural" one. Every possible output --- each output irrep for every pair of input irreps --- is created and returned independently. The outputs are not mixed with each other. Note how the multiplicities of the outputs are the product of the multiplicities of the respective inputs.

* `e3nn.o3.FullyConnectedTensorProduct`

.. jupyter-execute::

    tp = o3.FullyConnectedTensorProduct(
        irreps_in1='5x0e + 5x1e',
        irreps_in2='6x0e + 4x1e',
        irreps_out='15x0e + 3x1e'
    )
    print(tp)
    tp.visualize();

In a fully connected tensor product, all paths that lead to any of the irreps specified in ``irreps_out`` are created. Unlike `e3nn.o3.FullTensorProduct`, each output is a learned weighted sum of compatible paths. This allows `e3nn.o3.FullyConnectedTensorProduct` to produce outputs with any multiplicity; note that the example above has :math:`5 \times 6 + 5 \times 4 = 50` ways of creating scalars (``0e``), but the specified ``irreps_out`` has only 15 scalars, each of which is a learned weighted combination of those 50 possible scalars. The blue color in the visualization indicates that the path has these learnable weights.

All possible output irreps do **not** need to be included in ``irreps_out`` of a `e3nn.o3.FullyConnectedTensorProduct`: ``o3.FullyConnectedTensorProduct(irreps_in1='5x1o', irreps_in2='3x1o', irreps_out='20x0e')`` will only compute inner products between its inputs, since ``1e``, the output irrep of a vector cross product, is not present in ``irreps_out``. Note also in this example that there are 20 output scalars, even though the given inputs can produce only 15 unique scalars --- this is again allowed because each output is a learned linear combination of those 15 scalars, placing no restrictions on how many or how few outputs can be requested.

* `e3nn.o3.ElementwiseTensorProduct`

.. jupyter-execute::

    tp = o3.ElementwiseTensorProduct(
        irreps_in1='5x0e + 5x1e',
        irreps_in2='4x0e + 6x1e'
    )
    print(tp)
    tp.visualize();

In the elementwise tensor product, the irreps are multiplied one-by-one. Note in the visualization how the inputs have been split and that the multiplicities of the outputs match with the multiplicities of the input.

* `e3nn.o3.TensorSquare`

.. jupyter-execute::

    tp = o3.TensorSquare("5x1e + 2e")
    print(tp)
    tp.visualize();

The tensor square operation only computes the non-zero entries of a tensor times itself.
It also applies different normalization rules taking into account that a tensor time itself is statistically different from the product of two independent tensors.


.. autoclass:: e3nn.o3.TensorProduct
    :members:
    :show-inheritance:

.. autoclass:: e3nn.o3.FullyConnectedTensorProduct
    :members:
    :show-inheritance:

.. autoclass:: e3nn.o3.FullTensorProduct
    :members:
    :show-inheritance:

.. autoclass:: e3nn.o3.ElementwiseTensorProduct
    :members:
    :show-inheritance:

.. autoclass:: e3nn.o3.TensorSquare
    :members:
    :show-inheritance:
