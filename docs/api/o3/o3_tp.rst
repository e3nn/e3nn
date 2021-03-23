Tensor Product
==============

Two characteristics of all tensor products (denoted :math:`\otimes`) are:

#. The tensor product is *bilinear*: :math:`(\alpha x_1 + x_2) \otimes y = \alpha x_1 \otimes y + x_2 \otimes y` and :math:`x \otimes (\alpha y_1 + y_2) = \alpha x \otimes y_1 + x \otimes y_2`
#. The tensor product is *equivariant*: :math:`(D x) \otimes (D y) = D (x \otimes y)` (sorry for the very loose notation)

The class `TensorProduct` implements all tensor products of finite direct sums of irreducible representations (`Irreps`).

All the classes here inherit from the class `TensorProduct`.
Each class implements a special case of tensor product.

.. jupyter-execute::
    :hide-code:

    from e3nn import o3


.. jupyter-execute::

    o3.FullTensorProduct('2x0e + 3x1o', '5x0e + 7x1e').visualize()

The full tensor product is the "natural" one. Every possible output is created and the outputs are not mixed with each other. Note how the multiplicities of the outputs are the product of the multiplicities of the respective inputs.


.. jupyter-execute::

    o3.FullyConnectedTensorProduct('5x0e + 5x1e', '6x0e + 4x1e', '15x0e + 3x1e').visualize()

In a fully connected tensor product, all possible paths are created. The outputs are mixed together with learnable parameters. The red color indicates that the path is learned.


.. jupyter-execute::

    o3.ElementwiseTensorProduct('5x0e + 5x1e', '4x0e + 6x1e').visualize()

In the elementwise tensor product, the irreps are multiplied one by one. Note how the inputs have been split and how the multiplicities of the outputs match with the multiplicities of the input.


.. jupyter-execute::

    o3.Linear('5x0e + 4x1e', '6x0e + 7x1e').tp.visualize()

The linear operation is a special case of a tensor product with a constant scalar ``1.0``.


.. automodule:: e3nn.o3.tensor_product.tensor_product
    :members:
    :show-inheritance:
