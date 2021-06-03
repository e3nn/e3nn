.. _conv guide:

Convolution
===========

In this document we will implement an equivariant convolution with ``e3nn``.
We will implement this formula:

.. math::

    f'_i = \frac{1}{\sqrt{z}} \sum_{j \in \partial(i)} \; f_i \; \otimes\!(h(\|x_{ij}\|)) \; Y(x_{ij} / \|x_{ij}\|)

where

- :math:`f_i, f'_i` are the nodes input and output
- :math:`z` is the average `degree`_ of the nodes
- :math:`\partial(i)` is the set of neighbors of the node :math:`i`
- :math:`x_{ij}` is the relative vector
- :math:`h` is a multi layer perceptron
- :math:`Y` is the spherical harmonics
- :math:`x \; \otimes\!(w) \; y` is a tensor product of :math:`x` with :math:`y` parametrized by some weights :math:`w`

Boilerplate imports

.. jupyter-execute::

    import torch
    from torch_cluster import radius_graph
    from torch_scatter import scatter
    from e3nn import o3, nn
    from e3nn.math import soft_one_hot_linspace
    import matplotlib.pyplot as plt

Let's first define the irreps of the input and output features.

.. jupyter-execute::

    irreps_input = o3.Irreps("10x0e + 10x1e")
    irreps_output = o3.Irreps("20x0e + 10x1e")

And create a random graph using random positions and edges when the relative distance is smaller than ``max_radius``.

.. jupyter-execute::

    # create node positions
    num_nodes = 100
    pos = torch.randn(num_nodes, 3)  # random node positions

    # create edges
    max_radius = 1.8
    edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=num_nodes - 1)

    print(edge_src.shape)

    edge_vec = pos[edge_dst] - pos[edge_src]

    # compute z
    num_neighbors = len(edge_src) / num_nodes
    num_neighbors

``edge_src`` and ``edge_dst`` contain the indices of the nodes for each edge.
And we can also create some random input features.

.. jupyter-execute::

    f_in = irreps_input.randn(num_nodes, -1)

Note that out data is generated with a normal distribution. We will take care of having all the data following the ``component`` normalization (see :ref:`norm guide`).

.. jupyter-execute::

    f_in.pow(2).mean()  # should be close to 1

Let's start with

.. math::

    Y(x_{ij} / \|x_{ij}\|)

.. jupyter-execute::

    irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
    print(irreps_sh)

    sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')
    # normalize=True ensure that x is divided by |x| before computing the sh

    sh.pow(2).mean()  # should be close to 1

Now we need to compute :math:`\otimes(w)` and :math:`h`.
Let's create the tensor product first, it will tell us how many weights it needs.

.. jupyter-execute::

    tp = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)

    print(f"{tp} needs {tp.weight_numel} weights")

    tp.visualize();

in this particual choice of irreps we can see that the l=1 component of the spherical harmonics cannot be used in the tensor product.
In this example it's the equivariance to inversion that prohibit the use of l=1.
If we don't want the equivariance to inversion we can declare all irreps to be even (``irreps_sh = Irreps("0e + 1e + 2e")``).

To implement :math:`h` that has to map the relative distances to the weights of the tensor product we will embed the distances using a basis function and then feed this embedding to a neural network.
Let's create that embedding. Here is the base functions we will use:

.. jupyter-execute::

    num_basis = 10

    x = torch.linspace(0.0, 2.0, 1000)
    y = soft_one_hot_linspace(
        x,
        start=0.0,
        end=max_radius,
        number=num_basis,
        basis='smooth_finite',
        cutoff=True,
    )

    plt.plot(x, y);

Note that this set of functions are all smooth and are strictly zero beyond ``max_radius``.
This is useful to get a convolution that is smooth although the sharp cutoff at ``max_radius``.

Let's use this embedding for the edge distances and normalize it properly (``component`` i.e. second moment close to 1).

.. jupyter-execute::

    edge_length_embedding = soft_one_hot_linspace(
        edge_vec.norm(dim=1),
        start=0.0,
        end=max_radius,
        number=num_basis,
        basis='smooth_finite',
        cutoff=True,
    )
    edge_length_embedding = edge_length_embedding.mul(num_basis**0.5)

    print(edge_length_embedding.shape)
    edge_length_embedding.pow(2).mean()  # the second moment

Now we can create a MLP and feed it

.. jupyter-execute::

    fc = nn.FullyConnectedNet([num_basis, 16, tp.weight_numel], torch.relu)
    weight = fc(edge_length_embedding)

    print(weight.shape)
    print(len(edge_src), tp.weight_numel)

    # For a proper notmalization, the weights also need to be mean 0
    print(weight.mean(), weight.std())  # should close to 0 and 1

Now we can compute the term

.. math::

    f_i \; \otimes\!(h(\|x_{ij}\|)) \; Y(x_{ij} / \|x_{ij}\|)

The idea is to compute this quantity per edges, so we will need to "lift" the input feature to the edges.
For that we use ``edge_src`` that contains, for each edge, the index of the source node.

.. jupyter-execute::

    summand = tp(f_in[edge_src], sh, weight)

    print(summand.shape)
    print(summand.pow(2).mean())  # should be close to 1

Only the sum over the neighbors is remaining

.. math::

    f'_i = \frac{1}{\sqrt{z}} \sum_{j \in \partial(i)} \; f_i \; \otimes\!(h(\|x_{ij}\|)) \; Y(x_{ij} / \|x_{ij}\|)

.. jupyter-execute::

    f_out = scatter(summand, edge_dst, dim=0, dim_size=num_nodes)

    f_out = f_out.div(num_neighbors**0.5)

    f_out.pow(2).mean()  # should be close to 1


Now we can put everything into a function

.. jupyter-execute::

    def conv(f_in, pos):
        edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=len(pos) - 1)
        edge_vec = pos[edge_dst] - pos[edge_src]
        sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')
        emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, max_radius, num_basis, 'smooth_finite', False).mul(num_basis**0.5)
        return scatter(tp(f_in[edge_src], sh, fc(emb)), edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)

Now we can check the equivariance

.. jupyter-execute::

    rot = o3.rand_matrix()
    D_in = irreps_input.D_from_matrix(rot)
    D_out = irreps_output.D_from_matrix(rot)

    # rotate before
    f_before = conv(f_in @ D_in.T, pos @ rot.T)

    # rotate after
    f_after = conv(f_in, pos) @ D_out.T

    torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4)

The tensor product dominates the execution time:

.. jupyter-execute::

    import time
    wall = time.perf_counter()

    edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=len(pos) - 1)
    edge_vec = pos[edge_dst] - pos[edge_src]
    print(time.perf_counter() - wall); wall = time.perf_counter()

    sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')
    print(time.perf_counter() - wall); wall = time.perf_counter()

    emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, max_radius, num_basis, 'smooth_finite', False).mul(num_basis**0.5)
    print(time.perf_counter() - wall); wall = time.perf_counter()

    weight = fc(emb)
    print(time.perf_counter() - wall); wall = time.perf_counter()

    summand = tp(f_in[edge_src], sh, weight)
    print(time.perf_counter() - wall); wall = time.perf_counter()

    scatter(summand, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)
    print(time.perf_counter() - wall); wall = time.perf_counter()



.. _degree: https://en.wikipedia.org/wiki/Degree_(graph_theory)
