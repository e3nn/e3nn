Point inputs with periodic boundary conditions
================================================================

This example shows how to give point inputs with periodic boundary conditions
(e.g. crystal data) to a Euclidean neural network built with ``e3nn``. For a specific
application, this code should be modified with a more tailored network design.

.. jupyter-execute::

    import torch
    import e3nn
    import ase
    import ase.neighborlist
    import torch_geometric
    import torch_geometric.data
    
    default_dtype = torch.float64
    torch.set_default_dtype(default_dtype)

Example crystal structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, we create some crystal structures which have periodic boundary conditions.

.. jupyter-execute::

    # A lattice is a 3 x 3 matrix
    # The first index is the lattice vector (a, b, c)
    # The second index is a Cartesian index over (x, y, z)
    
    # Polonium with Simple Cubic Lattice
    po_lattice = torch.eye(3) * 3.340  # Cubic lattice with edges of length 3.34 AA
    po_coords = torch.tensor([[0., 0., 0.,]])
    po_types = ['Po']
    
    # Silicon with Diamond Structure
    si_lattice = torch.tensor([
        [0.      , 2.734364, 2.734364],
        [2.734364, 0.      , 2.734364],
        [2.734364, 2.734364, 0.      ]
    ])
    si_coords = torch.tensor([
        [1.367182, 1.367182, 1.367182],
        [0.      , 0.      , 0.      ]
    ])
    si_types = ['Si', 'Si']
    
    po = ase.Atoms(symbols=po_types, positions=po_coords, cell=po_lattice, pbc=True)
    si = ase.Atoms(symbols=si_types, positions=si_coords, cell=si_lattice, pbc=True)

Create and store periodic graph data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use the `ase.neighborlist.neighbor_list` algorithm and a ``radial_cutoff`` 
distance to define which edges to include in the graph to represent
interactions with neighboring atoms. Note that for a convolutional network, the 
number of layers determines the receptive field, i.e. how “far out” any given atom
can see. So even if a we use a ``radial_cutoff = 3.5``, a two layer network
effectively sees ``2 * 3.5 = 7`` distance units (in this case Angstroms) away and a
three layer network ``3 * 3.5 = 10.5`` distance units. We then store our data
in ``torch_geometric.data.Data`` objects that we will batch with 
``torch_geometric.data.DataLoader`` below.

.. jupyter-execute::

    radial_cutoff = 3.5  # Only include edges for neighboring atoms within a radius of 3.5 Angstroms.
    type_encoding = {'Po': 0, 'Si': 1}
    type_onehot = torch.eye(len(type_encoding))
    
    dataset = []
    
    dummy_energies = torch.randn(2, 1, 1)  # dummy energies for example
    
    for crystal, energy in zip([po, si], dummy_energies):
        # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
        # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
        edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list("ijS", a=crystal, cutoff=radial_cutoff, self_interaction=True)
        
        data = torch_geometric.data.Data(
            pos=torch.tensor(crystal.get_positions()),
            lattice=torch.tensor(crystal.cell.array).unsqueeze(0),  # We add a dimension for batching
            x=type_onehot[[type_encoding[atom] for atom in crystal.symbols]],  # Using "dummy" inputs of scalars because they are all C
            edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
            edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
            energy=energy  # dummy energy (assumed to be normalized "per atom")
        )
        
        dataset.append(data)
        
    print(dataset)


The first ``Data`` object is for simple cubic Polonium which has 7
edges: 6 for nearest neighbors and 1 as a “self” edge, ``6 + 1 = 7``.
The second ``Data`` object is for diamond Silicon which has 10 edges: 4
nearest neighbors for each of the two atoms and 2 “self” edges, one for
each atom, ``4 * 2 + 1 * 2 = 10``. The lattice of each structure has a
shape of ``[1, 3, 3]`` such that when we batch examples, the batched
lattices will have shape ``[batch_size, 3, 3]``.

Graph Batches
~~~~~~~~~~~~~
``torch_geometric.data.DataLoader`` create batches of 
differently sized structures and produces `Data` objects containing a batch when
iterated over.

.. jupyter-execute::

    batch_size = 2
    dataloader = torch_geometric.data.DataLoader(dataset, batch_size=batch_size)
    
    for data in dataloader:
        print(data)
        print(data.batch)
        print(data.pos)
        print(data.x)

``data.batch`` is the batch index which is tensor of shape
``[batch_size]`` that stores which points or “atoms” belong to which 
example. In this case, since we only have two examples in our batch, the batch
tensor only contains the numbers ``0`` and ``1``. The batch index is
often passed to ``scatter`` `operations to aggregate per examples
values <https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html>`__,
e.g. the total energy for a single crystal structure.

For more details on batching with ``torch_geometric``, please see `this
page <https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html>`__.

Relative distance vectors of edges with periodic boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To calculate the vectors associated with each edge for a given `Data` 
object representing a single example, we use the following expression:

.. parsed-literal::

    edge_src, edge_dst = data['edge_index'][0], data['edge_index'][1]
    edge_vec = (data['pos'][edge_dst] - data['pos'][edge_src] 
                + torch.einsum('ni,nij->j', data['edge_shift'], data['lattice']))

The first line in the definition of ``edge_vec`` is simply how one normally computes
relative distance vectors given two points. The second line adds the contribution
to the relative distance vector due to crossing unit cell boundaries i.e. 
if atoms belong to different images of the unit cell. As we will see below, we can
modify this expression to also include the ``data['batch']`` tensor when handling
batched data.

One Approach: Adding a Preprocessing Method to the Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While ``edge_vec`` can be stored in the `Data` object, it can also be calculated
by adding a preprocessing method to the Network. For this example, we create a 
modified version of the example network ``SimpleNetwork`` `documented
here <https://docs.e3nn.org/en/stable/api/nn/models/v2103.html#simple-network>`__
with `source code
here <https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2103/gate_points_networks.py>`__.
``SimpleNetwork`` is a good starting point to check your data pipeline
but should be replaced with a more tailored network for your specific
application.

.. jupyter-execute::

    from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
    from typing import Dict, Union
    import torch_scatter
    
    class SimplePeriodicNetwork(SimpleNetwork):
        def __init__(self, **kwargs):
            """The keyword `pool_nodes` is used by SimpleNetwork to determine 
            whether we sum over all atom contributions per example. In this example, 
            we want use a mean operations instead, so we will override this behavior.
            """
            self.pool = False
            if kwargs['pool_nodes'] == True:
                kwargs['pool_nodes'] = False
                kwargs['num_nodes'] = 1.
                self.pool = True
            super().__init__(**kwargs)
        
        # Overwriting preprocess method of SimpleNetwork to adapt for periodic boundary data
        def preprocess(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
            if 'batch' in data:
                batch = data['batch']
            else:
                batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)
    
            edge_src = data['edge_index'][0]  # Edge source
            edge_dst = data['edge_index'][1]  # Edge destination
            
            # We need to compute this in the computation graph to backprop to positions 
            # We are computing the relative distances + unit cell shifts from periodic boundaries
            edge_batch = batch[edge_src]
            edge_vec = (data['pos'][edge_dst] 
                        - data['pos'][edge_src] 
                        + torch.einsum('ni,nij->j', data['edge_shift'], data['lattice'][edge_batch]))
            
            return batch, data['x'], edge_src, edge_dst, edge_vec
        
        def forward(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
            # if pool_nodes was set to True, use scatter_mean to aggregate
            output = super().forward(data)
            if self.pool == True:
                return torch_scatter.scatter_mean(output, data.batch, dim=0)  # Take mean over atoms per example
            else:
                return output


We define and run the network.

.. jupyter-execute::

    net = SimplePeriodicNetwork(
        irreps_in="2x0e",  # One hot scalars (L=0 and even parity) on each atom to represent atom type
        irreps_out="1x0e",  # Single scalar (L=0 and even parity) to output (for example) energy
        max_radius=radial_cutoff, # Cutoff radius for convolution
        num_neighbors=10.0,  # scaling factor based on the typical number of neighbors
        pool_nodes=True,  # We pool nodes to predict total energy
    )

When we apply the network to our data, we get one scalar per example.

.. jupyter-execute::

    for data in dataloader:
        print(net(data).shape)  # One scalar per example


