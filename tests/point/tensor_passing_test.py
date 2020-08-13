import torch
from torch.autograd import gradcheck

from e3nn.point.message_passing import TensorPassingHomogenous, TensorPassingLayer, TensorPassingContext
from e3nn.radial import GaussianRadialModel
from e3nn.non_linearities.rescaled_act import swish
from functools import partial

from torch_geometric.data import Data


def test_tpnn():
    torch.set_default_dtype(torch.float64)

    representations = [[(15, 0)], [(2, 2), (3, 1), (4, 2)], [(2, 0), (2, 1)]]
    radial_model = partial(GaussianRadialModel, max_radius=3.2, min_radius=0.7, number_of_basis=10, h=100, L=3, act=swish)
    model = TensorPassingHomogenous(representations, radial_model)
    model = model.cuda()

    features = torch.randn(5, 15, device='cuda:0', requires_grad=True)
    pos = torch.randn(5, 3, device='cuda:0')
    edge_index = torch.tensor([[1, 2, 3, 4], [3, 1, 4, 0]], dtype=torch.int64, device='cuda:0')
    origin_pos = pos[edge_index[0]]
    neighbor_pos = pos[edge_index[1]]
    rel_vectors = neighbor_pos - origin_pos

    abs_distances = torch.norm(rel_vectors, 2, -1, keepdim=True)
    rel_vectors = rel_vectors / abs_distances

    graph = Data(x=features, edge_index=edge_index, pos=pos, abs_distances=abs_distances.squeeze(), rel_vec=rel_vectors)
    output = model(graph)
    print(output)


def test_tensor_passing_layer_gradients():
    torch.set_default_dtype(torch.float64)

    representations = [[(15, 0, 0)], [(3, 0, 0), (4, 1, 0), (1, 2, 0)]]
    Rs_in = representations[0]
    Rs_out = representations[1]

    context = TensorPassingContext(representations)
    context = context.cuda()

    radial_model = partial(GaussianRadialModel, max_radius=3.2, min_radius=0.7, number_of_basis=10, h=100, L=3, act=swish)
    layer = TensorPassingLayer(Rs_in, Rs_out, context.named_buffers_pointer, radial_model)
    layer = layer.cuda()

    features = torch.randn(5, 15, device='cuda:0', requires_grad=True)
    pos = torch.randn(5, 3, device='cuda:0')
    edge_index = torch.tensor([[1, 2, 3, 4], [3, 1, 4, 0]], dtype=torch.int64, device='cuda:0')
    origin_pos = pos[edge_index[0]]
    neighbor_pos = pos[edge_index[1]]
    rel_vectors = neighbor_pos - origin_pos

    abs_distances = torch.norm(rel_vectors, 2, -1, keepdim=True)
    rel_vectors = rel_vectors / abs_distances

    inputs = (edge_index, features, abs_distances.squeeze(), rel_vectors)
    test = gradcheck(layer, inputs)
    assert test