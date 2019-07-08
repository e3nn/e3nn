# pylint: disable=C, R, arguments-differ, redefined-builtin
import torch
from se3cnn import SE3Kernel
from se3cnn.kernel import gaussian_window_wrapper
from se3cnn.point_utils import neighbor_difference_matrix, neighbor_feature_matrix


class SE3Convolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window=gaussian_window_wrapper, dyn_iso=False, verbose=False, **kwargs):
        super().__init__()

        self.kernel = SE3Kernel(Rs_in, Rs_out, size, radial_window=radial_window, dyn_iso=dyn_iso, verbose=verbose)
        self.kwargs = kwargs

    def __repr__(self):
        return "{name} ({kernel}, kwargs={kwargs})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
            kwargs=self.kwargs,
        )

    def forward(self, input):
        return torch.nn.functional.conv3d(input, self.kernel(), **self.kwargs)


class SE3ConvolutionTranspose(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window=gaussian_window_wrapper, dyn_iso=False, verbose=False, **kwargs):
        super().__init__()

        self.kernel = SE3Kernel(Rs_out, Rs_in, size, radial_window=radial_window, dyn_iso=dyn_iso, verbose=verbose)
        self.kwargs = kwargs

    def __repr__(self):
        return "{name} ({kernel}, kwargs={kwargs})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
            kwargs=self.kwargs,
        )

    def forward(self, input):
        return torch.nn.functional.conv_transpose3d(input, self.kernel(), **self.kwargs)


class SE3PointConvolution(torch.nn.Module):
    def __init__(self, kernel):
        super().__init__()

        self.kernel = kernel

    def __repr__(self):
        return "{name} ({kernel})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
        )

    def forward(self, input, difference_mat, relative_mask=None):
        """
        :param input: tensor [[batch], channel, points]
        :param difference_mat: tensor [[batch], points, points, xyz]
        :param relative_mask: [[batch], points, points]
        """

        kernel = self.kernel(difference_mat)

        if input.dim() == 2:
            # No batch dimension
            if relative_mask is not None:
                kernel = torch.einsum('ba,dcba->dcba', (relative_mask, kernel))
            output = torch.einsum('ca,dcba->db', (input, kernel))
        elif input.dim() == 3:
            # Batch dimension
            # Apply relative_mask to kernel (if examples are not all size N, M)
            if relative_mask is not None:
                kernel = torch.einsum('nba,dcnba->dcnba', (relative_mask, kernel))
            output = torch.einsum('nca,dcnba->ndb', (input, kernel))

        return output


class SE3PointNeighborConvolution(torch.nn.Module):
    def __init__(self, kernel):
        super().__init__()

        self.kernel = kernel

    def __repr__(self):
        return "{name} ({kernel})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
        )

    def forward(self, input, coords, neighbors, relative_mask=None):
        difference_matrix = neighbor_difference_matrix(neighbors, coords)  # [N, K, 3]
        neighbors_input = neighbor_feature_matrix(neighbors, input)  # [C, N, K]

        kernel = self.kernel(difference_matrix)

        if input.dim() == 2:
            # No batch dimension
            if relative_mask is not None:
                kernel = torch.einsum('ba,dcba->dcba', (relative_mask, kernel))
            output = torch.einsum('cba,dcba->db', (neighbors_input, kernel))
        elif input.dim() == 3:
            # Batch dimension
            # Apply relative_mask to kernel (if examples are not all size N, M)
            if relative_mask is not None:
                kernel = torch.einsum('nba,dcnba->dcnba', (relative_mask, kernel))
            output = torch.einsum('ncba,dcnba->ndb', (neighbors_input, kernel))

        return output


