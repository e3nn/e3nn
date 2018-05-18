import torch

# def get_mask(image_shape, index):
#     if torch.cuda.is_available():
#         mask = torch.cuda.ByteTensor(*image_shape).fill_(0)
#     else:
#         mask = torch.zeros(image_shape).byte()
#
#     for i in range(index.shape[0]):
#         if ((index[i, 1, :] - index[i, 0, :]) > 0).all():
#             mask[i,
#                  index[i, 0, 0]:index[i, 1, 0],
#                  index[i, 0, 1]:index[i, 1, 1],
#                  index[i, 0, 2]:index[i, 1, 2]] = 1
#     return mask