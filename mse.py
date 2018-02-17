import torch.nn as nn


class MeanSquaredError(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False):
        super(MeanSquaredError, self).__init__()
        self.use_visibility = use_visibility

    def forward(self, *inputs):
        x, t = inputs
        diff = x - t
        N = diff.numel()/2
        diff = diff.view(-1)
        return diff.dot(diff)/N

def mean_squared_error(x, t, v, use_visibility=False):

    return MeanSquaredError()(x, t)
