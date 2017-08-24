import torch
import numpy as np
from torch.autograd import Variable


def logpdf(value,mean,L):
    assert (value.size() == self._mean.size())
    value = VariableCast(value)
    return torch.log(
        -0.5 * (value - self._mean).mm(self._L.inverse().mm(self._L.inverse().t())).mm((value - self._mean).t())) \
           + self._constant