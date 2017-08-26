# from DistributionClass.agnostic_tensor import *
import numpy as np
from torch.autograd import Variable
import torch
from HMC.Distributions.src import distributions as dis

## testing Normal
mean = torch.Tensor([0.5]).unsqueeze(-1)
std  = torch.Tensor([0.6]).unsqueeze(-1)
normal_obj = dis.Normal(mean, std)
sample = normal_obj.sample(num_samples = 1)
# print(sample)
# print(normal_obj.logpdf(sample))
def true_grad_laplace(sample, loc, scale, diff):
    if diff == 'location':
        return torch.sign(sample - loc) / scale
    elif diff == 'sample':
        return torch.sign(sample - loc) / scale
    else:
        return -2/scale - torch.abs(sample- loc)


## testing Laplace
location = Variable(mean, requires_grad = True)
scale    = Variable(std)
laplace_obj     = dis.Laplace(location, scale)
sample          = laplace_obj.sample()
laplace_logpdf  = laplace_obj.logpdf(sample,location,scale)
diff_logpdf     = torch.autograd.grad([laplace_logpdf], [location])
print('Sample ', sample)
print('Laplace logpdf ', laplace_logpdf)
# print('Diffferential of sample',diff_sample
print('Differential of logpdf ', diff_logpdf)
print('True grad wrt to sample', true_grad_laplace(sample,location,scale,diff = 'location'))

# Testing categorical

# Testing Bernoulli
