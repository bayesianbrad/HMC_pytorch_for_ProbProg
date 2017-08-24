# from DistributionClass.agnostic_tensor import *
import numpy as np
from torch.autograd import Variable
import torch

torch.manual_seed(1234)
def VariableCast(value):
    if isinstance(value, torch.autograd.variable.Variable):
        return value
    elif torch.is_tensor(value):
        return Variable(value)
    else:
        return Variable(torch.Tensor([value]))

class MultivariateNormal():
    """MultivariateIndependentNormal simple class"""
    def __init__(self, mean, covariance):
        """Initialize this distribution with mean, covariance.

        input:
            mean: Tensor/Variable
                [ dim_1, ..., dim_N]
            covariance: Tensor/Variable
                covariance \in \mathbb{R}^{N \times N}
        """
        assert(mean.size()[0] == covariance.size()[0])
        assert (mean.size()[0] == covariance.size()[1])
        self._mean = mean
        self._covariance = covariance
        # cholesky decomposition returns upper triangular matrix. Will not accept Variables
        self._L = torch.potrf(self._covariance.data)
    def sample(self):
        # Returns a sample of a multivariate normal X ~ N(mean, cov)
        # A column vecotor of X ~ N(0,I)
        uniform_normals = torch.Tensor(self._mean.size()).normal_().t()

        if isinstance(self._mean, Variable):
            return self._mean.detach() + \
                Variable(self._L.t().mm(uniform_normals))
        else:
            return self._L.t().mm(uniform_normals) + self._mean

    def pdf(self, value):
        assert(value.size() == self._mean.size())
        # CAUTION: If the covariance is 'Unknown' then we will
        # not be returned the correct derivatives.
        value = VariableCast(value)
        # the sqrt root of a det(cov) : sqrt(det(cov)) == det(L.t()) = \Pi_{i=0}^{N} L_{ii}
        self._constant = torch.pow(2*np.pi,value.size()[1]) * self._L.t().diag().prod()
        return self._constant * torch.exp(-0.5*(value - self._mean).mm(self._L.inverse().mm(self._L.inverse().t())).mm((value - self._mean).t()))
        #     torch.prod(
        #     (
        #         1 / torch.sqrt(2 * self._variance * np.pi) * torch.exp(
        #             -0.5 * (value - self._mean)**2 / self._variance
        #         )
        #     ).view(-1),
        #     dim=0
        # ).squeeze(0)
    # squeeze doesn't do anything here, for our use.
    # view(-1), infers to change the structure of the
    # calculation, so it is transformed to a column vector
    # dim = 0, implies that we take the products all down the
    # rows

    def logpdf(self, value):
        assert(value.size() == self._mean.size())
        value = VariableCast(value)
        return torch.log(-0.5*(value - self._mean).mm(self._L.inverse().mm(self._L.inverse().t())).mm((value - self._mean).t())) \
        + self._constant


class Normal():
    """Normal random variable"""
    def __init__(self, mean, std):
        """Initialize this distribution with mean, variance.

        input:
            mean: Tensor/Variable [batch_size, num_particles]
            variance: Tensor/Variable [batch_size, num_particles]
        """
        self.mean = VariableCast(mean)
        self.std = VariableCast(std)


    def sample(self, num_samples = 1):
        # x = Variable(torch.randn(1), requires_grad = True)
        # #sample = Variable.add(Variable.mul(x, Variable.sqrt(self.variance), self.mean))
        # sample = x * torch.sqrt(self.variance) + self.mean
        # sample.retain_grad()

        x = torch.randn(1)
        sample = Variable(x * (self.std.data) + self.mean.data, requires_grad = True)
        return sample #.detach()


    def logpdf(self, value):
        mean = self.mean
        var = self.std**2
        value = VariableCast(value)

        # pdf: 1 / torch.sqrt(2 * var * np.pi) * torch.exp(-0.5 * torch.pow(value - mean, 2) / var)
        return (-0.5 *  torch.pow(value - mean, 2) / var - 0.5 * torch.log(2 * var * np.pi))

class MultivariateIndependentLaplace():
    """MultivariateIndependentLaplace random variable"""
    def __init__(self, location, scale):
        """Initialize this distribution with location, scale.

        input:
            location: Tensor/Variable
                [ dim_1, ..., dim_N]
            scale: Tensor/Variable
                [ dim_1, ..., dim_N]
        """
        assert(location.size() == scale.size())
        self._location = location
        self._scale = scale

    def sample(self, batch_size, num_particles):
        assert(list(self._location.size()[:2]) == [batch_size, num_particles])
        uniforms = torch.Tensor(self._location.size()).uniform_() - 0.5
        if isinstance(self._location, Variable):
            uniforms = Variable(uniforms)
            return self._location.detach() - self._scale.detach() * \
                torch.sign(uniforms) * torch.log(1 - 2 * torch.abs(uniforms))
        else:
            return self._location - self._scale * torch.sign(uniforms) * \
                torch.log(1 - 2 * torch.abs(uniforms))

    def sample_reparameterized(self, batch_size, num_particles):
        assert(list(self._location.size()[:2]) == [batch_size, num_particles])

        standard_laplace = MultivariateIndependentLaplace(
            location=Variable(torch.zeros(self._location.size())),
            scale=Variable(torch.ones(self._scale.size()))
        )

        return self._location + self._scale * standard_laplace.sample(
            batch_size, num_particles
        )

    def pdf(self, value, batch_size, num_particles):
        assert(value.size() == self._location.size())
        assert(list(self._location.size()[:2]) == [batch_size, num_particles])

        return torch.prod(
            (
                torch.exp(-torch.abs(value - self._location) / self._scale) /
                (2 * self._scale)
            ).view(batch_size, num_particles, -1),
            dim=2
        ).squeeze(2)

    def logpdf(self, value, batch_size, num_particles):
        assert(value.size() == self._location.size())
        assert(list(self._location.size()[:2]) == [batch_size, num_particles])

        return torch.sum(
            (
                -torch.abs(value - self._location) /
                self._scale - torch.log(2 * self._scale)
            ).view(batch_size, num_particles, -1),
            dim=2
        ).squeeze(2)

class Laplace():
    """Laplace random variable"""
    def __init__(self, location, scale):
        """Initialize this distribution with location, scale.

        input:
            location: Tensor/Variable [batch_size, num_particles]
            scale: Tensor/Variable [batch_size, num_particles]
        """
        assert(len(location.size()) == 2)
        self._multivariate_independent_laplace = MultivariateIndependentLaplace(
            location=location.unsqueeze(-1),
            scale=scale.unsqueeze(-1)
        )

    def sample(self, batch_size, num_particles):
        return self._multivariate_independent_laplace.sample(
            batch_size, num_particles
        )

    def sample_reparameterized(self, batch_size, num_particles):
        return self._multivariate_independent_laplace.sample_reparameterized(
            batch_size, num_particles
        )

    def pdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_laplace.pdf(
            value.unsqueeze(-1), batch_size, num_particles
        )

    def logpdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_laplace.logpdf(
            value.unsqueeze(-1), batch_size, num_particles
        )