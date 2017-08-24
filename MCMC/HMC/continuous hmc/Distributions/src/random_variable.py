import collections
import numpy as np
import torch
from torch.autograd import Variable


class RandomVariable():
    """Base class for random variables. Supported methods:
        - sample(batch_size, num_particles)
        - sample_reparameterized(batch_size, num_particles)
        - logpdf(value, batch_size, num_particles)
    """

    def sample(self, batch_size, num_particles):
        """Returns a sample of this random variable."""

        raise NotImplementedError

    def sample_reparameterized(self, batch_size, num_particles):
        """Returns a reparameterized sample of this random variable."""

        raise NotImplementedError

    def pdf(self, value, batch_size, num_particles):
        """Evaluate the density of this random variable at a value. Returns
        Tensor/Variable [batch_size, num_particles].
        """

        raise NotImplementedError

    def logpdf(self, value, batch_size, num_particles):
        """Evaluate the log density of this random variable at a value. Returns
        Tensor/Variable [batch_size, num_particles].
        """

        raise NotImplementedError


class MultivariateIndependentLaplace(RandomVariable):
    """MultivariateIndependentLaplace random variable"""
    def __init__(self, location, scale):
        """Initialize this distribution with location, scale.

        input:
            location: Tensor/Variable
                [ dim_1, ..., dim_N]
            scale: Tensor/Variable
                [dim_1, ..., dim_N]
        """
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
    def sample(self):
        uniforms = torch.Tensor(self._location.size()).uniform_() - 0.5
        # why the half that would make the scale between [-0.5,0.5]
        if isinstance(self._location, Variable):
            uniforms = Variable(uniforms)
            return self._location.detach() - self._scale.detach() * \
                torch.sign(uniforms) * torch.log(1 - 2 * torch.abs(uniforms))
        else:
            return self._location - self._scale * torch
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


class MultivariateIndependentNormal(RandomVariable):
    """MultivariateIndependentNormal random variable"""
    def __init__(self, mean, variance):
        """Initialize this distribution with mean, variance.

        input:
            mean: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
            variance: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
        """
        assert(mean.size() == variance.size())
        assert(len(mean.size()) > 2)
        self._mean = mean
        self._variance = variance

    def sample(self, batch_size, num_particles):
        assert(list(self._mean.size()[:2]) == [batch_size, num_particles])

        uniform_normals = torch.Tensor(self._mean.size()).normal_()
        if isinstance(self._mean, Variable):
            return self._mean.detach() + \
                Variable(uniform_normals) * torch.sqrt(self._variance.detach())
        else:
            return uniform_normals * torch.sqrt(self._variance) + self._mean

    def sample_reparameterized(self, batch_size, num_particles):
        assert(list(self._mean.size()[:2]) == [batch_size, num_particles])

        standard_normal = MultivariateIndependentNormal(
            mean=Variable(torch.zeros(self._mean.size())),
            variance=Variable(torch.ones(self._variance.size()))
        )

        return self._mean + torch.sqrt(self._variance) * \
            standard_normal.sample(batch_size, num_particles)

    def pdf(self, value, batch_size, num_particles):
        assert(value.size() == self._mean.size())
        assert(list(self._mean.size()[:2]) == [batch_size, num_particles])

        return torch.prod(
            (
                1 / torch.sqrt(2 * self._variance * np.pi) * torch.exp(
                    -0.5 * (value - self._mean)**2 / self._variance
                )
            ).view(batch_size, num_particles, -1),
            dim=2
        ).squeeze(2)

    def logpdf(self, value, batch_size, num_particles):
        assert(value.size() == self._mean.size())
        assert(list(self._mean.size()[:2]) == [batch_size, num_particles])

        return torch.sum(
            (
                -0.5 * (value - self._mean)**2 / self._variance -
                0.5 * torch.log(2 * self._variance * np.pi)
            ).view(batch_size, num_particles, -1),
            dim=2
        ).squeeze(2)


class MultivariateIndependentPseudobernoulli(RandomVariable):
    """MultivariateIndependentPseudobernoulli random variable"""
    def __init__(self, probability):
        """Initialize this distribution with probability.

        input:
            probability: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
        """
        assert(len(probability.size()) > 2)
        self._probability = probability

    def sample(self, batch_size, num_particles):
        assert(
            list(self._probability.size()[:2]) == [batch_size, num_particles]
        )
        if isinstance(probability, Variable):
            return self._probability.detach()
        else:
            return self._probability

    def sample_reparameterized(self, batch_size, num_particles):
        assert(
            list(self._probability.size()[:2]) == [batch_size, num_particles]
        )

        return self._probability

    def pdf(self, value, batch_size, num_particles):
        assert(value.size() == self._probability.size())
        assert(
            list(self._probability.size()[:2]) == [batch_size, num_particles]
        )

        return torch.prod(
            (
                self._probability**value * (1 - self._probability)**(1 - value)
            ).view(batch_size, num_particles, -1),
            dim=2
        ).squeeze(2)

    def logpdf(self, value, batch_size, num_particles, epsilon=1e-10):
        assert(value.size() == self._probability.size())
        assert(
            list(self._probability.size()[:2]) == [batch_size, num_particles]
        )

        return torch.sum(
            (
                value * torch.log(self._probability + epsilon) +
                (1 - value) * torch.log(1 - self._probability + epsilon)
            ).view(batch_size, num_particles, -1),
            dim=2
        ).squeeze(2)


class Laplace(RandomVariable):
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


class Normal(RandomVariable):
    """Normal random variable"""
    def __init__(self, mean, variance):
        """Initialize this distribution with mean, variance.

        input:
            mean: Tensor/Variable [batch_size, num_particles]
            variance: Tensor/Variable [batch_size, num_particles]
        """
        assert(len(mean.size()) == 2)
        self._multivariate_independent_normal = MultivariateIndependentNormal(
            mean=mean.unsqueeze(-1),
            variance=variance.unsqueeze(-1)
        )

    def sample(self, batch_size, num_particles):
        return self._multivariate_independent_normal.sample(
            batch_size, num_particles
        ).squeeze(-1)

    def sample_reparameterized(self, batch_size, num_particles):
        return self._multivariate_independent_normal.sample_reparameterized(
            batch_size, num_particles
        ).squeeze(-1)

    def pdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_normal.pdf(
            value.unsqueeze(-1), batch_size, num_particles
        )

    def logpdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_normal.logpdf(
            value.unsqueeze(-1), batch_size, num_particles
        )


class Pseudobernoulli(RandomVariable):
    """Pseudobernoulli random variable"""
    def __init__(self, probability):
        """Initialize this distribution with probability.

        input:
            probability: Tensor/Variable [batch_size, num_particles]
        """
        assert(len(probability.size()) == 2)
        self._multivariate_independent_pseudobernoulli = \
            MultivariateIndependentPseudobernoulli(
                probability=probability.unsqueeze(-1)
            )

    def sample(self, batch_size, num_particles):
        return self._multivariate_independent_pseudobernoulli.sample(
            batch_size, num_particles
        ).squeeze(-1)

    def sample_reparameterized(self, batch_size, num_particles):
        return self._multivariate_independent_pseudobernoulli.\
            sample_reparameterized(batch_size, num_particles).squeeze(-1)

    def pdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_pseudobernoulli.pdf(
            value.unsqueeze(-1), batch_size, num_particles
        )

    def logpdf(self, value, batch_size, num_particles, epsilon=1e-10):
        return self._multivariate_independent_pseudobernoulli.logpdf(
            value.unsqueeze(-1), batch_size, num_particles, epsilon=epsilon
        )
'''
Probability distribution helpers.
Implemented:
    - Normal
    - Pseudobernoulli
    - Laplace
'''
import torch
import numpy as np
from utils.agnostic_tensor import *
from torch.autograd import Variable

## Normal distribution
def sample_normal(mean, var):
    '''
    returns a torch.FloatTensor / torch.cuda.FloatTensor of samples from Normal(mean, var)
    input:
        mean: Tensor/Variable [dim_1 * ... * dim_N]
        var: Tensor/Variable [dim_1 * ... * dim_N]
    output: Tensor/Variable [dim_1 * ... * dim_N]
    '''
    ret = Tensor(mean.size()).normal_()
    if isinstance(mean, Variable):
        ret = Variable(ret)
    ret = ret.mul(torch.sqrt(var)).add(mean)
    return ret

def normal_pdf(x, mean, var):
    '''
    returns normal pdfs
    input:
        x: Tensor/Variable [dim_1 * ... * dim_N]
        mean: Tensor/Variable [dim_1 * ... * dim_N]
        var: Tensor/Variable [dim_1 * ... * dim_N]
    output: Tensor/Variable [dim_1 * ... * dim_N]
    '''

    return 1 / torch.sqrt(2 * var * np.pi) * torch.exp(-0.5 * torch.pow(x - mean, 2) / var)

def normal_logpdf(x, mean, var):
    '''
    returns normal pdfs
    input:
        x: Tensor/Variable [dim_1 * ... * dim_N]
        mean: Tensor/Variable [dim_1 * ... * dim_N]
        var: Tensor/Variable [dim_1 * ... * dim_N]
    output: Tensor/Variable [dim_1 * ... * dim_N]
    '''

    return (-0.5 * torch.pow(x - mean, 2) / var - 0.5 * torch.log(2 * var * np.pi))

class Pseudobernoulli_pdf(RandomVariable):
    '''A class for the pseudobernoulli distrubtion
    methods:
    --------
    sample
    pdf
    log_pdf

    attributes
    ----------
    '''
    ## Pseudo Bernoulli distribution
    def sample_pseudobernoulli(prob):
        '''
        returns a Tensor
        input:
            prob: Tensor [dim_1 * ... * dim_N]
        output: Tensor [dim_1 * ... * dim_N]
        '''
        return prob

    def pseudobernoulli_pdf(x, prob):
        '''
        input:
            x: Tensor/Variable [dim_1 * ... * dim_N]
            prob: Tensor/Variable [dim_1 * ... * dim_N]
        output: Tensor/Variable [dim_1 * ... * dim_N]
        '''
        return prob**x * (1 - prob)**(1 - x)

    def pseudobernoulli_logpdf(x, prob, epsilon=1e-10):
        '''
        input:
            x: Tensor/Variable [dim_1 * ... * dim_N]
            prob: Tensor/Variable [dim_1 * ... * dim_N]
            epsilon: number. small value to prevent numerical instabilities (default 1e-10)
        output: Tensor/Variable [dim_1 * ... * dim_N]
        '''
        return (x * torch.log(prob + epsilon) + (1 - x) * torch.log(1 - prob + epsilon))

## Laplace distribution
## https://en.wikipedia.org/wiki/Laplace_distribution
def sample_laplace(location, scale):
    '''
    input:
        location: Tensor [dim_1 * ... * dim_N]
        scale: Tensor [dim_1 * ... * dim_N]
    output: Tensor [dim_1 * ... * dim_N]
    '''

    uniforms = Tensor(location.size()).uniform_() - 0.5
    return location - scale * torch.sign(uniforms) * torch.log(1 - 2 * torch.abs(uniforms))

def laplace_pdf(x, location, scale):
    '''
    returns Laplace pdfs
    input:
        x: Tensor/Variable [dim_1 * ... * dim_N]
        location: Tensor/Variable [dim_1 * ... * dim_N]
        scale: Tensor/Variable [dim_1 * ... * dim_N]
    output: Tensor/Variable [dim_1 * ... * dim_N]
    '''

    return torch.exp(-torch.abs(x - location) / scale) / (2 * b)

def laplace_logpdf(x, location, scale):
    '''
    returns Laplace logpdfs
    input:
        x: Tensor/Variable [dim_1 * ... * dim_N]
        location: Tensor/Variable [dim_1 * ... * dim_N]
        scale: Tensor/Variable [dim_1 * ... * dim_N]
    output: Tensor/Variable [dim_1 * ... * dim_N]
    '''

    return (-torch.abs(x - location) / scale - torch.log(2 * scale))
