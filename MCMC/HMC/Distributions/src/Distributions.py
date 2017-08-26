import numpy as np
import torch
from torch.autograd import Variable
from core import ContinuousRandomVariable, DiscreteRandomVariable, VariableCast
torch.manual_seed(1234) # For testing only

# Note to user: Only the Laplace pdf is implemented as that is that is needed for this
# current project. The code is written for the pdfs but not
# ---------------------------------------------------------------------
# CONTINUOUS DISTRIBUTIONS
# ---------------------------------------------------------------------


class Laplace(ContinuousRandomVariable):
    """Laplace random variable

    methods
    -------
    sample
    sample_reprameterized *
    pdf
    logpdf"""
    def __init__(self, location, scale):
        """Initialize this distribution with location, scale.

        input:
            location: Tensor/Variable [batch_size, num_particles]
            scale: Tensor/Variable [batch_size, num_particles]
        """
        self._location = VariableCast(location)
        self._scale    = VariableCast(scale)
        self._multivariate_independent_laplace = MultivariateIndependentLaplace(
            location=self._location,
            scale=self._scale
        )

    def sample(self):
        return self._multivariate_independent_laplace.sample()
    # def sample_reparameterized(self, batch_size, num_particles):
    #     return self._multivariate_independent_laplace.sample_reparameterized(
    #         batch_size, num_particles)
    # def pdf(self, value):
    #     return self._multivariate_independent_laplace.pdf(
    #         value)
    def logpdf(self, value, loc, scale):
        return -torch.div(torch.abs(value - self._location ),self._scale) - torch.log(2 * self._scale)

class Normal(ContinuousRandomVariable):
    """Normal random variable
    Returns  a normal distribution object class

    methods
    --------
    sample   - returns a sample X ~ N(mean,std) as a Variable
    pdf
    logpdf
    """
    def __init__(self, mean, std):
        """Initialize this distribution with mean, variance.

        input:
            mean: Tensor/Variable [1 , ... , ndim]
            variance: Tensor/Variable [1, ... , ndim]
        """
        self._mean = VariableCast(mean)
        self._std = VariableCast(std)


    def sample(self, num_samples = 1):
        # x ~ N(0,1)

        x = torch.randn(num_samples)
        sample = Variable(x * (self._std.data) + self._mean.data, requires_grad = True)
        return sample #.detach()


    def logpdf(self, value):
        mean = self._mean
        var = self._std**2
        value = VariableCast(value)

        # pdf: 1 / torch.sqrt(2 * var * np.pi) * torch.exp(-0.5 * torch.pow(value - mean, 2) / var)
        return (-0.5 *  torch.pow(value - mean, 2) / var - 0.5 * torch.log(2 * var * np.pi))

class MultivariateIndependentLaplace(ContinuousRandomVariable):
    """MultivariateIndependentLaplace random variable"""
    def __init__(self, location, scale):
        """Initialize this distribution with location, scale.

        input:
            location: Tensor/Variable
                [ 1, ..., N]
            scale: Tensor/Variable
                [1, ..., N]
        """
        self._location = VariableCast(location)
        self._scale = VariableCast(scale)

    def sample(self):
        uniforms = torch.Tensor(self._location.size()).uniform_() - 0.5
        uniforms = VariableCast(uniforms)
        return self._location - self._scale * torch.sign(uniforms) * \
                torch.log(1 - 2 * torch.abs(uniforms))
    def sample(self):
        uniforms = torch.Tensor(self._location.size()).uniform_() - 0.5
        # why the half that would make the scale between [-0.5,0.5]
        uniforms = VariableCast(uniforms)
        return self._location - self._scale*torch.sign(uniforms) *\
                                    torch.log(1 - 2 * torch.abs(uniforms))
    def sample_reparameterized(self):

        standard_laplace = MultivariateIndependentLaplace(
            location=VariableCast(torch.zeros(self._location.size())),
            scale=VariableCast(torch.ones(self._scale.size()))
        )

        return self._location + self._scale * standard_laplace.sample()

class MultivariateNormal(ContinuousRandomVariable):
    """MultivariateIndependentNormal simple class"""

    def __init__(self, mean, covariance):
        """Initialize this distribution with mean, covariance.

        input:
            mean: Tensor/Variable
                [ dim_1, ..., dim_N]
            covariance: Tensor/Variable
                covariance \in \mathbb{R}^{N \times N}
        """
        assert (mean.size()[0] == covariance.size()[0])
        assert (mean.size()[0] == covariance.size()[1])
        self._mean = mean
        self._covariance = covariance
        # cholesky decomposition returns upper triangular matrix. Will not accept Variables
        self._L = torch.potrf(self._covariance.data)

    def sample(self):
        # Returns a sample of a multivariate normal X ~ N(mean, cov)
        # A column vecotor of X ~ N(0,I)  - IS THIS ACTUALLY TRUE ?? NOT SURE, ALL POINTS HAVE
        # THE RIGHT MEAN AND VARIANCE N(0,1)
        uniform_normals = torch.Tensor(self._mean.size()).normal_().t()

        if isinstance(self._mean, Variable):
            return self._mean.detach() + \
                   Variable(self._L.t().mm(uniform_normals))
        else:
            return self._L.t().mm(uniform_normals) + self._mean

    # def pdf(self, value):
    #     assert (value.size() == self._mean.size())
    #     # CAUTION: If the covariance is 'Unknown' then we will
    #     # not be returned the correct derivatives.
    #     print('****** Warning ******')
    #     print(' IF COVARIANCE IS UNKNOWN AND THE DERIVATIVES ARE NEEDED W.R.T IT, THIS RETURNED FUNCTION \n \
    #     WILL NOT RECORD THE GRAPH STRUCTURE OF THE FULL PASS, ONLY THE CALCUALTION OF THE PDF')
    #     value = VariableCast(value)
    #     # the sqrt root of a det(cov) : sqrt(det(cov)) == det(L.t()) = \Pi_{i=0}^{N} L_{ii}
    #     self._constant = torch.pow(2 * np.pi, value.size()[1]) * self._L.t().diag().prod()
    #     return self._constant * torch.exp(
    #         -0.5 * (value - self._mean).mm(self._L.inverse().mm(self._L.inverse().t())).mm(
    #             (value - self._mean).t()))

    def logpdf(self, value):
        print('****** Warning ******')
        print(' IF COVARIANCE IS UNKNOWN AND THE DERIVATIVES ARE NEEDED W.R.T IT, THIS RETURNED FUNCTION \n \
        WILL NOT RECORD THE GRAPH STRUCTURE OF THE FULL PASS, ONLY THE CALCUALTION OF THE LOGPDF')
        assert (value.size() == self._mean.size())
        value = VariableCast(value)
        return torch.log(-0.5 * (value - self._mean).mm(self._L.inverse().mm(self._L.inverse().t())).mm(
            (value - self._mean).t())) \
               + self._constant


# ---------------------------------------------------------------------
# DISCRETE DISTRIBUTIONS
# ---------------------------------------------------------------------
class Categorical(DiscreteRandomVariable):
    """
    Categorical over 0,...,N-1 with arbitrary probabilities, 1-dimensional rv, long type.
    """
    def __init__(self, p=None, p_min=1E-6, size=None):
        if size:
            assert len(size) == 2, str(size)
            p = VariableCast(1 / size[1])
        else:
            assert len(p.size()) == 2, str(p.size())
        assert torch.min(p.data) >= 0, str(torch.min(p.data))
        assert torch.max(torch.abs(torch.sum(p.data, 1) - 1)) <= 1E-5
        self._p = torch.clamp(p, p_min)
    def logpmf(self, value):
        return torch.log(self._p.gather(1, value)).squeeze()

    def sample(self):
        return self._p.multinomial(1, True)

    def entropy(self):
        return - torch.sum(self._p * torch.log(self._p), 1).squeeze()

class Pseudobernoulli(DiscreteRandomVariable):
    """Pseudobernoulli random variable"""

    def __init__(self, probability):
        """Initialize this distribution with probability.

        input:
        probability - Type: Float tensor
        """
        self._multivariate_independent_pseudobernoulli = \
            MultivariateIndependentPseudobernoulli(
                probability=probability.unsqueeze(-1)
            )

    def sample(self, batch_size, num_particles):
        return self._multivariate_independent_pseudobernoulli.sample()

    def sample_reparameterized(self, batch_size, num_particles):
        return self._multivariate_independent_pseudobernoulli. \
            sample_reparameterized().squeeze(-1)

    # def pmf(self, value, batch_size, num_particles):
    #     return self._multivariate_independent_pseudobernoulli.pdf(
    #         value.unsqueeze(-1), batch_size, num_particles
    #     )

    def logpmf(self, value, epsilon=1e-10):
        return self._multivariate_independent_pseudobernoulli.logpdf(
            value.unsqueeze(-1), epsilon=epsilon)


class MultivariateIndependentPseudobernoulli(DiscreteRandomVariable):
    """MultivariateIndependentPseudobernoulli random variable"""

    def __init__(self, probability):
        """Initialize this distribution with probability.

        input:
            probability: Tensor/Variable
                [1,..., N]
        """
        self._probability = probability

    def sample(self):
        return  VariableCast(self._probability)

    def sample_reparameterized(self):
        return VariableCast(self._probability)

    # def pmf(self, value):
    #     assert (value.size() == self._probability.size())
    #     value       = VariableCast(value)
    #     probability = VariableCast(self._probability)
    #     return torch.prod(
    #         (
    #             probability ** value * (1 - probability) ** (1 - value)), dim = 0)

    def logpmf(self, value, epsilon=1e-10):
        assert (value.size() == self._probability.size())
        value       = VariableCast(value)
        probability = VariableCast(self._probability)

        return torch.sum(
                value * torch.log(self._probability + epsilon) +
                (1 - value) * torch.log(1 - self._probability + epsilon))


# ---------------------------------------------------------------------------------------
# Unused and maybe used in the future
# ---------------------------------------------------------------------------------------
# class MultivariateIndependentLaplace(ContinuousRandomVariable):
#     """MultivariateIndependentLaplace random variable"""
#     def __init__(self, location, scale):
#         """Initialize this distribution with location, scale.
#
#         input:
#             location: Tensor/Variable
#                 [ 1, ..., N]
#             scale: Tensor/Variable
#                 [1, ..., N]
#         """
#         self._location = location
#         self._scale = scale
#
#     def sample(self, batch_size, num_particles):
#         uniforms = torch.Tensor(self._location.size()).uniform_() - 0.5
#         if isinstance(self._location, Variable):
#             uniforms = Variable(uniforms)
#             return self._location.detach() - self._scale.detach() * \
#                 torch.sign(uniforms) * torch.log(1 - 2 * torch.abs(uniforms))
#         else:
#             return self._location - self._scale * torch.sign(uniforms) * \
#                 torch.log(1 - 2 * torch.abs(uniforms))
#     def sample(self,num_samples):
#         uniforms = torch.Tensor(self._location.size()).uniform_() - 0.5
#         # why the half that would make the scale between [-0.5,0.5]
#         if isinstance(self._location, Variable):
#             uniforms = Variable(uniforms)
#             return self._location.detach() - self._scale.detach() * \
#                 torch.sign(uniforms) * torch.log(1 - 2 * torch.abs(uniforms))
#         else:
#             return self._location - self._scale * torch
#     def sample_reparameterized(self, num_samples):
#
#         standard_laplace = MultivariateIndependentLaplace(
#             location=VariableCast(torch.zeros(self._location.size())),
#             scale=VariableCast(torch.ones(self._scale.size()))
#         )
#
#         return self._location + self._scale * standard_laplace.sample(num_samples)
#         )
#
#     def pdf(self, value, batch_size, num_particles):
#         assert(value.size() == self._location.size())
#         assert(list(self._location.size()[:2]) == [batch_size, num_particles])
#
#         return torch.prod(
#             (
#                 torch.exp(-torch.abs(value - self._location) / self._scale) /
#                 (2 * self._scale)
#             ).view(batch_size, num_particles, -1),
#             dim=2
#         ).squeeze(2)
#
#     def logpdf(self, value, batch_size, num_particles):
#         assert(value.size() == self._location.size())
#         assert(list(self._location.size()[:2]) == [batch_size, num_particles])
#
#         return torch.sum(
#             (
#                 -torch.abs(value - self._location) /
#                 self._scale - torch.log(2 * self._scale)
#             ).view(batch_size, num_particles, -1),
#             dim=2
#         ).squeeze(2)

# class MultivariateNormal(ContinuousRandomVariable):
#     """MultivariateIndependentNormal simple class"""
#     def __init__(self, mean, covariance):
#         """Initialize this distribution with mean, covariance.
#
#         input:
#             mean: Tensor/Variable
#                 [ dim_1, ..., dim_N]
#             covariance: Tensor/Variable
#                 covariance \in \mathbb{R}^{N \times N}
#         """
#         assert(mean.size()[0] == covariance.size()[0])
#         assert (mean.size()[0] == covariance.size()[1])
#         self._mean = mean
#         self._covariance = covariance
#         # cholesky decomposition returns upper triangular matrix. Will not accept Variables
#         self._L = torch.potrf(self._covariance.data)
#     def sample(self):
#         # Returns a sample of a multivariate normal X ~ N(mean, cov)
#         # A column vecotor of X ~ N(0,I)
#         uniform_normals = torch.Tensor(self._mean.size()).normal_().t()
#
#         if isinstance(self._mean, Variable):
#             return self._mean.detach() + \
#                 Variable(self._L.t().mm(uniform_normals))
#         else:
#             return self._L.t().mm(uniform_normals) + self._mean
#
#     def pdf(self, value):
#         assert(value.size() == self._mean.size())
#         # CAUTION: If the covariance is 'Unknown' then we will
#         # not be returned the correct derivatives.
#         print('****** Warning ******')
#         print(' IF COVARIANCE IS UNKNOWN AND THE DERIVATIVES ARE NEEDED W.R.T IT, THIS RETURNED FUNCTION \n \
#         WILL NOT RECORD THE GRAPH STRUCTURE OF THE COVARIANCE' )
#         value = VariableCast(value)
#         # the sqrt root of a det(cov) : sqrt(det(cov)) == det(L.t()) = \Pi_{i=0}^{N} L_{ii}
#         self._constant = torch.pow(2*np.pi,value.size()[1]) * self._L.t().diag().prod()
#         return self._constant * torch.exp(-0.5*(value - self._mean).mm(self._L.inverse().mm(self._L.inverse().t())).mm((value - self._mean).t()))
#         #     torch.prod(
#         #     (
#         #         1 / torch.sqrt(2 * self._variance * np.pi) * torch.exp(
#         #             -0.5 * (value - self._mean)**2 / self._variance
#         #         )
#         #     ).view(-1),
#         #     dim=0
#         # ).squeeze(0)
#     # squeeze doesn't do anything here, for our use.
#     # view(-1), infers to change the structure of the
#     # calculation, so it is transformed to a column vector
#     # dim = 0, implies that we take the products all down the
#     # rows