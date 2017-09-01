import torch
import numpy as np
from torch.autograd import Variable
import Distributions.distributions as dis
from core import VariableCast
class program:
    ''''This needs to be a function of all free variables.
         If I provide a map of all values and computes the log density
         and assigns values rather than samples.
         If I don't provide then it samples
         For any variable values provided in the map, they are assigned

         method
         def eval

         Needs to be a class '''
    # def __init__(self):
    #     '''Generating code, returns  a map of variable names / symbols
    #      store all variables of interest / latent parameters in here.
    #       Strings -  A list of all the unique numbers of the para'''
    #     # self.params = [{'x' + Strings[i] : None} for i in range(len(Strings))]
    #     self.params  = {'x':None}

    def generate(self):
        ''' Generates the initial state and returns the samples and logjoint evaluated at initial samples  '''

        ################## Start FOPPL input ##########
        logp = [] # empty list to store logps of each variable
        a = VariableCast(1.0)
        b = VariableCast(1.41)
        normal_object = dis.Normal(a, b)
        x = normal_object.sample()

        std  = VariableCast(1.73)
        obs2 = VariableCast(7.0)
        p_y_g_x    = dis.Normal(x, std)

        # TO DO Ask Yuan, is it either possible to have once an '.logpdf' method is initiated can we do a
        # logp.append(<'variable upon which .logpdf method used'>)
        logp.append(normal_object.logpdf(x))
        logp.append(p_y_g_x.logpdf(obs2))
        # TO DO We will need to su m all the logs here.
        # Do I have them stored in a dictionary with the value
        # or do we have a separate thing for the logs?
        ################# End FOPPL output ############

        # sum up all logs
        logp_x_y   = torch.zeros(1,1)
        for logprob in logp:
            logp_x_y = logp_x_y + logprob
        return logp_x_y, x
    def eval(self, values, grad = False):
        ''' Takes a map of variable names, to variable values . This will be continually called
            within the leapfrog step

        values      -       Type: python dict object
                            Size: len(self.params)
                            Description: dictionary of 'parameters of interest'
        grad        -       Type: bool
                            Size: -
                            Description: Flag to denote whether the gradients are needed or not
        '''

        ################## Start FOPPL input ##########
        logp = [] # empty list to store logps of each variable
        a = VariableCast(1.0)
        b = VariableCast(1.41)
        normal_object = dis.Normal(a, b)

        std  = VariableCast(1.73)
        obs2 = VariableCast(7.0)
        # Need a better way of dealing with values. As ideally we have a dictionary (hash map)
        # then we say if values['x']
        p_y_g_x    = dis.Normal(values, std)

        logp.append(normal_object.logpdf(values))
        logp.append(p_y_g_x.logpdf(obs2))

        ################# End FOPPL output ############
        logjoint = torch.zeros(1, 1)

        for logprob in logp:
            logpjoint = logpjoint + logprob

        if grad:
            gradients = self.calc_grad(logjoint, values)
            return gradients, values
        else:
            return logjoint, values
    def free_vars(self):
        return self.params

    def calc_grad(self, logjoint, values):
        ''' Stores the gradients, grad, in a tensor, where each row corresponds to each the
            data from the Variable of the gradients '''
        # Assuming values is a dictionary we could extract the values into a list as follows
        if isinstance(dict, values):
            self.params = list(values.values())
        grad      = torch.autograd.grad([logjoint], self.params, grad_outputs= torch.ones(self.params[0].data))
        # note: Having grad_outputs set to the dimensions of the first element in the list, implies that we believe all
        # other values are the same size.
        gradients = torch.Tensor(len(self.params), self.params[0].data.size())
        for i in range(len(self.params)):
           gradients[i][:] = grad[i][0].data.unsqueeze(0) # ensures that each row of the grads represents a params grad
        return gradients


class programif():
    ''''This needs to be a function of all free variables.
         If I provide a map of all vlues and computes the log density
         and assigns values rather than samples.
         If I don't provide then it samples
         For any variable values provided in the map, they are assigned

         method
         def eval

         Needs to be a class '''
    def __init__(self):
        '''Generating code, returns  a map of variable names / symbols '''
        self.params = {'x': None}

    def eval(self, values):
        ''' Takes a map of variable names, to variable values '''
        a = VariableCast(0.0)
        b = VariableCast(1)
        normal_object = Normal(a, b)
        if values['x'] is not None:
            x = Variable(values['x'], requires_grad=True)
        # else:
        #     x = normal_object.sample()
        #     x = Variable(x.data, requires_grad = True)
        if torch.gt(x,torch.zeros(x.size()))[0][0]:
            yttyu

        logp_x = normal_object.logpdf(x)
        std = VariableCast(1.73)
        p_y_g_x = Normal(x, std)
        obs2 = VariableCast(7.0)
        logp_y_g_x = p_y_g_x.logpdf(obs2)
        logp_x_y = Variable.add(logp_x, logp_y_g_x)
        return logp_x_y, {'x':x.data}
    def free_vars(self):
        return self.params
