import torch
import numpy as np
from torch.autograd import Variable
import Distributions.distributions as dis
from core import VariableCast
class program():
    ''''This needs to be a function of all free variables.
         If I provide a map of all values and computes the log density
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
        logp = [] # empty list to store logps of each variable
        a = VariableCast(1.0)
        b = VariableCast(1.41)
        normal_object = dis.Normal(a, b)
        if values['x'] is not None:
            x = Variable(values['x'], requires_grad=True)

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

        # sum up all logs
        logp_x_y   = torch.zeros(1,1)
        for logprob in logp:
            logp_x_y = logp_x_y + logprob
        for k in values.keys():
            logp_x_y = logp_x_y + torch.log(values[k])

        return logp_x_y, {'x':x.data}
    def free_vars(self):
        return self.params

    # print(logp_x_y.grad_fn)
    # grad_x = torch.autograd.grad(outputs = [logp_x_y], inputs= [x])
    # print(grad_x)
# print(grad1 + grad2)
# print(x)
# print(logp_x_y)
#
# print("gradient: ", x.grad.data)


### tf code
# a= tf.constant(1.0)
# b= tf.constant(1.0)
# normal_object = Normal(mu=a, sigma=b)
# x = tf.Variable( normal_object.sample())   #sample
# p_x = normal_object.log_pdf( x) if normal_object.is_continuous else normal_object.log_pmf( x)   #prior
# obs1= tf.constant(1.0)
# p_y_g_x = Normal(mu=x, sigma=obs1)
# obs2= tf.constant(7.0)
# y22543 = obs2
# p24046 = p_y_g_x.log_pdf( y22543) if p_y_g_x.is_continuous else p_y_g_x.log_pmf( y22543) #obs, log likelihood
# p24047 = tf.add_n([p_x,p24046])
# # return E from the model
#
# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(p24047)
# # printing E:
# print(sess.run(x))
# writer = tf.summary.FileWriter( './Graph_Output/g24048', sess.graph)
# sess.close()

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
