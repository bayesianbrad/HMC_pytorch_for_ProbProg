import torch
import numpy as np
from torch.autograd import Variable
from z import *

a= VariableCast(1.0)
b= VariableCast(1.41)
normal_object = Normal(a, b)
x = Variable(torch.Tensor([2]),  requires_grad = True)
# x.detach()
# x = normal_object.sample()   #sample
logp_x = normal_object.logpdf(x)
# logp_x.backward(retain_graph = True)
# grad1  = x.grad.data
# print(grad1)
std   = VariableCast(1.73)
p_y_g_x = Normal(x, std)
obs2    = VariableCast(7.0)
# y22543 = obs2

logp_y_g_x = p_y_g_x.logpdf(obs2)
# logp_y_g_x.backward(retain_graph = True)
# grad2 = x.grad.data
# print(grad2)
# x.grad.data.zero_()
#
# print(logp_y_g_x.grad_fn)
logp_x_y = Variable.add(logp_x,logp_y_g_x)
print(logp_x_y)
# print(logp_x_y.grad_fn)
grad_x = torch.autograd.grad(outputs = [logp_x_y], inputs= [x])
print(grad_x)
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