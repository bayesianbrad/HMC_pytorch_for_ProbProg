import torch
from torch.autograd import Variable


x = torch.Tensor([2])
y = torch.Tensor([4])
mean = torch.Tensor([0])
var = torch.Tensor([4])
vars = {'Gaussian':{'Parameters  of interest':[x, y], 'Additional params':[mean, var]}}

for k in vars.keys():
    for kk in vars[k]:
        if kk == 'Parameters of interest':
            for i in vars[kk][k]:
                vars[kk][k] = []
                if isistance(i, Variable):
                    temp = Variable(i.data, requires_grad = True)
                else:
                    temp  = Variable(i, requires_grad = True)
                var[kk][k].append(temp)
        else:
            print() # Deal with distribution parameters here

c24039= VariableCast(1.0)
c24040= VariableCast(2.0)
x24041 = Normal(c24039, c24040)
x22542 = Variable(torch.Tensor([0.0]),  requires_grad = True)
# x22542.detach()
# x22542 = x24041.sample()   #sample
p24042 = x24041.logpdf( x22542)
c24043 = VariableCast(3.0)
x24044 = Normal(x22542, c24043)
c24045 = VariableCast(7.0)
y22543 = c24045
p24046 = x24044.logpdf( y22543)
p24047 = Variable.add(p24042,p24046)

print(x22542)
print(p24047)
grad_x22542 = torch.autograd.grad([p24047], [x22542] )
print('gradient of x22542 ', x22542)