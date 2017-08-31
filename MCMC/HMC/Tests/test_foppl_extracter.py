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
