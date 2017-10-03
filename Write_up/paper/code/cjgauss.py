c24039= torch.Tensor([1.0])
c24040= torch.Tensor([2.0])
# prior
d24041 = Normal(c24039, c24040)
# sample
x22542 = Variable(d24041.sample().data,\
  requires_grad = True)
# log prior
p24042 = x24041.logpdf( x22542)
c24043= torch.Tensor([3.0])
# likelihood
d24044 = Normal(x22542, c24043)
c24045= torch.Tensor([7.0])
# obs
y22543 = c24045
# log likelihood
p24046 = d24044.logpdf( y22543)
# log joint
p24047 = Variable.add(p24042,p24046)

return p24047, x22542