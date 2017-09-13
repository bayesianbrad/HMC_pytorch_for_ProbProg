c24039= Variable(torch.Tensor([1.0]))
c24040= Variable(torch.Tensor([2.0]))
x24041 = Normal(c24039, c24040)
x22542 = Variable(torch.Tensor([0.0]),\
  requires_grad = True)
# x22542.detach()
# x22542 = x24041.sample()   #sample
p24042 = x24041.logpdf( x22542)
c24043= Variable(torch.Tensor([3.0]))
x24044 = Normal(x22542, c24043)
c24045= Variable(torch.Tensor([7.0]))
y22543 = c24045
p24046 = x24044.logpdf( y22543)
p24047 = Variable.add(p24042,p24046)

return p24047, x22542