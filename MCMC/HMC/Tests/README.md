x = 2
y = 4
mean = 0
var = 4
vars = {'Gaussian':{'Parameters  of interest':[x, y], 'Additional params':[mean, var]}}

for k in vars.keys():
    for kk in vars[k]:

        print(kk,vars[k][kk])