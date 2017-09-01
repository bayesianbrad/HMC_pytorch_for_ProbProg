import numpy as np
import torch

def bernoulli_samples(n, p):
    x = 0
    k = 0

    for i in range(n-1):
        u = np.uniform(