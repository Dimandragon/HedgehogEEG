import torch
import math

def periodicalFunctionEncodding(value, max_value, dim):
    out = torch.zeros(dim)
    n = max_value / math.pi
    #print(n, dim-2)
    for i in range(0, dim):
        if i % 2 == 0:
            out[i] = math.sin(value / n**(i/(dim-2)))
        else:
            out[i] = math.cos(value / n**((i-1)/(dim-2)))
    return out