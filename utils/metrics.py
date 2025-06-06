import numpy as np
from math import gamma

def levy_flight(dim, scale=1.0):
    """生成莱维飞行步长"""
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v) ** (1 / beta))
    
    return scale * step