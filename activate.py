#fonction d'activation (tangente hyperbolique)
import math

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def tanh_prime(x):
    return 1 - tanh(x)**2
"""
def tanh(x):
    return 1/(1+math.exp(-x))

def tanh_prime(x):
    return tanh(x)*(1-tanh(x))

"""