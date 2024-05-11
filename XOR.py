import math
from perceptron import Network
from entrylayer import entrylayer
from activationlayer import ActivationLayer
from activate import tanh, tanh_prime
from perte import mse, mse_prime


x_train=[[0,0], [1,0] , [0,1] , [1,1]]
y_train=[[0] , [1] , [1]  , [0]]

net = Network()
net.add(entrylayer(2,3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(entrylayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train , y_train  , epochs=100000 , learning_rate=0.01)

out =net.predict(x_train)

print(out)