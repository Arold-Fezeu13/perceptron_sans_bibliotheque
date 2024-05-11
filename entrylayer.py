#class pour representer les neuronnes de la couche d'entrée 
from layer import Layer
import random 

# la couche entrée hérite de lz forme générale des couches du perceptron 

class entrylayer(Layer):

    # input_size = nombre de données entrées 
    # output_size = nombre de neuronnes de la couche  suivante

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)  # appel du constructeur de la classe mère
        self.output_size = output_size 
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(output_size)] for _ in range(input_size)]
        self.bias = [random.uniform(0, 0.5) for _ in range(output_size)]


    # calculer la sortie de chaque neuronne en fonction de l'entrée(Propagation dans le neuronne )
    def forward_propagation(self, input_data):
        self.input = input_data
        
        # Initialiser la liste des valeurs de la propagation d'une couche à l'autre 
        self.output = [0] * self.output_size
        
        #calculer la sortie pour chaque neuronne de la couche 
        for i in range(self.output_size):
            neuron_output = 0
            for j in range(self.input_size):
                neuron_output += input_data[j] * self.weights[j][i]
            # Ajout du biais 
            neuron_output += self.bias[i]
            self.output[i] = neuron_output
            
        return self.output


    # calcule dE/dW, dE/dB pour une erreur de sortie donnée=dE/dY. Renvoie input_error=dE/dX.(retropropagation du gradient)

    def backward_propagation(self, output_error, learning_rate):
        # Initialize input error
        input_error = [0] * self.input_size
        
        # Calculer l'erreur d'entrée
        for i in range(self.input_size):
            for j in range(self.output_size):
                input_error[i] += output_error[j] * self.weights[i][j]
        
        # mis à jour des poids 
        weights_error = [[0] * self.output_size for _ in range(self.input_size)]
        for i in range(self.input_size):
            for j in range(self.output_size):
                weights_error[i][j] = self.input[i] * output_error[j]
                self.weights[i][j] -= learning_rate * weights_error[i][j]
        
        # mis à jour des biais 
        for j in range(self.output_size):
            self.bias[j] -= learning_rate * output_error[j]
        
        return input_error
    
    def get_weights(self):
        return {'weights' :self.weights , 'biais':self.bias }

    def set_weights(self, loadata):
        self.weights = loadata['weights']
        self.bias = loadata['biais']
