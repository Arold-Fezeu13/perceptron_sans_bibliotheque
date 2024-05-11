from layer import Layer

#representer le traitement des données au sein des couches cachées

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime): #activation=fonction d'activation et activation_prime sa dérivée
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        
        # Calcul des données de sortie de la couche cachée avec la fonction d'activation 
        self.output = [self.activation(x) for x in input_data]
        
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # Calculer input_error = dE/dX pour une output_error = dE/dY donnée
        input_error = [self.activation_prime(self.input[i]) * output_error[i] for i in range(len(output_error))]
        return input_error
