# Classe de base (pour toutes les couches de réseaux de neurones layer=couche)
class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        self.output = None

    # calcule la sortie Y d'une couche pour une entrée X donnée
    def forward_propagation(self, input):
        raise NotImplementedError

    # calcule dE/dX pour une erreur de sortie dE/dY donnée (et met à jour les paramètres si nécessaire)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
