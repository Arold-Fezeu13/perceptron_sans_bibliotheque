import math

# Définition de la classe Network
class Network:
    def __init__(self):
        self.layers = []   # Liste des couches du réseau
        self.loss = None   # Fonction de perte
        self.loss_prime = None   # Dérivée de la fonction de perte

    # Ajout d'une couche au réseau
    def add(self, layer):
        self.layers.append(layer)

    # Définition de la fonction de perte à utiliser
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Prédiction des sorties pour les données d'entrée données
    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)   # Propagation avant à travers les couches du réseau
            result.append(output)
        return result

    # Entraînement du réseau
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)   # Propagation avant pour obtenir les prédictions du réseau
                err += self.loss(y_train[j], output)   # Calcul de l'erreur de prédiction
                error = self.loss_prime(y_train[j], output)   # Calcul de l'erreur
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)   # Propagation arrière pour ajuster les poids du réseau en fonction de l'erreur
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))   # Affichage de l'erreur moyenne sur tous les échantillons d'entraînement pour l'epoch en cours
