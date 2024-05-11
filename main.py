import math
import pickle
from perceptron import Network
from entrylayer import entrylayer
from activationlayer import ActivationLayer
from activate import tanh, tanh_prime
from perte import mse, mse_prime
from normaliser import norm

# Données d'entraînement
#gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome


# Création du réseau

net = Network()
e=entrylayer(8, 5)
net.add(e)
net.add(ActivationLayer(tanh, tanh_prime))
e2=entrylayer(5, 3 )
net.add(e2)
net.add(ActivationLayer(tanh, tanh_prime))
e3=entrylayer(3, 1)
net.add(e3)
net.add(ActivationLayer(tanh, tanh_prime))
net.use(mse , mse_prime)

print("*******************MENU*********************\n\n"
      "1. Entrainer le modèle \n"
      "2. Tester le modèle \n"
      "3. Predire en fonction d'une entrée")

try:
    choix=int(input("\n Entrer le numero de votre choix : "))
except ValueError:
    print('entrer un entier')

match choix:
    case 1: 
        x_train=[]
        y_train=[]
        liste_vide=[]
        input_data=[]
        input_result=[]
        dataset=open("Diabetes.csv" , "r")
        for line in dataset:
            input_data=line.split(',')
            input_data[-1]=input_data[-1].strip()
            input_data= [float(data) for data in input_data]
            x_train.append(input_data[:-1])
            input_result=[input_data[-1]]
            y_train.append(input_result)
            line=dataset.readline
        dataset.close()
        x_train=norm(x_train)
        net.fit(x_train, y_train , epochs=100000  , learning_rate=0.01)

        with open('poids1', 'wb') as f:
            pickle.dump(e.get_weights(), f)

        with open('poids2', 'wb') as f2:
            pickle.dump(e2.get_weights(), f2)

        with open('poids3', 'wb') as f3:
            pickle.dump(e3.get_weights(), f3)

    case 2:
        x_test=[]
        liste_vide=[]
        input_data=[]
        dataset=open("test.csv" , "r")
        for line in dataset:
            input_data=line.split(',')
            input_data[-1]=input_data[-1].strip()
            input_data= [float(data) for data in input_data]
            x_test.append(input_data[:-1])
        dataset.close()
        x_test=norm(x_test)
        with open('poids3', 'rb') as f3:
            e3.set_weights(pickle.load(f3))

        with open('poids2', 'rb') as f2:
            e2.set_weights(pickle.load(f2))

        with open('poids1', 'rb') as f:
            e.set_weights(pickle.load(f))

        x_result=net.predict(x_test)
        for i in x_result:
            if i[0] > 0.9:
                print("personne atteinte de diabete " , i[0])
            elif i[0]<0.01:
                print("persone saine " , i[0])
    case 3:

        liste=[]

        try:
            liste.append(float(input("Entrer votre nombre de grossesses (0  si vous etes un hommes ) : ")))
        except ValueError: 
            print("vous devez entre un nombre  ")
        try:
            liste.append(float(input("Entrer le taux de glucose : ")))
        except ValueError: 
            print("vous devez entre un nombre ")
        try:
            liste.append(float(input("Entrer la tension arterielle :")))
        except ValueError: 
            print("vous devez entre un nombre ")
        try:
            liste.append(float(input("Entrer l'epaisseur de la peau  :")))
        except ValueError: 
            print("vous devez entre un nombre ")
        try:
            liste.append(float(input("Entrer le taux d'insullin : ")))
        except ValueError: 
            print("vous devez entre un nombre ")
        try:
            liste.append(float(input("Entrer l'indice de massse coporelle (IMC) : ")))
        except ValueError: 
            print("vous devez entre un nombre ")
        try:
            liste.append(float(input("Entrer le taux de presence de diabete dans votre pedigree : ")))
        except ValueError: 
            print("vous devez entre un nombre ")
        try:
            liste.append(float(input("Entrer l'age : ")))
        except ValueError: 
            print("vous devez entre un nombre ")

        with open('poids3', 'rb') as f3:
            e3.set_weights(pickle.load(f3))

        with open('poids2', 'rb') as f2:
            e2.set_weights(pickle.load(f2))

        with open('poids1', 'rb') as f:
            e.set_weights(pickle.load(f))

        liste=norm([liste])
        x_result=net.predict(liste)
        for i in x_result:
            if i[0] > 0.9:
                print("\npersonne atteinte de diabete " , i[0])
            elif i[0]<0.01:
                print("\npersone saine " , i[0])

