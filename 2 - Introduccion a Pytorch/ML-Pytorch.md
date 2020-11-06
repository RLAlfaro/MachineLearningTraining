# Introduccion a Machine Learning

## Conceptos

1. `Inteligencia Artificial:` La capacidad que puede tener una maquina para realizar las tareas de un ser humano, cumpliendolas al mismo nivel de eficiencia o incluso mejor.
2. `Machine Learning:` Subcampo de la Inteligencia artificial ( Heredado) en donde se programa a una máquina para hacer una acción en específico y solo para desempeñar esa acción. En este caso la maquina no tiene curva de aprendizaje ni libertad de ello, solo esta programada y diseñada para la tarea que se requiera en específico.
- **Aprendizaje supervisado:** La data necesita tener etiquetas en la data con el fin de poder emitir clasificaciones.
- **Aprendizaje no supervisado:** La data no tiene etiquetas, solo variables de entrada que permiten hacer grupos y de estos obtener algún tipo de información de los datos alojados.
- **Aprendizaje reforzado:** En donde se busca que nuestro ente de machine learning tome decisiones para maximizar el tipo de recompensa que se quiere obtener.
3. `Deep Learning:` Maneja redes neuronales y la complejidad es relativa depende de la complejidad de la acción o resultado que queremos obtener.
- **Redes Neuronales:** Entes que se interconectan (Redes con nodos) simulando el proceso neuronal del cerebro, en donde tenemos Inputs, Hiddens y Outputs. Este proceso no es exactamente igual al del cerebro pero se asemeja mucho.

### Otros conceptos

- `Modelo:` Define una relación entre features y labels. Usando las variables(es necesario normalizar, para tener representación numérica) construimos una ecuación para predecir.
- `Training:`  Darle un dataset al modelo y permite aprender de datos con label. Se deben hacer múltiples iteraciones para reducir la pérdida. Debemos reducir éste loss Para reducir el loss debemos escoger el peso correcto para cada label.
- `Inference:` Usar el modelo para realizar predicciones.

## Introduccion a Pytorch

`Instalacion` https://pytorch.org/get-started/locally/#with-cuda
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

## Algoritmos más utilizados

### Supervised Learning

1. Regresion:  Regresion Lineal, Regresion Logistica, Naive Bayes

2. Clasificacion: K-Neighbors, Decision Trees, Random Forest

### Unsupervised Learning
1. Cluster/agrupamiento: K-means,
- Segmentacion de mercado, Sistema de recomendacion
2. Reduccion de Dimension: 
- Visualizacion de BigData y Descubrimiento de estructuras

### Reinforcement Learning

 

## Data a trabajar

Espero que si alguien tuvo alguna duda con esto también le pueda ayudar.

data[:h, :] es una notación usada en numpy para manipular los arreglos sin
tener que separarlos.

El arreglo tiene la forma siguiente donde los puntos en X están alojados en el las posiciones [n][0] y los puntos Y están en [n][1]

[ [x1, y1],
[x2, y2],
[x3, y3]
…
[xn, yn] ]

dado que se necesitan 2 nubes separadas entonces hay que asignar una separación del primer punto a la mitad, entonces
data[ : h, : ]
es la notación para decir: ve de inicio a h en la coordenada n y ve de inicio a fin en la coordenada m.
esto hace que por cada punto X,Y se le asigne la operación de multiplicar por 3 en el caso de este ejercicio y para la siguiente mitad:
data[h : , : ]
es la notación para decir: ve de la mitad h al final en la coordenada n y ve de inicio a fin en la coordenada m

cuando se hace el plt.scatter() hay que darle un arreglo en x y un arreglo en y por lo que
data[ : , 0]
dice: ve de inicio a fin en la coordenada n tomando solo el primer elemento de la coordenada m, haciendo lo mismo para Y
data[ : , 1]
dice: ve de inicio a fin en la coordenada n tomando solo el segundo elemento de la coordenada m,

``` python
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def _get_data(pares_de_datos):
    mitad_de_pares = pares_de_datos//2
    dimen = 2
    return np.random.randn(pares_de_datos, dimen)*3, mitad_de_pares, dimen


def _grafics(labelx, labely, color, s, alpha):
    if plt.scatter(labelx, labely, c=color, s=s, alpha=alpha):
        print('Grafics [OK]')
    else:
        print('Grafics [FAIL]')

def _grafics_with_plot(labelx, labely):
    if plt.plot(labelx, labely):
        print('Grafics [OK]')
    else:
        print('Grafics [FAIL]')


def training_losses(inpt, outpt, model, loss_function, optimizer, losses, iterations):
    for i in range(iterations):
        result = model(inpt)
        loss = loss_function(result, outpt)
        losses.append(loss.data)

        optimizer.zero_grad()
        loss.backward()                                                                  # Back Propagation
        optimizer.step()
    print('last loss: {}.'.format(float(loss)))
    _grafics_with_plot(range(iterations), losses)
    return model



def create_model(inpt, outpt):
    model = nn.Sequential(nn.Linear(2,1), nn.Sigmoid())
    loss_function = nn.BCELoss() # BCE es el Binary cross entropy
    optimizer = optim.SGD(model.parameters(), lr=0.015)
    model = training_losses(inpt, outpt, model, loss_function, optimizer, losses=[], iterations=2000)
    return model


def main(data, mitad_de_pares, pares_de_datos, colors):
    target = np.array([0]*mitad_de_pares + [1]*mitad_de_pares).reshape(pares_de_datos,1)
    inpt = torch.from_numpy(data).float().requires_grad_()
    outpt = torch.from_numpy(target).float()
    model = create_model(inpt, outpt)
    input_data = torch.Tensor([[-5, -6]])
    prediction = model(input_data).data[0][0] > 0.5
    print('La probabilidad sera de: {}, por lo tanto la es de color: {}.'.format(prediction.data, colors[prediction]))

if __name__ == '__main__':
    pares_de_datos = 100
    data, mitad_de_pares, dimen = _get_data(pares_de_datos)
    colors = ['blue', 'red']
    color = np.array([colors[0]]*mitad_de_pares + [colors[1]]*mitad_de_pares).reshape(pares_de_datos)
    _grafics(data[:,0], data[:,1], color, s=75, alpha=0.6)
    data[:mitad_de_pares, :] -= 3*np.ones((mitad_de_pares, dimen))
    data[mitad_de_pares:, :] += 3*np.ones((mitad_de_pares, dimen))
    _grafics(data[:,0], data[:,1], color, s=75, alpha=0.6)
    main(data, mitad_de_pares, pares_de_datos, colors)
```

Gradiente: es un vector, tiene dirección y magnitud y se calcula con una derivada parcial.
**Derivada parcial: **encuentra un vector con dirección, la dirección a la cual deberíamos ir. Ésta derivada se hace con respecto a múltiples pesos. Proceso iterativo hasta llegar al punto mínimo
Learning rate: Me dice que tan grandes o pequeños son los pasos que se dan con cada iteración en la derivada parcial. Hay que definir el LR de forma que se llegue al mínimo eficientemente.
SGD: Stochastic Gradient Descent : Aleatorio, Se trabaja con un batch, es un grupo de nuestro dataset, se calculan los gradientes. Se busca que el aprendizaje de nuestro modelo minimiza el porcentaje de pérdidas.

