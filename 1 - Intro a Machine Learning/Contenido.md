# Introduccion a Machine Learning

## Introduccion a las herramientras

### Numpy

Biblioteca de Python usualmente utilizada para la Ciencia de Datos

* Facil de usar
* Adecuada para los arreglos
* Agil

```python
#Resumen comandos
import numpy as np

#Crear arreglos
my_array = np.array([1, 2, 3, 4, 5])                # Resultado: array([1, 2, 3, 4, 5])
np.array( [[‘x’, ‘y’, ‘z’], [‘a’, ‘c’, ‘e’]])       # Resultado:      [[‘x’ ‘y’ ‘z’]
#                                                                     [‘a’ ‘c’ ‘e’]]
np.zeros(5)                                         # Resultado: array([0., 0., 0., 0., 0.])
np.ones(5)                                          # Resultado: array([1., 1., 1., 1., 1.])
np.arange(25)                                       # Resultado: array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
np.arange(5, 30)                                    # Resultado: array([ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
np.arange(5, 50, 5)                                 # Resultado: array([ 5, 10, 15, 20, 25, 30, 35, 40, 45])


# Operaciones
len(my_array)           #   Numero Elementos del arreglo
type(variable)          # Conocer tipo de dato
np.sum(my_array)        # Suma Unidimencional  
np.max(my_array)        # Numero Maximo del arreglo

# Manejo de Datos

np.order(x)             # Ordenar un arreglo (por default)
np.sort(arreglo, order = ‘llave’)      #Ordenar por la PRIMARY KEY
```

### Pandas

<span style="color:teal"> ver archivo Jupyter!! </span>

```python
import pandas as pd

# Arreglos

pd.Series([5, 10, 15, 20, 25])              # Serie Comun

lst = [‘Hola’, ‘mundo’, ‘robótico’]         # DataFrame
df = pd.DataFrame(lst)

data = {‘Nombre’:[‘Juan’, ‘Ana’, ‘Toño’, ‘Arturo’],     #Dataframe con llaves
‘Edad’:[25, 18, 23, 17],
‘Pais’: [‘MX’, ‘CO’, ‘BR’, ‘MX’] }
df = pd.DataFrame(data)

pd.read_csv(“archivo.csv”)                  # Lectura de Archivo

#Manejo de Datos
data.head(n)        # Mostrar n datos de head
data.tail()         # Mostrar ultimos datos
data.shape          # Mostrar Dimensiones del DATAFRAME

data.__columna__    # Mostrar los datos de una columna
data.columns        # Muestra toda las columnas
data[‘__columna__’].describe()      #Descripcion de columna

data.sort_index(axis = 0, ascending = False)                    # Ordenar Datos por id
data.sort_values(by="namae",axis = 0, ascending = False)        # Ordenar Datos por valor
```

### Scikit Learn

*Biblioteca de Python* que está conformada por algoritmos de clasificación, regresión, reducción de la dimensionalidad y clustering. Es una biblioteca clave en la aplicación de algoritmos de Machine Learning, tiene los métodos básicos para llamar un algoritmo, dividir los datos en entrenamiento y prueba, entrenarlo, predecir y ponerlo a prueba.

- Variedad de modulos
- Versatilidad
- Facilidad de uso

<span style="color:blue"> **Modelos relevantes**</span>

```python
from sklearn import tree # Modelo de arbol de desicion, sirve igual con regresion lineal
from sklearn import preprocessing # Simplemente para pre-procesar datos
from sklearn import train_test_split # Conjunto de entrenamiento y evaluacion
from sklearn import metrics # metricas necesarias para analizar la ejecucion de nuestros modelos

            # Ejemplo

# División del conjunto de datos para entrenamiento y pruebas:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Entrenar modelo:
[modelo].fit(X_train, y_train)

# Predicción del modelo:
Y_pred = [modelo].predict(X_test)

# Matriz de confusión:
metrics.confusion_matrix(y_test, y_pred)

# Calcular la exactitud:
metrics.accuracy_score(y_test, y_pred)
```

## Introduccion a Machine Learning

- Herramientas : `PyTorch` y `TensorFlow`
Subcategoría de ML que crea diferentes niveles, de abstracción que representa los datos. Se usan tensores para representar estructuras de datos más complejas.
- Neuronas: Capa de entrada, capas ocultas y capa de salida. Para poder aprender se necesita una función de activación (ReLU) 
- ReLU: Permite el paso de todos los valores positivos sin cambiarlos, pero asigna todos los valores negativos a 0.
- TensorFlow: Biblioteca de código abierto desarrollado por google, capaz de construir y entrenar redes neuronales.

```python
# Importar la biblioteca
import tensorflow as tf

# Importar el modelo
from tensorflow import keras

# Cargar conjunto de datos de Tensor Flow
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Crear modelo secuencial
model = keras.Sequential([keras.layers.Flatten(input_shape = (28, 28)), keras.layers.Dense(128, activation = tf.nn.relu), keras.layers.Dense(10, activation = tf.nn.softmax)])

#Compilación del modelo
model.compile(optimizer = tf.train.AdamOptimizer(), loss = ‘sparse_categorical_crossentropy’, metrics = [‘accuracy’])

# Entrenamiento
model.fit(train_images, train_labels, epochs = 5)

# Evaluación del modelo
test_loss, test_acc = model.evaluate( test_images, test_labels )

# Predicción del modelo
model.predict(test_images)
```
















https://platzi.com/clases/1708-fundamentos-ml/23044-introduccion-al-curso/

