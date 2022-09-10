# ==============================================================================================
# Autor: Juan Carlos Varela Tellez
# Fecha de inicio: 09/09/2022
# Fecha de finalizacion: 09/09/2022
# ==============================================================================================
#
# ==============================================================================================
# En caso de no tener las bibliotecas necesarias, utilizar los siguientes comandos:
# python -m pip install numpy
# python -m pip install pandas
# python -m pip install scikit-learn
# ==============================================================================================
#
# ==============================================================================================
# Este codigo es una continuacion indirecta del siguiente repositorio:
# https://github.com/JuanVaTe/RetoModulo2Framework
# Se recomienda leerlo antes de continuar, aunque no es necesario para entender
# este archivo
# ==============================================================================================
#
# ==============================================================================================
# Cuando se habla de un modelo de machine learning, en el ojo publico se piensa
# que es magia; un conjunto de comandos magicos donde se meten datos y salen mas datos,
# sin embargo, esto no es verdad. Es un conjunto de instrucciones estructuradas
# que se utilizan en una situacion en especifico, y es por esta razon que no todos
# los modelos se deben de utilizar en todas las situaciones. Hay un arte llamado
# "afinamiento de modelos" que es basicamente eso, afinar un modelo para que funcione
# con un conjunto de datos en especifico. Este concepto es lo que se va a mostrar
# en este codigo.
#
# Utilizaremos un modelo de arbol de decision predisenado por mi. El codigo completo
# lo puedes encontrar aqui: https://github.com/JuanVaTe/RetoModulo2Framework
# Este modelo se optimizo para poder predecir a las personas mas propensas a tener
# una apoplejia, sin embargo, vamos a ver que tan bien es su rendimiento cuando
# se utiliza el mismo modelo para una situacion completamente diferente, y a
# partir de ahi vamos a afinarlo hasta que cumpla nuestras espectativas con nuestros
# datos nuevos.
# ==============================================================================================

# Importamos librerias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Ahora leeremos nuestro data-set nuevo
mushroom_data = pd.read_csv("Data/mushrooms.csv")

# Primero necesitamos saber que contiene nuestro data-set y sus tipos de datos, asi que
# sacaremos diferentes metricas para mas adelante poder hacer decisiones mas
# informadas

print(mushroom_data.head())

print(mushroom_data.info())

# Nuestro data-set cuenta con 22 columnas, 21 siendo variables independientes y
# 1 siendo la variable independiente.
# Tambien es necesario ver la documentacion para saber que significan estos valores:
# 
