# ==============================================================================================
# Autor: Juan Carlos Varela Tellez
# Fecha de inicio: 09/09/2022
# Fecha de finalizacion: 10/09/2022
# ==============================================================================================
#
# ==============================================================================================
# En caso de no tener las bibliotecas necesarias, utilizar los siguientes comandos:
# python -m pip install numpy
# python -m pip install pandas
# python -m pip install seaborn
# python -m pip install matplotlib
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
# partir de ahi vamos a afinarlo hasta que cumpla nuestras expectativas con nuestros
# datos nuevos.
# ==============================================================================================

# Importamos librerias
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Cambio de configuracion en pandas
pd.set_option('display.max_rows', None)

# Ahora leeremos nuestro data-set nuevo
# Fuente: https://www.kaggle.com/datasets/uciml/mushroom-classification
mushroom_data = pd.read_csv("Data/mushrooms.csv")

# Primero necesitamos saber que contiene nuestro data-set y sus tipos de datos, asi que
# sacaremos diferentes metricas para mas adelante poder hacer decisiones mas
# informadas

print(mushroom_data.head())

print(mushroom_data.info())

# Nuestro data-set cuenta con 22 columnas, 21 siendo variables independientes y
# 1 siendo la variable independiente.
# Tambien es necesario ver la documentacion para saber que significan estos valores:
# https://github.com/JuanVaTe/RetoModulo2Analisis/blob/main/Data/documentation.txt
# Dentro de la documentacion y muestra de los datos, podemos observar que
# la columna 'veil-type' cuenta con unicamente un valor, asi que al preprocesarlo
# quitaremos esta carcteristica.

# Lo bueno: no hay valores nulos pero si valores no nulos que representen valores nulos
# (como un valor de '?', que es igual a no tener valor), asi que despues del analisis
# estadistico vamos a decidir si conservar la columna ya que son muchas caracteristicas
# o quitar las filas que tiene este dato invalido ya que tenemos muchas entradas.
# Lo malo: todas las caracteristicas son cualitativas y casi ninguna es binaria, lo cual
# va a ser un problema al cuantificar estos valores ya que nos vamos a quedar
# con muchas columnas.

# Para que no haya dudas, la variable dependiente indica si un champinon es venenoso
# o comestible

# Ya que no se tiene que hacer limpieza de datos, tenemos que empezar con el preprocesamiento
# directamente, asi que es tiempo de separar nuestras variables dependientes e independientes.

mushroom_x = mushroom_data.drop(['class'], axis=1)
mushroom_y = mushroom_data['class']

# Ya que nuestros datos estan separados, vamos a empezar a cuantificarlos, que es basicamente
# cuando reemplazas datos no numericos con datos numericos. Pandas cuenta con una funcion
# que nos va a ayudar con eso

# Cuantificando la variable dependiente
mushroom_y = pd.get_dummies(mushroom_y, drop_first=True)['p']

# Ahora toca cuantificar las variables indpendientes

# Cuantificamos cap-shape
dummy_cap_shape = pd.get_dummies(mushroom_x['cap-shape'], prefix='cap-shape')

# Cuantificamos cap-surface
dummy_cap_surface = pd.get_dummies(mushroom_x['cap-surface'], prefix='cap-surface')

# Cuantificamos cap-color
dummy_cap_color = pd.get_dummies(mushroom_x['cap-color'], prefix='cap-color')

# Cuantificamos bruises
dummy_bruises = pd.get_dummies(mushroom_x['bruises'], prefix='bruises')

# Cuantificamos odor
dummy_odor = pd.get_dummies(mushroom_x['odor'], prefix='odor')

# Cuantificamos gill-attachment
dummy_gill_attachment = pd.get_dummies(mushroom_x['gill-attachment'], prefix='gill-attachment')

# Cuantificamos gill-spacing
dummy_gill_spacing = pd.get_dummies(mushroom_x['gill-spacing'], prefix='gill-spacing')

# Cuantificamos gill-size
dummy_gill_size = pd.get_dummies(mushroom_x['gill-size'], prefix='gill-size')

# Cuantificamos gill-color
dummy_gill_color = pd.get_dummies(mushroom_x['gill-color'], prefix='gill-color')

# Cuantificamos stalk-shape
dummy_stalk_shape = pd.get_dummies(mushroom_x['stalk-shape'], prefix='stalk-shape')

# Cuantificamos stalk-root
dummy_stalk_root = pd.get_dummies(mushroom_x['stalk-root'], prefix='stalk-root')

# Cuantificamos stalk-surface-above-ring
dummy_stalk_surface_above_ring = pd.get_dummies(mushroom_x['stalk-surface-above-ring'], prefix='stalk-surface-above-ring')

# Cuantificamos stalk-surface-below-ring
dummy_stalk_surface_below_ring = pd.get_dummies(mushroom_x['stalk-surface-below-ring'], prefix='stalk-surface-below-ring')

# Cuantificamos stalk-color-above-ring
dummy_stalk_color_above_ring = pd.get_dummies(mushroom_x['stalk-color-above-ring'], prefix='stalk-color-above-ring')

# Cuantificamos stalk-color-below-ring
dummy_stalk_color_below_ring = pd.get_dummies(mushroom_x['stalk-color-below-ring'], prefix='stalk-color-below-ring')

# Cuantificamos veil-color
dummy_veil_color = pd.get_dummies(mushroom_x['veil-color'], prefix='veil-color')

# Cuantificamos ring-number
dummy_ring_number = pd.get_dummies(mushroom_x['ring-number'], prefix='ring-number')

# Cuantificamos ring-type
dummy_ring_type = pd.get_dummies(mushroom_x['ring-type'], prefix='ring-type')

# Cuantificamos spore-print-color
dummy_spore_print_color = pd.get_dummies(mushroom_x['spore-print-color'], prefix='spore-print-color')

# Cuantificamos population
dummy_population = pd.get_dummies(mushroom_x['population'], prefix='population')

# Cuantificamos habitat
dummy_habitat = pd.get_dummies(mushroom_x['habitat'], prefix='habitat')

# Concatenamos todos los dummies
mushroom_x_all = pd.concat([dummy_cap_shape, dummy_cap_surface, dummy_cap_color,
                            dummy_bruises,
                            dummy_odor,
                            dummy_gill_attachment, dummy_gill_spacing, dummy_gill_size, dummy_gill_color,
                            dummy_stalk_shape, dummy_stalk_root,
                            dummy_stalk_surface_above_ring, dummy_stalk_surface_below_ring,
                            dummy_stalk_color_above_ring, dummy_stalk_color_below_ring,
                            dummy_veil_color,
                            dummy_ring_number, dummy_ring_type,
                            dummy_spore_print_color,
                            dummy_population,
                            dummy_habitat], axis=1)

# Como se puede observar, quitamos la caracteristica de veil-type.

# Con los datos cuantificados, ahora podemos hacer nuestro analisis estadistico y encontrar
# correlaciones entre nuestras caracteristicas y nuestra variable dependiente.

correlation = pd.concat([mushroom_y, mushroom_x_all], axis=1).corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlation, annot=True)
plt.show()

# Debido a la monstruosidad de matriz de correlacion que obtuvimos, investigue en internet
# si es posible obtener los pares de valores con mayor correlacion.
# El siguiente pedazo de codigo no es mio aunque lo modifique para que se adaptara a este problema
# Creditos van para:
# HYRY: https://stackoverflow.com/users/772649/hyry (autor de la respuesta)
# Michel de Ruiter: https://stackoverflow.com/users/357313/michel-de-ruiter (mantuvo la respues actualizada)
# Link de la pregunta en StackOverflow:
# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas

c = correlation.abs()

s = c.unstack()
mayor_correlacion = s.sort_values(kind="quicksort")

# Hasta aqui acaba el codigo investigado

print("Correlaciones con la variable dependiente ==========")
print(mayor_correlacion[['p']])
print("====================================================\n")

# Podemos observar que hay una gran variedad de correlaciones que van desde 0.016 hasta 0.78, lo cual es una
# correlacion fuerte. Debido a la gran cantidad de correlaciones fuertes haremos lo siguiente.
# Para el preprocesamiento de datos, vamos a quedarnos con 2 data-sets de variables independientes:
# - Data-set con solamente las caracteristicas mas correlacionadas (0.5 o mas en indice de correlacion)
# - Data-set con todas las caracteristicas
# Como se hizo con el ejercicio anterior, esto nos va a dar mas espacio de experimentacion cuando empecemos
# a afinar nuestro modelo.

mushroom_x_corr = mushroom_x_all[['bruises_f', 'bruises_t', 'gill-color_b', 'gill-size_b', 'gill-size_n',
                                  'ring-type_p', 'stalk-surface-below-ring_k', 'stalk-surface-above-ring_k',
                                  'odor_f', 'odor_n']]

# Por ultimo toca escalar todos los datos para empezar a experimentar con los modelos

escalador_all = StandardScaler()
escalador_all.fit(mushroom_x_all)
mushroom_x_all_scaled = escalador_all.transform(mushroom_x_all)

escalador_corr = StandardScaler()
escalador_corr.fit(mushroom_x_corr)
mushroom_x_corr_scaled = escalador_corr.transform(mushroom_x_corr)

# Ahora toca la modularizacion de los datos

train_x_all, test_x_all, train_y_all, test_y_all = train_test_split(mushroom_x_all_scaled, mushroom_y, random_state=0)

train_x_corr, test_x_corr, train_y_corr, test_y_corr = train_test_split(mushroom_x_corr_scaled, mushroom_y, random_state=0)

# Empezemos probando el arbol de decision que fue preparado para el reto pasado
# (El alfa fue copiado de forma manual ya que al fin y al cabo, el mejor arbol
# de decision que se utilizo fue ese)

decision_tree_all = DecisionTreeClassifier(random_state=0,
                                           max_depth=12,
                                           ccp_alpha=0.00088584501076195)
decision_tree_all.fit(train_x_all, train_y_all)

print("Arbol de decision podado (todos los datos) ======================================")
print("Puntaje de entrenamiento:", decision_tree_all.score(train_x_all, train_y_all))
print("Puntaje de validacion:", decision_tree_all.score(test_x_all, test_y_all))
print("Alfa:", 0.00088584501076195)
print("=======================================================\n")

decision_tree_corr = DecisionTreeClassifier(random_state=0,
                                            max_depth=12,
                                            ccp_alpha=0.00088584501076195)
decision_tree_corr.fit(train_x_corr, train_y_corr)

print("Arbol de decision podado (datos correlacionados) ======================================")
print("Puntaje de entrenamiento:", decision_tree_corr.score(train_x_corr, train_y_corr))
print("Puntaje de validacion:", decision_tree_corr.score(test_x_corr, test_y_corr))
print("Alfa:", 0.00088584501076195)
print("=======================================================\n")

# Bueno, eso no lo esperaba... (genuinamente no tenia idea de que esto iba a pasar,
# me quede viendo la pantalla durante unos 5 minutos porque no lo creia...)
# Tener un modelo con un 100% de precision siempre es bienvenido, sin embargo,
# esta se supone que era la oportunidad para explicar porque aplicar siempre
# lo mismo a todos los problemas era malo, pero esto es un dato atipico.

# Hagamos esto, haremos un arbol de decision que tengo un puntaje muy malo y
# de ahi lo empezaremos a afinar...

decision_tree_2_all = DecisionTreeClassifier(random_state=0,
                                             ccp_alpha=10)
decision_tree_2_all.fit(train_x_all, train_y_all)

print("Arbol de decision podado (todos los datos) ======================================")
print("Puntaje de entrenamiento:", decision_tree_2_all.score(train_x_all, train_y_all))
print("Puntaje de validacion:", decision_tree_2_all.score(test_x_all, test_y_all))
print("=======================================================\n")

decision_tree_2_corr = DecisionTreeClassifier(random_state=0,
                                              ccp_alpha=10)
decision_tree_2_corr.fit(train_x_corr, train_y_corr)

print("Arbol de decision podado (datos correlacionados) ======================================")
print("Puntaje de entrenamiento:", decision_tree_2_corr.score(train_x_corr, train_y_corr))
print("Puntaje de validacion:", decision_tree_2_corr.score(test_x_corr, test_y_corr))
print("=======================================================\n")

# Este arbol de decision tiene un puntaje muy bajo, tanto en el entrenamiento
# como en la validacion. Esto es por 3 razones:
# - Sesgo alto: los valores predecidos estan muy lejos de los valores verdaderos
# - Varianza baja: el modelo no se mueve lo suficiente para poder predecir de forma correcta los valores
# - Underfitting: el modelo no es capaz de generalizar debido a su baja complejidad
# En realidad, estas 3 caracteristicas se deben a que el modelo es muy simple.
# Resolver esto es muy sencillo, solamente hay que aumentarle la complejidad al modelo

# ¿Por que nuestro modelo es simple?
# La razon principal es porque el valor alpha en un arbol de deicision le indica
# al modelo cuando "podar" ramas, lo que significa que el modelo no alcanza una
# convergencia porque no se le permitio crecer mas

# ¿Cuales son los hiperparametros que puedo utilizar para subir la complejidad del arbol
# de decision?
# Normalmente los arboles de decision tienden a ser tan complejos que datos nuevos que llegan
# para predecir los predice mal ya que el modelo se "memorizo" los datos de entrenamiento
# Para aumentar la complejidad de este modelo podemos dejar que crezca lo maximo que
# pueda quitandole el parametro 'ccp_alpha'
# De la misma forma, quitandole el hiperparametro 'max_depth' puede dejar que el arbol crezca
# lo que sea posible, lo cual tampoco es muy recomendable debido a los recursos
# computacionales que necesita
# Asimismo, quitandole el hiperparametro 'min_samples_leaf' puede ayudar a que su complejidad
# aumente ya que este parametro limita la generacion de una hoja si es que no hay suficientes
# datos en ese nodo, lo cual no deja generar el arbol completo

# En resumen, si ambos puntajes son bajos, indica underfitting con las 3 caracteristicas
# nombradas, asi que el plan de accion es aumentar la complejidad del modelo
# Si el puntaje de entrenamiento es muy alto y el de validacion muy bajo, esto
# indica overfitting, haciendo referencia a que el modelo "memoriza" los datos de entrenamiento
# y cualquier registro nuevo no lo puede predecir de manera correcta ya que va a tender a dar
# una respuesta del mismo modulo de entrenamiento.
# Cuando ocurre overfitting es necesario bajar la complejidad del modelo, asi como usar mas datos
# de entrenamiento para que pueda generalizar de mejor manera

# Dejar que se haga el arbol de decision completo
decision_tree_3_all = DecisionTreeClassifier(random_state=0)
decision_tree_3_all.fit(train_x_all, train_y_all)

print("Arbol de decision completo (todos los datos) ======================================")
print("Puntaje de entrenamiento:", decision_tree_3_all.score(train_x_all, train_y_all))
print("Puntaje de validacion:", decision_tree_3_all.score(test_x_all, test_y_all))
print("=======================================================\n")

decision_tree_3_corr = DecisionTreeClassifier(random_state=0)
decision_tree_3_corr.fit(train_x_corr, train_y_corr)

print("Arbol de decision completo (datos correlacionados) ======================================")
print("Puntaje de entrenamiento:", decision_tree_3_corr.score(train_x_corr, train_y_corr))
print("Puntaje de validacion:", decision_tree_3_corr.score(test_x_corr, test_y_corr))
print("=======================================================\n")

# En esta ocasion en especifico, nuestro modelo llego a un puntaje del 100% en ambos
# modulos, lo cual indica que es basicamente un modelo perfecto
# No hizo falta afinarlo mas, pero esto no significa que vaya a pasar siempre

# Para acabar, vamos a compararlo con nuestro "primer" modelo de arbol de decision
# utilizando metricas de rendimiento y la matriz de confusion

# Funcion para sacar metricas de rendimiento
def metricas_rendimiento(matriz_confusion):
    exactitud = (matriz_confusion[0][0] + matriz_confusion[1][1]) / (
                matriz_confusion[0][0] + matriz_confusion[0][1] + matriz_confusion[1][0] + matriz_confusion[1][1])

    try:
        precision = matriz_confusion[0][0] / (matriz_confusion[0][0] + matriz_confusion[1][0])
    except:
        precision = 0

    exhaustividad = matriz_confusion[0][0] / (matriz_confusion[0][0] + matriz_confusion[0][1])

    try:
        puntaje_F1 = (2 * precision * exhaustividad) / (precision + exhaustividad)
    except:
        puntaje_F1 = 0

    return exactitud, precision, exhaustividad, puntaje_F1

# Listado de modelos para ciclar la obtencion de metricas y matrices de confusion

models = [['Arbol de decision 1 (Todos los datos)', 'all', decision_tree_2_all],
          ['Arbol de decision 1 (Datos correlacionados)', 'corr', decision_tree_2_corr],
          ['Arbol de decision 2 (Todos los datos)', 'all', decision_tree_3_all],
          ['Arbol de decision 2 (Datos correlacionados)', 'corr', decision_tree_3_corr]]

for trio in models:
    if trio[1] == 'all':
        conf_matrix = confusion_matrix(test_y_all, trio[2].predict(test_x_all))
    else:
        conf_matrix = confusion_matrix(test_y_corr, trio[2].predict(test_x_corr))

    acc, prec, recall, F1_score = metricas_rendimiento(conf_matrix)

    print("=============================================")
    print(f"Metricas de rendimiento para modelo de {trio[0]}")
    print("Matriz de confusion:")
    print(conf_matrix)
    print(f"Exactitud     : {acc}")
    print(f"Precision     : {prec}")
    print(f"Exhaustividad : {recall}")
    print(f"Puntaje F1    : {F1_score}")
    print("=============================================\n")

# Aqui se puede observar que el mejor modelo sin duda es el Arbol de deicision 2
# utilizando el data-set completo
# Considerando que la gran mayoria de las variables independientes aportaban informacion
# importante debido a su alta correlacion promedio, es normal que al utilizar todas
# las caracteristicas se obtenga un resultado mas preciso.

# EXTRA: ¿Por que obtuvo un puntaje perfecto, incluso cuando al principio se estaba
# utilizando un modelo no optimizado para el problema?
# Lo siguiente que voy a explicar es simplemente una teoria que pude formular de forma
# empirica
# Cuando checamos nuestra lista de correlaciones con respecto a nuestra variable dependiente
# (queda recalcar que era si el hongo era comestible o no) note algo interesante,
# la gran mayoria de las caracteristicas (mas de la mitad) contaban con un indice de
# correlacion considerable (>0.2), lo cual indicaba que la mayor parte de los datos iban a dar
# informacion importante al modelo y no iban a generar ruido ni a "contaminar" al modelo final
# Pero ¿Por que fue esto?
# Estas caracteristicas eran en su mayoria colores, formas y el olor del hongo, claro
# ejemplo del fenomeno llamado 'aposematismo', lo cual se refiere a la advertencia de
# un ser vivo, ante cualquier depredador potencial, que comerlo no vale la pena. [1]
# Esto lo podemos comparar con las ranas venenosas. La gran mayoria de las ranas venenosas
# cuentan con colores brillantes, esto es porque le indican al depredador que no vale
# la pena cazarlos y al final llegan a un acuerdo tacito "si tiene colores brillantes,
# no lo comas"
# Esto tambien pasa con los hongos, si algun hongo tiene un olor fetido, es la forma de
# decir del hongo que no es comestible ya que "no vale la pena" pasar por el dolor de estomago
# que te va a ocasionar. Tambien pasa cuando el hongo tiene un color brillante.
# Es por esto que las correlaciones mas altas entre nuestra variable dependiente que se tuvieron
# vinieron de saber si el hongo tenia un olor fetido o no, asi como varias caracteristicas sobre
# los colores
# Esta es la razon por la cual, de todos los modelos, el arbol de decision fue el mejor en utilizarse
# ya que literalmente funciona a base de decisiones (como su nombre lo indica) y, tomando esas
# pocas decisiones con mucho peso, es capaz de clasificar de forma correcta y predecir incluso
# haciendo overfitting

# [1]- https://en.wikipedia.org/wiki/Aposematism
