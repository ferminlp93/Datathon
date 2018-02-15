#############
# LIBRERIAS #
#############

import numpy as np
import pandas as pd
import warnings
import collections
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, ElasticNetCV, ElasticNet, HuberRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


#############
# FUNCIONES #
#############

#RANGO INTERCUARTILICO
def iqr_calculate(column):
    description = column.describe()
    iqr = description['75%']-description['25%']
    return iqr

#LIMITES VALORES ATIPICOS
def outliers_limits(column):
    description = column.describe()
    iqr = iqr_calculate(column)
    outliers_limits = [description['25%']-1.5*iqr,description['75%']+1.5*iqr]
    return outliers_limits

#LIMITES VALORES ATIPICOS EXTREMOS
def extreme_outliers_limits(column):
    description = column.describe()
    iqr = iqr_calculate(column)
    outliers_limits = [description['25%']-3*iqr,description['75%']+3*iqr]
    return outliers_limits

#MOSTRAR HISTOGRAMA PARA UNA COLUMNA DADA
def mostrar_histograma(column, bins=100, label = None):
    print('Histograma:')
    fig, ax = plt.subplots()
    ax.hist(column, bins)
    ax.set_xlabel(label)
    plt.show()

#ELIMINA VALORES ATIPICOS
def delete_outliers(data, column):
    limits = outliers_limits(column)
    data_without_outliers = data[column<=limits[1]]
    data_without_outliers = data_without_outliers[column>=limits[0]]
    return data_without_outliers

#ELIMINA VALORES ATIPICOS EXTREMOS
def delete_extreme_outliers(data, column):
    limits = extreme_outliers_limits(column)
    data_without_extreme_outliers = data[column<=limits[1]]
    data_without_extreme_outliers = data_without_extreme_outliers[column>=limits[0]]
    return data_without_extreme_outliers
    
#MUESTRA DIAGRAMA DE CAJAS
def mostrar_diagrama_cajas(column):
    print('Diagrama de cajas:')
    plt.boxplot(column, vert=False)
    plt.show()

#MUESTRA LA MATRIZ DE CORRELACIONES
def mostrar_matriz_correlacion(data):
    print('Matriz de correlaciones:')
    correlation=data.corr() #obtain the correlation matrix
    sns.set()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.heatmap(correlation,ax=ax)
    plt.show()

#MUESTRA MATRIZ CON VARIABLES DE FUERTE CORRELACION
def mostrar_matriz_correlacion_fuerte(data, valor = 0.7):  
    print('Matriz de correlaciones de variables con correlacion superior a ', valor, ':')
    correlation=data.corr() #obtain the correlation matrix
    aux=(abs(correlation)-np.identity(correlation.shape[0])).max() #maximum correlation of each variable
    selected_feats=aux[aux>valor].index#take only variables whose maximum correlation is strong.
    sns.set()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.heatmap(correlation.loc[selected_feats,selected_feats],ax=ax,annot=True,fmt='.2f')
    plt.show()
    
#MUESTRA MATRIZ DE CORRELACION CON LAS N VARIABLES MAS FUERTES
def mostrar_matriz_correlacion_variables_fuertes(data, variables = 20):  
    print('Matriz de correlaciones de las ', variables, ' variables mas fuertes:')
    correlation=data.corr()
    aux=abs(correlation['Poder_Adquisitivo']).sort_values(ascending=False) #sort variables by their correlation with SalePrice
    selected_feats=aux[0:variables].index
    sns.set()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.heatmap(correlation.loc[selected_feats,selected_feats], annot=True,fmt='.2f',ax=ax)
    plt.show()

    
#MUESTRA LA MATRIZ DE DISPERSION
def mostrar_matriz_dispersion(column_a, column_b, label_a = None, label_b = None):
    fig, ax = plt.subplots()
    ax.scatter(column_a,column_b)
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    plt.show()

#MUESTRA MATRIZ DE DISPERSION CON VARIABLES DE FUERTE CORRELACION
def mostrar_matriz_dispersion_fuerte(data, valor = 0.7):
    print('Matrices de dispersion de variables con correlacion superior a ', valor, ':')
    correlation = data.corr()
    aux=abs(correlation)-np.identity(correlation.shape[0]).max()
    selected_feats=aux[aux>valor].index
    sns.set(style="ticks")
    sns.pairplot(data[selected_feats])
    plt.show()
    
#MUESTRA MATRIZ DE DISPERSION DE LAS N VARIABLES MAS FUERTES
def mostrar_matriz_dispersion_variables_fuertes(data, variables = 20):
    print('Matrices de disperion de las ', variables, ' variables mas fuertes:')
    correlation = data.corr()
    aux=abs(correlation['Poder_Adquisitivo']).sort_values(ascending=False)
    selected_feats=aux[1:variables+1].index #Seleccionamos las variables mas fuertes excluyendo PA
    sns.set(style="ticks")
    sns.pairplot(data[selected_feats])
    plt.show()

#SELECCIONA BLOQUE DE DATOS
def mostrar_histograma_bloque(bloque = 'Imp_Cons'):
    for col in numeric_cols:
        if(col.find(bloque) != -1):
           print(col)
           mostrar_histograma(traindata_without_outliers[col], 50)

#TRANSFORMACION LOGARITMICA
def transformacion_logaritmica(column):
    logarithmic_col = [np.log10(column)]
    return logarithmic_col

#TRANSFORMACION EXPONENCIAL
def transformacion_exponencial(column):
    exponential_col = [np.power(10,y) for y in column]
    return exponential_col

#CREA UNA VARIABLE RELACION
def crear_variable_relacion(column_a, column_b):
    new_column = [a / b for a,b in zip(column_a,column_b)]
    return new_column
    




