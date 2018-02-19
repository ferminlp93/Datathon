#############
# LIBRERIAS #
#############

import numpy as np
import pandas as pd
import warnings
import os
from sklearn.cluster import MiniBatchKMeans
from Utilidades import *


"""
|-----------------------------------|
|FUNCIONES PARA LIMPIEZA DE DATOS|
|-----------------------------------|
"""
#Agrupa por columnas, saca la media de la columna target para cada grupo y ordena dichos grupos segun la media obtenida
def map_order_col_by_target(X,col,target):
    
    grouped_by_col=X.groupby([col])[target].mean().sort_values()
    mapping={}
    for i, index in enumerate(grouped_by_col.index):
        mapping[index]=i

    return mapping

#Cantidad de valores faltantes de una o mas (o todas las) columnas
def n_missings(X,cols=''):
    if cols:
        return X[cols].isnull().sum() #solo algunas columnas
    else: 
        return X.isnull().sum() #todas las columnas

    
#GENERA LOS VALORES FALTANTES DE FORMA ALEATORIA PONDERADA POR LAS FRECUENCIAS RELATIVAS
#ES DECIR, CADA VALOR FALTANTE ES MAS PROBABLE QUE SE RELLENE CON LOS MAS FRECUENTES, 
#ASI SE MANTIENE LA DISTRIBUCION DE PROBABILIDAD ORIGINAL INTACTA    
def rellenar_faltantes_random_ponderado(X,col):
    #OBTENEMOS LA FRECUENCIA RELATIVA DE CADA VALOR
    val_count=X[col].value_counts()/X.shape[0]
    
    #A LA QUE TIENE MAS FRECUENCIA LE AÑADIMOS LO QUE FALTA PARA LLEGAR A 1(EN LA DIVISION SE PIERDEN DECIMALES Y FALTABA 0.0001 O ASI)
    val_count.iloc[0]+=1-val_count.values.sum()
    
    n_col_nulls=X[col].isnull().sum()
    return np.random.choice(val_count.index,n_col_nulls,p=val_count.values)

def cargar_originales(train,test):
    print('Cargando los datasets originales...')
    traindata=pd.read_csv(train)#reading the data
    testdata=pd.read_csv(test)
    return traindata,testdata

def crear_variables(traindata,testdata):
    print('Creando Variables para enriquecer el modelo...')
    #Ver el nombre de las columnas para saber cuales sumar (buscar la mayor correlación)
    list_columns=list(traindata.columns.values)
    Names_Imp_cons=list_columns[0:17]
    #Importe de consumos habituales del cliente en base a sus operaciones con tarjetas y domiciliaciones más comunes
    traindata['Imp_Cons_total'] = traindata[Names_Imp_cons].sum(axis=1)
    #Importe de los saldos de los distintos productos financieros.
    traindata['Imp_Cons_total_2']=traindata['Imp_Cons_total']**2
    traindata['Imp_Cons_total_3']=traindata['Imp_Cons_total']**3
    Names_Imp_Sal=list_columns[17:38]
    traindata['Imp_Sal_total'] = traindata[Names_Imp_Sal].sum(axis=1)
    #Tenencia de los distintos productos financieros. Son indices entre 0 y 2 por lo que el sumatorio también valdra
    #ya que cuanta más puntuación saque el sumatorio más productos compra
    traindata['Imp_Sal_total_2']=traindata['Imp_Sal_total']**2
    traindata['Imp_Sal_total_3']=traindata['Imp_Sal_total']**3
    Names_Ind_Prod=list_columns[38:62]
    traindata['Ind_Prod_total'] = traindata[Names_Ind_Prod].sum(axis=1)
    #Número de operaciones a través de los distintos productos financieros.
    traindata['Ind_Prod_total_2'] = traindata['Ind_Prod_total']**2
    traindata['Ind_Prod_total_3'] = traindata['Ind_Prod_total']**3
    Names_Num_Oper=list_columns[62:82]
    traindata['Num_Oper_total'] = traindata[Names_Num_Oper].sum(axis=1)

    traindata['Num_Oper_total_2'] = traindata['Num_Oper_total']**2
    traindata['Num_Oper_total_3'] = traindata['Num_Oper_total']**3
    #datasetfinal
    #Se intenta buscar una relación entre el salario consumo y el numero de operaciones
    traindata['Relacion']=((traindata.Imp_Sal_total+traindata.Ind_Prod_total)*traindata.Num_Oper_total)-traindata.Imp_Cons_total
    

    traindata['Relacion_2'] = traindata['Relacion']**2
    traindata['Relacion_3'] = traindata['Relacion']**3
    #hacemos lo mismo con lo de test (la lista que contiene nombres nos sirve la misma)

    testdata['Imp_Cons_total'] = testdata[Names_Imp_cons].sum(axis=1)
    testdata['Imp_Cons_total_2'] = testdata['Imp_Cons_total']**2
    testdata['Imp_Cons_total_3'] = testdata['Imp_Cons_total']**3
    testdata['Imp_Sal_total'] = testdata[Names_Imp_Sal].sum(axis=1)
    testdata['Imp_Sal_total_2'] = testdata['Imp_Sal_total']**2
    testdata['Imp_Sal_total_3'] = testdata['Imp_Sal_total']**3
    testdata['Ind_Prod_total'] = testdata[Names_Ind_Prod].sum(axis=1)
    testdata['Ind_Prod_total_2'] = testdata['Ind_Prod_total']**2
    testdata['Ind_Prod_total_3'] = testdata['Ind_Prod_total']**3
    testdata['Num_Oper_total'] = testdata[Names_Num_Oper].sum(axis=1)
    testdata['Num_Oper_total_2'] = testdata['Num_Oper_total']**2
    testdata['Num_Oper_total_3'] = testdata['Num_Oper_total']**3
    testdata['Relacion']=((testdata.Imp_Sal_total+testdata.Ind_Prod_total)*testdata.Num_Oper_total)-testdata.Imp_Cons_total
    testdata['Relacion_2']= testdata['Relacion']**2
    testdata['Relacion_3']= testdata['Relacion']**3
    
def concat_data_trans(traindata,testdata):
    print('Concatenando los datasets para el procesado...')
    numeric_cols=traindata.select_dtypes(include=[np.number]).columns#select only numerical
    nominal_cols=traindata.select_dtypes(exclude=[np.number]).columns#select only non numerical
    

    #CONCATENAMOS AMBOS CONJUNTOS PARA APLICAR TRANSFORMACIONES
    #Sacamos columna para el clasificador

    data=pd.concat([traindata,testdata],axis=0,ignore_index=True) #concatenate training and test set for future transformat
    return data,numeric_cols,nominal_cols

def cat_to_dummy(data):
    print('Pasando categoricas a dummies...')
    categorical_cols=['Ind_Prod_02', 'Ind_Prod_03',
       'Ind_Prod_04', 'Ind_Prod_05', 'Ind_Prod_06', 'Ind_Prod_07',
                      'Ind_Prod_09', 'Ind_Prod_10', 'Ind_Prod_11',
       'Ind_Prod_12', 'Ind_Prod_13', 'Ind_Prod_14', 'Ind_Prod_15',
       'Ind_Prod_16', 'Ind_Prod_18', 'Ind_Prod_19',
       'Ind_Prod_20', 'Ind_Prod_21', 'Ind_Prod_22', 'Ind_Prod_23',
       'Ind_Prod_24']
    
    return pd.get_dummies(data,columns=categorical_cols)


def socio_demo(data):
    print('Tratando la columna Socio_Demo_01...')
    '''
    TRATAMIENTO DE LA COLUMNA Socio_Demo_01
        -conversion a variable numerica con mapeo para que los valores de Socio_Demo_01 vayan 
            de menor a mayor media de poder adquisitivo
        -imputacion de valores faltantes
        -conversion a variable entera
    '''
    
    #CONVERTIMOS TODOS LOS VALORES EN NUMEROS 
    mapping=map_order_col_by_target(data,'Socio_Demo_01','Poder_Adquisitivo')
    data['Socio_Demo_01']=data['Socio_Demo_01'].map(mapping)
    
    #RELLENAMOS LOS VALORES FALTANTES DE Socio_Demo_01
    data.loc[data['Socio_Demo_01'].isnull(),'Socio_Demo_01']=rellenar_faltantes_random_ponderado(data,'Socio_Demo_01')
    
    #ESTABA TODAVIA EN FLOAT, LO PASAMOS A INT
    data['Socio_Demo_01']=data['Socio_Demo_01'].astype('int')
    
    '''
    FIN DE TRATAMIENTO DE Socio_Demo_01
    
    
    CREACION DE LA COLUMNA CLUSTER
        -agrupamos las muestras en 5 clusters y creamos la columna cluster:
            muy pobres, pobres, intermedios, ricos, muy ricos
        -Mapeamos la columna cluster para que los clusters vayan de menor a mayor media de poder adquisitivo
            (Siendo 0 muy pobre y 4 muy rico)
    '''
    return data

def create_cluster(data):
    print('Creando columna de cluster...')
    '''
     CREACION DE LA COLUMNA CLUSTER
        -agrupamos las muestras en 5 clusters y creamos la columna cluster:
            muy pobres, pobres, intermedios, ricos, muy ricos
        -Mapeamos la columna cluster para que los clusters vayan de menor a mayor media de poder adquisitivo
            (Siendo 0 muy pobre y 4 muy rico)
    '''
    kmeans=MiniBatchKMeans(5)
    data['cluster']=kmeans.fit_predict(data.drop(['Poder_Adquisitivo','ID_Customer'],axis=1))
    mapping=map_order_col_by_target(data,'cluster','Poder_Adquisitivo')
    data['cluster']=data['cluster'].map(mapping)
    '''
    FIN DE CREACION DE LA COLUMNA CLUSTER
    '''
    return data

def norm_and_sep(data,traindata,testdata,numeric_cols):
    print('Separando los datasets...')
    #Normalizar dataset
    #stdSc = StandardScaler()

    #numeric_cols=numeric_cols[numeric_cols!='Poder_Adquisitivo'] #We don't want to scale SalePrice

    #data.loc[:, numeric_cols] = stdSc.fit_transform(data.loc[:, numeric_cols])

    #SEPARAMOS DE NUEVO LOS CONJUNTOS

    traindata=data.iloc[:traindata.shape[0],:] 
    testdata=data.iloc[traindata.shape[0]:,:]
    testdata=testdata.drop('Poder_Adquisitivo',axis=1) #We drop the unknown variable in the test. It was just filled with NAs
    return traindata,testdata

def prep_dist_datasets(traindata,testdata):
    print('Preparando los datasets para el guardado...')
    #Preparamos los datasets con distintos datos para comprobar cual correlaciona mejor en los regresores
    traindata_Sin_totales = traindata.drop(['Imp_Cons_total','Imp_Sal_total','Ind_Prod_total','Num_Oper_total','Relacion', \
                                         'Imp_Cons_total_2','Imp_Sal_total_2','Ind_Prod_total_2','Num_Oper_total_2','Relacion_2', \
                                         'Imp_Cons_total_3','Imp_Sal_total_3','Ind_Prod_total_3','Num_Oper_total_3','Relacion_3'],axis=1)

    Test_data_Sin_totales = testdata.drop(['Imp_Cons_total','Imp_Sal_total','Ind_Prod_total','Num_Oper_total','Relacion', \
                                         'Imp_Cons_total_2','Imp_Sal_total_2','Ind_Prod_total_2','Num_Oper_total_2','Relacion_2', \
                                         'Imp_Cons_total_3','Imp_Sal_total_3','Ind_Prod_total_3','Num_Oper_total_3','Relacion_3'],axis=1)

    traindata_Con_totales = traindata.drop(['Imp_Cons_total_2','Imp_Sal_total_2','Ind_Prod_total_2','Num_Oper_total_2','Relacion_2', \
                                         'Imp_Cons_total_3','Imp_Sal_total_3','Ind_Prod_total_3','Num_Oper_total_3','Relacion_3'],axis=1)

    Test_data_Con_totales = testdata.drop(['Imp_Cons_total_2','Imp_Sal_total_2','Ind_Prod_total_2','Num_Oper_total_2','Relacion_2', \
                                         'Imp_Cons_total_3','Imp_Sal_total_3','Ind_Prod_total_3','Num_Oper_total_3','Relacion_3'],axis=1)

    traindata_Con_totales_y_cuadrados = traindata.drop(['Imp_Cons_total_3','Imp_Sal_total_3','Ind_Prod_total_3','Num_Oper_total_3','Relacion_3'],axis=1)

    
    Test_data_Con_totales_y_cuadrados = testdata.drop(['Imp_Cons_total_3','Imp_Sal_total_3','Ind_Prod_total_3','Num_Oper_total_3','Relacion_3'],axis=1)
    
   
    
    return traindata_Sin_totales,Test_data_Sin_totales,traindata_Con_totales,\
    Test_data_Con_totales,traindata_Con_totales_y_cuadrados,Test_data_Con_totales_y_cuadrados


def create_folders():
    #Comprobamos si existen las carpetas y si no las creamos
    print('Creacion de carpetas...')
    if not os.path.exists('Total'):
        os.makedirs('Total')
    if not os.path.exists('Sin_Totales'):
        os.makedirs('Sin_Totales')
    if not os.path.exists('Con_Totales'):
        os.makedirs('Con_Totales')
    if not os.path.exists('Con_Totales_y_Cuadrados'):
        os.makedirs('Con_Totales_y_Cuadrados')
    if not os.path.exists('Total_Visualizacion'):
        os.makedirs('Total_Visualizacion')

def save_datasets(traindata,testdata,traindata_Sin_totales,Test_data_Sin_totales,traindata_Con_totales,\
                 Test_data_Con_totales,traindata_Con_totales_y_cuadrados,Test_data_Con_totales_y_cuadrados,Train_visu,Test_visu):
    #Procedemos a guardar los datasets
    print('saving files...')    
    print('Total')
    traindata.to_csv('./Total/traindata.csv', sep=',', encoding='utf-8',index=False)
    testdata.to_csv('./Total/TEST.csv', sep=',', encoding='utf-8',index=False)
    
    print('Dataset de Visualizacion')
    Train_visu.to_csv('./Total_Visualizacion/traindata.csv', sep=',', encoding='utf-8',index=False)
    Test_visu.to_csv('./Total_Visualizacion/TEST.csv', sep=',', encoding='utf-8',index=False)
    
    print('Sin_Totales')
    traindata_Sin_totales.to_csv('./Sin_Totales/traindata.csv', sep=',', encoding='utf-8',index=False)
    Test_data_Sin_totales.to_csv('./Sin_Totales/TEST.csv', sep=',', encoding='utf-8',index=False)
    
    print('Con_Totales')
    traindata_Con_totales.to_csv('./Con_Totales/traindata.csv', sep=',', encoding='utf-8',index=False)
    Test_data_Con_totales.to_csv('./Con_Totales/TEST.csv', sep=',', encoding='utf-8',index=False)
    
    print('Con_Totales_y_Cuadrados')
    traindata_Con_totales_y_cuadrados.to_csv('./Con_Totales_y_Cuadrados/traindata.csv', sep=',', encoding='utf-8',index=False)
    Test_data_Con_totales_y_cuadrados.to_csv('./Con_Totales_y_Cuadrados/TEST.csv', sep=',', encoding='utf-8',index=False)
    print('Saved')
