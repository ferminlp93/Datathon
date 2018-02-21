#############
# LIBRERIAS #
#############

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,KFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from bayes_opt import BayesianOptimization
from functools import partial
from multiprocessing import Pool
from Utilidades import *

"""
|------------------------------------------------|
|FUNCIONES PARA SELECCION Y VALIDACION DE MODELOS|
|------------------------------------------------|
"""


#CREA UNA TABLA DE MODELOS, PARAMETROS Y RESULTADOS DADA UNA LISTA DE MODELOS
def get_models_table(models):
    models_table={}
    model_names=[get_model_name(model) for model in models]
    models_table['Regressor']=model_names
    models_table['model']=models
    models_table['error']=np.repeat(np.nan,len(models))
    models_table['params']=[{} for i in range(len(models))]

    return pd.DataFrame(models_table).set_index('Regressor')

#ELIMINA LAS PREDICCIONES POR DEBAJO DEL MINIMO Y MAXIMO DE TRAIN
#PARA EVITAR PREDICCIONES DESORBITADAMENTE PEQUENAS O GRANDES
def remove_outlier_predictions(y_pred,y_train):
    min_val, max_val = y_train.min(), y_train.max()
    result=y_pred.copy()
    result[y_pred<min_val]= min_val
    result[y_pred>max_val]= max_val
    
    return result


def log_scoring(y,y_pred):
    return mean_absolute_error(np.power(10,y),np.power(10,y_pred))
    


#CALCULA LA PUNTUACION DE UN MODELO ELIMINANDO LOS OUTLIERS EXTREMOS
def validate(model,X,y,scoring,verbose,indexes):
    train_index,test_index=indexes
    X_matrix = to_matrix(X)
    y_matrix = to_matrix(y)
        
    newX,newy=delete_outliers(X_matrix[train_index],y_matrix[train_index])
    result=scoring(y_matrix[test_index],fit_predict(clone(model),newX,newy,X_matrix[test_index]))
    if verbose:
        print(result)
    return result

#OBTIENE EL NOMBRE DE UN MODELO
def get_model_name(model):
    
    return str(model.__class__).split('.')[-1].split("'")[0]


#ENTRENA EL MODELO Y OBTIENE LAS PREDICCIONES DEL TEST
def fit_predict(model,X,y,X_test):
    model.fit(X,y)
    return remove_outlier_predictions(model.predict(X_test),y)


#CALCULA (DE FORMA PARALELA) LA PUNTUACION DE CROSS VALIDATION DE UN MODELO
def error_cv(model,X, y, verbose=0, metric=log_scoring, cv=4):
    if verbose>0:
        print('cross validation with',cv,' folds')
    kfold = KFold(n_splits=cv, shuffle=True)    
    pool=Pool(4)
    result=np.array(pool.map(partial(validate,model,X,y,metric,verbose),list(kfold.split(X, y)),chunksize=1))
    pool.close()
    
    return abs(result).mean()

#CONVIERTE A ENTEROS LOS PARAMETROS QUE DEBEN SERLO
#SE UTILIZA PARA PREVENIR AL MODELO DE POSIBLES CONFIGURACIONES DE PARAMETROS ERRONEAS
def cast_to_int(params):
    result={}
    for key in set(['n_estimators','min_samples_split','max_depth','max_bin','num_leaves','min_data_in_leaf']).intersection(set(params.keys())):
        result[key]=int(params[key])
    return result

#SE UTILIZA COMO FUNCION DE EVALUACION A MAXIMIZAR PARA EL PROCESO DE OPTIMIZACION BAYESSIANA
def model_evaluate(model,X,y,cv,metric,**params):
    return -error_cv(model(**cast_to_int(params)),X,y,metric=metric,cv=cv)

#CREA Y EJECUTA EL PROCESO DE OPTIMIZACION BAYESSIANA CON EL OBJETIVO DE ENCONTRAR LOS PARAMETROS OPTIMOS DE UN MODELO
def error_cv_param_grid(model,X, y,param_grid, verbose=0, metric=log_scoring, cv=4):
    
    rfcBO = BayesianOptimization(partial(model_evaluate,model=type(model),X=X,y=y,cv=cv,metric=metric),param_grid)
    rfcBO.maximize(n_iter=20)
    return abs(rfcBO.res['max']['max_val']),rfcBO.res['max']['max_params']


#EJECUTA (ESTIMANDO SUS PARAMETROS O NO) VARIOS MODELOS DEVOLVIENDO COMO RESULTADO SUS PUNTUACIONES DE CROSS VALIDATION
#Y SU CONFIGURACION DE PARAMETROS OPTIMA (SI LA OPCION ESTA HABILITADA)
#PUEDE TARDAR VARIAS HORAS (ESPECIALMENTE CON ESTIMACION DE PARAMETROS)
def compare_models(models_table, X, y, estimate_params=False, verbose=0, metric=log_scoring, cv=8):
    
    errors=[]
    params=[]
    for i in range(models_table.shape[0]):
        
        model=models_table['model'].iloc[i] 
        if estimate_params:
            score,param = error_cv_param_grid(model,X, y,estimate_params[i],verbose=verbose, metric=metric, cv=cv)
            params.append(param)
        
        else:
            score=error_cv(models[i],X, y,verbose=verbose ,metric=metric, cv=cv)
        
        if verbose>0:
            print(models_table.index[i],': ',score)

        errors.append(score)
    
    models_table['error']=errors
    if estimate_params:
        models_table['params']=params
        
        
#CLASE QUE DEFINE UN REGRESOR POR STACKING         
class Stacking_model(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_folds=4, metric=mean_absolute_error):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.metric = metric
   
    def fit(self, X, y):
        
        X_matrix = to_matrix(X)
        y_matrix = to_matrix(y)
                
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X_matrix, y_matrix):
                print('fitting',get_model_name(model))
                instance = clone(model)
                self.base_models_[i].append(instance)
                out_of_fold_predictions[holdout_index, i] = fit_predict(instance,X_matrix[train_index],y_matrix[train_index],X_matrix[holdout_index])
                
        
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X_test):
        df_test=pd.DataFrame()
        X_matrix = to_matrix(X_test)
        for i, base_models in enumerate(self.base_models_):
            df_test[i]=np.zeros(X_test.shape[0])
            for model in base_models:
                df_test[i]+=model.predict(X_matrix)
                
            df_test[i]/=len(base_models)
        
        return self.meta_model_.predict(df_test)
    
    
    def score(self,X_test,y_test):
        return self.metric(y_test,self.predict(X_test))
