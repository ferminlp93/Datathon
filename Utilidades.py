#############
# LIBRERIAS #
#############

import numpy as np
import pandas as pd
import warnings

"""
|---------------------|
|FUNCIONES COMPARTIDAS|
|---------------------|
"""

#RANGO INTERCUARTILICO
def iqr_calculate(column):
    description = column.describe()
    iqr = description['75%']-description['25%']
    return iqr

#CONVIERTE DATAFRAMES A MATRICES NUMPY
def to_matrix(X):
    X_matrix=X
    
    if type(X)== pd.core.frame.DataFrame or type(X)== pd.core.series.Series:
        X_matrix=X_matrix.as_matrix()
        
    return X_matrix

#LIMITES VALORES ATIPICOS EXTREMOS O NO
def outliers_limits(column, extreme=True):
    col_matrix=to_matrix(column)
    minq,maxq=np.percentile(col_matrix,[25,75])
    iqr = maxq-minq
    coef=(extreme+1)*1.5
    outliers_limits = minq-coef*iqr,maxq+coef*iqr
    return outliers_limits

#ELIMINA VALORES ATIPICOS
def delete_outliers(X, y, extreme=True):
    limits = outliers_limits(y,extreme)
    
    X_no_outliers = X[y<=limits[1]]
    y_no_outliers = y[y<=limits[1]]
    
    X_no_outliers = X_no_outliers[y_no_outliers>=limits[0]]
    y_no_outliers = y_no_outliers[y_no_outliers>=limits[0]]
    
    return X_no_outliers,y_no_outliers
