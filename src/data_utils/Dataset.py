import pandas as pd
import numpy as np
from .Download_yf import *
from typing import Tuple
from sklearn.preprocessing import StandardScaler

def build_dataset_returns(df: pd.DataFrame, n_lags: int = 10) -> tuple[np.ndarray, np.ndarray]:

    """
    Build a dataset for predicting the next return from N previous returns.
    
    Args:
        df: DataFrame with a Date index and a single column of returns
        n_lags: number of previous returns to use as features
    
    Returns:
        X: shape (n_samples, n_lags)
        y: shape (n_samples, 1)
    """
    returns = df.iloc[:, 0].values 
    
    X, y = [], []
    for i in range(n_lags, len(returns)):
        X.append(returns[i - n_lags:i])  
        y.append(returns[i])             
    
    return np.array(X), np.array(y).reshape(-1, 1)


def build_dataset_abs_returns(df: pd.DataFrame, n_lags: int = 10) -> tuple[np.ndarray, np.ndarray]:

    """
    Build a dataset for predicting the next return from N previous returns.
    
    Args:
        df: DataFrame with a Date index and a single column of returns
        n_lags: number of previous returns to use as features
    
    Returns:
        X: shape (n_samples, n_lags)
        y: shape (n_samples, 1)
    """
    returns = df.iloc[:, 0].values 
    
    X, y = [], []
    for i in range(n_lags, len(returns)):
        X.append(returns[i - n_lags:i])  
        y.append(np.abs(returns[i]))             
    
    return np.array(X), np.array(y).reshape(-1, 1)

def scale_data(X,y):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X)                                                                                                                                                                 

    y = scaler_y.fit_transform(y) 
    return X, y, scaler_x, scaler_y                                                                                                                                                                
