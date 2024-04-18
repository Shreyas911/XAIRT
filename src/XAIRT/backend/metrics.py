""" Custom metrics with Keras """
from __future__ import annotations

import warnings
import numpy as np
warnings.simplefilter("ignore")

from XAIRT.backend.types import MatrixNumpy

import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K

__all__ =["metricF1"]

# https://datascience.stackexchange.com/questions/105101/which-keras-metric-for-multiclass-classification
def metricF1(y_true: MatrixNumpy, y_pred: MatrixNumpy) -> float:    
    def recall_m(y_true: MatrixNumpy, y_pred: MatrixNumpy) -> float:
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 

    def precision_m(y_true: MatrixNumpy, y_pred: MatrixNumpy) -> float:
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))