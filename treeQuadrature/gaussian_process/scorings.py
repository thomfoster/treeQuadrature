from sklearn.metrics import r2_score
from scipy.stats import norm
import numpy as np

def r2(y_true, y_pred, sigma):
    return r2_score(y_true, y_pred)

def predictive_ll(y_true, y_pred, sigma):
    return np.sum(norm.logpdf(y_true, loc=y_pred, scale=sigma))
