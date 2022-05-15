#!/usr/bin/env python
# coding=utf-8

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

def evaluate(X, y, mapper=None, reg=None, transform=False):
    if transform:
        X = mapper.transform(X)
        X = sm.add_constant(X, has_constant='add')
    y_pred = reg.predict(X)
    return mean_absolute_error(y, y_pred)
