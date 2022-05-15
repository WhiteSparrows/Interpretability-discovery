#!/usr/bin/env python 
# coding=utf-8

import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn_pandas import DataFrameMapper # class for mapping pandas data frame columns to different sklearn transformations
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils import evaluate
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.inspection import PartialDependenceDisplay



X, y = shap.datasets.imdb()#Classification
X, y = shap.datasets.diabetes()#Regression
print(X.head())
print(f"Mean value of disease progression one year after baseline is {round(y.mean(), 3)}")



"""
Logistic regression
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)


"""
Classic template, here useless for categorical keep it just for template :)
"""
print(f"-------------------------------------------Beginning of training with Logostic regression")
catagorical_features = []
numerical_features = [c for c in X_train.columns if c not in catagorical_features]
cat = [([c], [OrdinalEncoder()]) for c in catagorical_features]
num = [([n], [SimpleImputer(), StandardScaler()]) for n in numerical_features]
mapper = DataFrameMapper(num + cat, df_out=True)
preprocessed_X_train = mapper.fit_transform(X_train)
preprocessed_X_train = sm.add_constant(preprocessed_X_train)
reg = sm.OLS(y_train, preprocessed_X_train).fit()


print(f"------------------------------------------End of training with Logostic regression")




train_mae = evaluate(X_train, y_train, mapper, reg, True)
test_mae = evaluate(X_test, y_test, mapper, reg, True)
print(f"train MAE = {round(train_mae, 3)}, test MAE = {round(test_mae, 3)} ")

print(reg.summary())



"""
Random forests
"""

print(f"-------------------------------------------Beginning of training with Random Forest")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
catagorical_features = []
numerical_features = [c for c in X_train.columns if c not in catagorical_features]
cat = [([c], [OrdinalEncoder()]) for c in catagorical_features]
num = [([n], [SimpleImputer(), StandardScaler()]) for n in numerical_features]
mapper = DataFrameMapper(num + cat, df_out=True)
reg = RandomForestRegressor()
pipeline = Pipeline([
    ('preprocess', mapper),
    ('reg', reg)
])
p = pipeline.fit(X_train, y_train)

train_mae = evaluate(X_train, y_train, reg=pipeline)
test_mae = evaluate(X_test, y_test, reg=pipeline)
print(f"train MAE = {round(train_mae, 3)}, test MAE = {round(test_mae, 3)} ")

sorted_idx = reg.feature_importances_.argsort()
features = numerical_features + catagorical_features
result = sorted(zip(features, reg.feature_importances_), key = lambda x: x[1], reverse=False)
plt.barh([x[0] for x in result], [x[1] for x in result])
plt.title('Features importances by decreasing order using Random forest')
plt.show()
print(f"-------------------------------------------End of training with Random Forest")

print(f"-------------------------------------------Beginning of training with NN")
preprocessed_X_train = mapper.fit_transform(X_train)

num_epochs = 50
learning_rate = 0.01
hidden_size = 32
batch_size = 50
input_dim = preprocessed_X_train.shape[1]
batch_no = preprocessed_X_train.shape[0] // batch_size
model = nn.Sequential(
    nn.Linear(input_dim, hidden_size),
    nn.Linear(hidden_size, 1)
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_batch = Variable(torch.FloatTensor(preprocessed_X_train.values[start:end]))
        y_batch = Variable(torch.FloatTensor(y_train[start:end]))
        optimizer.zero_grad()
        y_preds = model(x_batch)
        loss = criterion(y_preds, torch.unsqueeze(y_batch,dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 0:
        print("Epoch {}, Loss: {}".format(epoch, running_loss))

preprocessed_X_test = mapper.transform(X_test)
y_pred = model(torch.from_numpy(preprocessed_X_test.values).float()).flatten().detach().numpy()
test_mae = mean_absolute_error(y_test, y_pred)
preprocessed_X_train = mapper.transform(X_train)
y_pred = model(torch.from_numpy(preprocessed_X_train.values).float()).flatten().detach().numpy()
train_mae = mean_absolute_error(y_train, y_pred)
print(f"\ntrain MAE = {round(train_mae, 3)}, test MAE = {round(test_mae, 3)} ")


print(f"-------------------------------------------End of training with NN")

pipeline = Pipeline([
    ('preprocess', mapper),
    ('reg', reg)
])
p = pipeline.fit(X_train, y_train)

print(features)
explainer = shap.Explainer(pipeline.predict, X_train)
shap_values = explainer(X_test)



shap.initjs()
plt.title('Waterfall for logistic regression')
shap.plots.waterfall(shap_values[0])




fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Decision Tree")
rf_disp = PartialDependenceDisplay.from_estimator(reg, X_train, ["age", "bmi"], ax=ax)

















plt.show()
