import numpy as np
import pandas as pd
import warnings
import pickle

train = pd.read_csv("train_BS.csv")
test = pd.read_csv("test_BS.csv")

test["casual"] = np.NaN
test["registered"] = np.NaN
test["count"] = np.NaN

train["datetime"] =  pd.to_datetime(train["datetime"])
test["datetime"] = pd.to_datetime(test["datetime"])

train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["hour"] = train["datetime"].dt.hour
train["DOW"] = train["datetime"].dt.dayofweek

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["hour"] = test["datetime"].dt.hour
test["DOW"] = test["datetime"].dt.dayofweek

ind_var = ["workingday","temp","year","month","hour","DOW"]

X_org_train = train[ind_var]
Y_org_train = train["count"]

X_org_test = test[ind_var]
Y_org_test = test["count"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_org_train, Y_org_train, test_size = 0.25, random_state =42)

from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators = 500, max_features = 4, min_samples_leaf = 5, random_state = 42)
rf_model = regressor_rf.fit(X_train, Y_train)

pickle.dump(rf_model,open("model.pkl","wb"))
model = pickle.load(open("model.pkl","rb"))
