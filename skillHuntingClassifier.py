# -*- coding: utf-8 -*-
"""
Skill Hunting Classification using Machine Learning Methods


Business Problem:
    
    Predicting which class (average, highlighted) the players are in 
    according to the scores given to the characteristics of 
    the football players watched by the Scouts.
    
Created on Tue May  2 02:08:33 2023

@author: gokcegok
"""

# %% LIBRARIES

import pandas as pd
import eda
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# %% CV function

def crossVal(classifier, name, X, y, output=False):
    
    """
    Cross validation

    Parameters
    ----------
    classifier : classifier object
    
    output : boolean, optional
        If it is True, function returns the CV results. 
        Else, prints classification metrics. 
        The default is False.

    Returns
    -------
    cv_results : TYPE
        DESCRIPTION.

    """
    
    model = classifier.fit(X, y)
    
    cv_results = cross_validate(model, X, y, cv=5,
                                scoring=["accuracy", "f1", "precision",
                                         "recall", "roc_auc"])
    
    print(name)
    print("--------------------")
    print("Accuracy:", round(cv_results["test_accuracy"].mean(), 4))
    print("Recall:", round(cv_results["test_recall"].mean(), 4))
    print("Precision:", round(cv_results["test_precision"].mean(), 4))
    print("F1 Score:", round(cv_results["test_f1"].mean(), 4))
    print("ROC AUC:", round(cv_results["test_roc_auc"].mean(), 4), "\n")
    
    if output == True:
        return cv_results
    
    
# %% DATASET

# --------------------------------------------------------------------------- #
# The data set consists of information from Scoutium, which includes 
# the features and scores of the football players evaluated by the scouts 
# according to the characteristics of the footballers observed in the matches.
# --------------------------------------------------------------------------- #

attributes = pd.read_csv("scoutium_attributes.csv", sep=";")
potential_labels = pd.read_csv("scoutium_potential_labels.csv", sep=";")

# VARIABLES

# task_response_id: The id of the set of a scout's evaluations of all players 
# on a team's roster in a match

# match_id: Match ID

# evaluator_id: Scout ID

# player_id: Player ID

# position_id: The id of the position played by the relevant player in that match
# 1: Goalstopper, 2: Stoper, 3: Right Back, 4: Left Back, 
# 5: Defensive Midfielder, 6: Center Midfielder, 7: Right Winger, 
# 8: Left Winger, 9: Offensive Midfielder, 10: Forward

# analysis_id: The id of the set containing a scout's attribute evaluations 
# of a player in a match

# attribute_id: The id of each attribute the players were evaluated on

# attribute_value: Value (points) given by a scout to a player's attribute

# potential_label: Label indicating the final decision of a scout 
# regarding a player in a match. (target variable)


# MERGING DATASETS

data_ = pd.merge(attributes, potential_labels, how="left",
                on = ['task_response_id', 'match_id', 
                      'evaluator_id', 'player_id'])

data = data_.copy()

# %% 
# Dropping "1" (Goalkepper) in position_id column

data = data[data["position_id"] != 1]

# Dropping "below_average" in "potential_label".
# It makes up 1% of data(136/10730)
# Other classes: 'average', 'highlighted'

print(data["potential_label"].value_counts())

data = data[data["potential_label"] != "below_average"]

# Preparing data for classification

# Pivot table with 
# "player_id", "position_id" and "potential_label" in the index, 
# "attribute_id" in the columns and 
# the score given by the scouts to the players "attribute_value" in the values


data = pd.pivot_table(data, values='attribute_value', 
                       index=["player_id", "position_id", "potential_label"],
                       columns=["attribute_id"])

data.reset_index(inplace=True)

data.columns = data.columns.astype("str")

# we now have a dataset with each row holding 
# the points awarded to a player by the scouts, 
# the player's position and the target variable!

# %% DATA PREPROCESSING

# EXPLORATORY DATA ANALYSIS

eda.check_df(data)

cat_cols, num_cols, cat_but_car = eda.grab_col_names(data, cat_th=5)


for col in num_cols:
    eda.check_outlier(data, col)
    eda.num_summary(data, col, plot=True)

# LABEL ENCODING

label_encoder = LabelEncoder()
data["potential_label"] = label_encoder.fit_transform(data["potential_label"])

# %% CLASSIFICATION DATA

y = data["potential_label"]
X = data.drop(["position_id", "player_id", "potential_label"], axis=1)

# Scaling numeric columns
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)


# %% MODEL SELECTION

models = {"Logistic Regression": LogisticRegression(random_state=15),
          "CART": DecisionTreeClassifier(random_state=15),
          "Gradient Boosting": GradientBoostingClassifier(random_state=15),
          "XGBoost": XGBClassifier(random_state=15),
          "LightGBM": LGBMClassifier(random_state=15)}


for model_name in models.keys():
    
    crossVal(models[model_name], model_name, X, y)

# %% LightGBM 

# Parameter Optimization

lgbm_params = {"max_bin": [255, 150, 300],
               "max_depth": [100, 200, 300, 400, 500],
               "colsample_bytree": [0.5, 0.75, 1],
               "learning_rate": [0.05, 0.075, 0.01, 0.025]}

print("Searching for parameters...")

lgbm = LGBMClassifier().fit(X, y)

lgbm_best_search = GridSearchCV(estimator=lgbm, param_grid=lgbm_params,
                                cv=5, verbose=1).fit(X, y)

lgbm_best_search.best_params_

"""
Best params in first search: 
    
best_params_1 = {'colsample_bytree': 0.75,
                 'learning_rate': 0.025,
                 'max_bin': 255,
                 'max_depth': 100}
"""

lgbm_final = LGBMClassifier(**lgbm_best_search.best_params_,
                             random_state=15).fit(X, y)

crossVal(lgbm, "LGBM Primary Results", X, y)
crossVal(lgbm_final, "LGBM Final Results", X, y)

# %% Prediction

y_pred = lgbm_final.predict(X)
confusion_matrix(y, y_pred)


lgbm_params_ = {"colsample_bytree": [0.75, 0.80, 0.90],
               "learning_rate": [0.025, 0.03],
               "min_child_weight": [0.01, 0.05, 0.03],
               "num_leaves": [31, 25, 20, 15]}

lgbm_best_search_ = GridSearchCV(estimator=lgbm, param_grid=lgbm_params_,
                                cv=5, verbose=1).fit(X, y)

lgbm_final.get_params()
lgbm_best_search_.best_params_

"""
Best params in second search:
best_params_2 = {'colsample_bytree': 0.75,
                 'learning_rate': 0.03,
                 'min_child_weight': 0.01,
                 'num_leaves': 31}

"""
lgbm_final_ = LGBMClassifier(**lgbm_best_search_.best_params_,
                             random_state=15).fit(X, y)

crossVal(lgbm_final, "LGBM Final Results", X, y)
crossVal(lgbm_final_, "LGBM Final Results 2", X, y)

y_pred_ = lgbm_final_.predict(X)
tn, fp, fn, tp = confusion_matrix(y, y_pred_).ravel()
(tn, fp, fn, tp)

# %%

eda.plot_importance(lgbm_final, X, num=10)

data["predicted_label"] = label_encoder.inverse_transform(y_pred_)
