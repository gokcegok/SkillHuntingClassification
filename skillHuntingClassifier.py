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

eda.check_df(data)
# %% 
# Dropping "1" (Goalkepper) in position_id column

data = data[data["position_id"] != 1]

# Dropping "below_average" in "potential_label".
# It makes up 1% of data(136/10730)
# Other classes: 'average', 'highlighted'

print(data["potential_label"].value_counts())

data = data[data["potential_label"] != "below_average"]

# %% DATA PREPROCESSING

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