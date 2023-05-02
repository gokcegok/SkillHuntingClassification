# -*- coding: utf-8 -*-
"""
SCOUTIUM

Makine Ogrenmesi ile Yetenek Avciligi Siniflandirma

İs Problemi: 
    
    Scout’lar tarafından izlenen futbolcuların 
    özelliklerine verilen puanlara göre, 
    oyuncuların hangi sınıf (average, highlighted) 
    oyuncu olduğunun tahminlemesi
    
    Scout: Gözlemci ya da yetenek avcısı, 
    gelecek vadettiği düşünülen sporcuları 
    gözlemleyerek mevcut yeteneklerini ve 
    potansiyellerini tespit eden uzman kişi
Created on Tue May  2 02:08:33 2023

@author: gokcegok
"""

# %% Libraries

import pandas as pd
import eda
from sklearn.preprocessing


# %% Dataset

# --------------------------------------------------------------------------- #
# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre 
# scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan 
# özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.
# --------------------------------------------------------------------------- #

attributes = pd.read_csv("scoutium_attributes.csv", sep=";")

# Degiskenler

# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki 
# tüm oyunculara dair değerlendirmelerinin kümesi
# match_id: İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# position_id: İlgili oyuncunun o maçta oynadığı pozisyonun id’si
# 1: Kaleci, 2: Stoper, 3: Sağ bek, 4: Sol bek, 
# 5: Defansif orta saha, 6: Merkez orta saha, 7: Sağ kanat, 
# 8: Sol kanat, 9: Ofansif orta saha, 10: Forvet
# analysis_id: Bir scoutun bir maçta bir oyuncuya dair 
# özellik değerlendirmelerini içeren küme
# attribute_id: Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value: Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)

potential_labels = pd.read_csv("scoutium_potential_labels.csv", sep=";")

# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki 
# tüm oyunculara dair değerlendirmelerinin kümesi
# match_id: İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# potential_label: Bir scoutun bir maçta bir oyuncuyla ilgili 
# nihai kararını belirten etiket. (hedef değişken)


# Merging datasets

data_ = pd.merge(attributes, potential_labels, how="left",
                on = ['task_response_id', 'match_id', 
                      'evaluator_id', 'player_id'])

data = data_.copy()

eda.check_df(data)
# %%

# Dropping "Kaleci"(goalkepper) in position_id column

data = data[data["position_id"] != 1]

# Dropping "below_average" in "potential_label".
# It makes up 1% of data(136/10730)
# Other classes: 'average', 'highlighted'

print(data["potential_label"].value_counts())

data = data[data["potential_label"] != "below_average"]

# %%

# Pivot table with 
# "player_id", "position_id" and "potential_label" in the index, 
# "attribute_id" in the columns and 
# the score given by the scouts to the players "attribute_value" in the values


pivot = pd.pivot_table(data, values='attribute_value', 
                       index=["player_id", "position_id", "potential_label"],
                       columns=["attribute_id"])

pivot.reset_index(inplace=True)

pivot.columns = pivot.columns.astype("str")

# we now have a dataset with each row holding 
# the points awarded to a player by the scouts, 
# the player's position and the target variable!

# %%




























