# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:35:21 2022

@author: selca
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# uyarıları kapatma
import warnings

warnings.filterwarnings("ignore")

# veriyi içeri aktarma
veri = pd.read_csv("olimpiyatlar/athlete_events.csv")
veri.head()
# veri hakkında genel bilgi
veri.info()

# verinin temizlenmesi
veri.columns
print(veri.columns)

#sütun ismini değiştirme 
veri.rename(columns={'ID' : 'id',
                    'Name' : 'isim',
                    'Gender' : 'cinsiyet',
                    'Age' : 'yas',
                    'Height' : 'boy',
                    'Weight' : 'kilo',
                    'Team' : 'takim',
                    'NOC' : 'uok',
                    'Games' : 'oyunlar',
                    'Year' : 'yil',
                    'Season' : 'sezon',
                    'City' : 'sehir',
                    'Event' : 'etkinlik',
                    'Medal' : 'madalya', 
                    },inplace = True)
print(veri.columns)
veri.head()

#yararsız verileri çıkarmak (drop)

veri = veri.drop(["id", "oyunlar"], axis = 1)#axis = 1 -> 1 sütun manasında
veri.head()

# kayıp veri yani nan olan veriler -> biz suan ort gore dolduruyoruz



