# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:00:21 2021

@author: SiF
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests


file = pd.read_csv("avg-household-size.csv")
house_df = pd.DataFrame(file)

file = pd.read_csv("cancer_reg.csv")
cancer_df = pd.DataFrame(file)