# This a practice file for pandas datasets
 import numpy as np
 import pandas as pd
 
 df = pd.read_csv("../git/data.csv")

 df.isnull().any(axis = 0)
 df.describe()
 df.info()
 df.head()
 
