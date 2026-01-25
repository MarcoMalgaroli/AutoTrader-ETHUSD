import pandas as pd
import numpy as np

og = pd.read_csv('datasets/final/ETHUSD_D1_3055.csv', parse_dates = True)
print(og)
# Calcoliamo la matrice di correlazione
df = og.select_dtypes(include = [np.number])
print(df)
correlation = df.corr()['target'].sort_values(ascending = False)
print(correlation)

correlation = og.corr(numeric_only = True)['target'].sort_values(ascending = False)
print(correlation)
