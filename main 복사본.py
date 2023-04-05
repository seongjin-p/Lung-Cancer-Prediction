import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    '/Users/seongjin/workspace/Project - HAP/lung cancer patient.csv')

df['Level'] = df['Level'].replace({'High': 100, 'Medium': 66, 'Low': 33})

x = df[['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'OccuPational Hazards', 'Genetic Risk',
        'chronic Lung Disease', 'Smoking', 'Chest Pain', 'Coughing of Blood', 'Shortness of Breath', 'Dry Cough']].values
y = df['Level'].values

model = LinearRegression()
model.fit(x, y)
