import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the data into a Pandas dataframe
data = pd.DataFrame({
    'sales': [1000000, 1200000, 1150000, 1300000, 1400000],
    'tv_spend': [100000, 120000, 110000, 130000, 140000],
    'online_spend': [50000, 55000, 60000, 65000, 70000],
    'social_spend': [20000, 22000, 25000, 27000, 30000],
    'other_spend': [30000, 32000, 35000, 40000, 45000]
})

# Create the X and Y matrices for linear regression
X = data[['tv_spend', 'online_spend', 'social_spend', 'other_spend']]
X = sm.add_constant(X)
y = data['sales']

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())
