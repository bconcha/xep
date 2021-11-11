

import pandas as pd 
import numpy as np 

data = pd.read_csv(r'C:\Users\56982\Documents\xep\‪data_ready.csv', index_col = 0)
data['paidAt'] = pd.to_datetime(data['paidAt'])
data = data.sort_values(by = 'paidAt')
data.head()

# # First we predict the amount

data1 = data[['paidAt', 'amount', 'month']]

# +

#add previous sales to the next row
data1['prev_amount'] = data1['amount'].shift(1)
#drop the null values and calculate the difference
data1 = data1.dropna() #drop FAILED because they have no month or paidAt
data1['diff'] = (data1['amount'] - data1['prev_amount'])
# -

#adding lags
for inc in range(1,13):
    field_name = 'lag_' + str(inc)
    data1[field_name] = data1['amount'].shift(inc)
data1
#drop null values
data1 = data1.dropna().reset_index(drop=True)

# Import statsmodels.formula.api
import statsmodels.formula.api as smf
# Define the regression formula
model = smf.ols(formula='amount ~ lag_1', data=data1)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

model = smf.ols(formula='amount ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data=data1)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

# The lags dont explain the variation, so we try with another model


