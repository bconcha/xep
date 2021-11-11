
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

import shap
from catboost import CatBoostRegressor
# -

data = pd.read_csv(r'C:\Users\56982\Documents\xep\‪data_ready.csv', index_col = 0)
data['paidAt'] = pd.to_datetime(data['paidAt'])
data = data.sort_values(by = 'paidAt')
data.head()

# +

data.sample(n = 100)
# -

data[data['month']==9].shape[0]-data[data['month']==8].shape[0]
data[data['month']==8].shape[0]-data[data['month']==7].shape[0]

data['month'].value_counts(normalize = True)

data['month'].value_counts()
#Proximo mes 408

# +
#Dado que los promedios de transacciones bajan en los últimos 3 meses 
#Elegimos estos últimos 3 para poder predecir las compras del siguiente mes

to_predict = data[data['month']>=7].sample(n = 408)
# -

data.columns

# # CatBoost amount

data_amount = data.drop(columns = 'amountfinancedByXepelin')

# +
features=data_amount[['month','amount',
             'duplicatedTransaction','repetitionPayers', 'repetitionReceivers',
            'proportionPayerId', 'proportionReceiverId']]


X = features.drop('amount', axis=1)
y = features['amount']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42)
# -

model=CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, loss_function='RMSE',random_state=42)
model.fit(X_train, y_train,eval_set=(X_valid, y_valid));

# +

# Baseline errors, and display average baseline error
baseline_errors = abs(y_valid.mean() - y_valid)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), '%')
print()

# Use the forest's predict method on the test data
predictions = model.predict(X_valid)
# Calculate the absolute errors
errors = abs(predictions - y_valid)  
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), '%')
print()

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_valid)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print()
# -

# We dont really care about the accuracy on each element, we care about the total amount.

y_valid.sum()

predictions.sum()

abs((predictions.sum()-y_valid.sum())/y_valid.sum())*100

# The difference between the y_valid is a thirnd

# # CatBoost amountfinancedByXepelin

data_amountXep = data.drop(columns = 'amount')

# +
features=data_amountXep[['month','amountfinancedByXepelin',
             'duplicatedTransaction','repetitionPayers', 'repetitionReceivers',
            'proportionPayerId', 'proportionReceiverId']]


X = features.drop('amountfinancedByXepelin', axis=1)
y = features['amountfinancedByXepelin']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42)
# -

model=CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, loss_function='RMSE',random_state=42)
model.fit(X_train, y_train,eval_set=(X_valid, y_valid));

# +

# Baseline errors, and display average baseline error
baseline_errors = abs(y_valid.mean() - y_valid)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), '%')
print()

# Use the forest's predict method on the test data
predictions = model.predict(X_valid)
# Calculate the absolute errors
errors = abs(predictions - y_valid)  
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), '%')
print()

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_valid)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print()
# -

predictions.sum()

y_valid.sum()

abs((predictions.sum()-y_valid.sum())/y_valid.sum())*100

# # Catboost With Categorical in amount

data_amount = data.drop(columns = 'amountfinancedByXepelin')

# +
features=data_amount[['month','amount','PayerId','ReceiverId',
             'duplicatedTransaction','repetitionPayers', 'repetitionReceivers',
            'proportionPayerId', 'proportionReceiverId']]

features[['PayerId', 'ReceiverId']] = features[['PayerId', 'ReceiverId']].astype(str)

# -

X = features.drop('amount', axis=1)
y = features['amount']
categorical_features_indices = np.where(X.dtypes == np.object)[0]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42)

model=CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, loss_function='RMSE',random_state=42)
model.fit(X_train, y_train,cat_features=categorical_features_indices, eval_set=(X_valid, y_valid));

# +

# Baseline errors, and display average baseline error
baseline_errors = abs(y_valid.mean() - y_valid)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), '%')
print()

# Use the forest's predict method on the test data
predictions = model.predict(X_valid)
# Calculate the absolute errors
errors = abs(predictions - y_valid)  
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), '%')
print()

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_valid)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print()
# -

y_valid.sum()

predictions.sum() #se acerca más!!!!

abs((predictions.sum()-y_valid.sum())/y_valid.sum())*100

# # Catboost Categorical in amountfinancedByXepelin

data_amountXep = data.drop(columns = 'amount')

# +

features=data_amountXep[['month','amountfinancedByXepelin','PayerId','ReceiverId',
             'duplicatedTransaction','repetitionPayers', 'repetitionReceivers',
            'proportionPayerId', 'proportionReceiverId']]

features[['PayerId', 'ReceiverId']] = features[['PayerId', 'ReceiverId']].astype(str)


X = features.drop('amountfinancedByXepelin', axis=1)
y = features['amountfinancedByXepelin']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42)
# -



model=CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, loss_function='RMSE',random_state=42)
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_valid, y_valid));

# +

# Baseline errors, and display average baseline error
baseline_errors = abs(y_valid.mean() - y_valid)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), '%')
print()

# Use the forest's predict method on the test data
predictions = model.predict(X_valid)
# Calculate the absolute errors
errors = abs(predictions - y_valid)  
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), '%')
print()

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_valid)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print()
# -

y_valid.sum()

predictions.sum()

abs((predictions.sum()-y_valid.sum())/y_valid.sum())*100


