

import pandas as pd 
import numpy as np
# -

data = pd.read_csv(r'C:\Users\56982\Documents\xep\data_test2.csv')

data
#we can see that invoiceId is not related with when the transaction is executed.

original_columns = data.columns

data.dtypes

# # Analyze data on the months given

data['paidAt'] = pd.to_datetime(data['paidAt'])

data.sort_values(by = 'paidAt')

data['month'] = data['paidAt'].dt.month
#data['year'] = data['date'].dt.year #892 on 2021, no more years

data.groupby('month')[['amount','amountfinancedByXepelin']].sum().plot(kind = 'bar', title = 'Montos por Mes')

data.groupby('month')['amountfinancedByXepelin'].count().plot(kind = 'bar', title = 'N Transacciones')
#¿Por qué hay menos transacciones el mes de septiembre?

#Se ve que hay transacciones desde los primeros hasta el último día de septiembre
data[data['month']==9].sort_values('paidAt', ascending = False)

data[data['status']=='PROCESSING']

# We can see that the last month there are less transactions, probably because they are still processing when the data was extracted. Thas is why for the modeling terms we will assume that all the processing data is from the last month

data.loc[data['status'] == 'PROCESSING', ['month']] = 9
data.groupby('month')[['amount','amountfinancedByXepelin']].sum().plot(kind = 'bar',  title = 'Montos por Mes')

data.groupby('month')['amountfinancedByXepelin'].count().plot(kind = 'bar', title = 'N Transacciones')

data.groupby('month')[['amount','amountfinancedByXepelin']].mean().plot(kind = 'bar', title = 'Promedio de Transacciones')

data_days1 = data.groupby('paidAt').amount.sum().reset_index()
data_days2 = data.groupby('paidAt').amountfinancedByXepelin.sum().reset_index()

# +
import plotly.graph_objs as go
from chart_studio import plotly
import plotly.offline as pyoff


#plot monthly sales
plot_data1 = go.Scatter(
        x=data_days1['paidAt'],
        y=data_days1['amount'],
    #    name = 'amount'
    )

plot_data2 = go.Scatter(
        x=data_days2['paidAt'],
        y=data_days2['amountfinancedByXepelin'],
    #    name = 'amountfinancedByXepelin'
    )

plot_layout = go.Layout(
        title='Montly Amounts'
    )
data_days = [plot_data1, plot_data2]
fig = go.Figure(data=data_days, layout=plot_layout)
pyoff.iplot(fig)


# +

data_month1 = data.groupby('month').amount.sum().reset_index()
data_month2 = data.groupby('month').amountfinancedByXepelin.sum().reset_index()

#plot monthly sales
plot_data1 = go.Scatter(
        x=data_month1['month'],
        y=data_month1['amount'],
    #    name = 'amount'
    )

plot_data2 = go.Scatter(
        x=data_month2['month'],
        y=data_month2['amountfinancedByXepelin'],
    #    name = 'amountfinancedByXepelin'
    )

plot_layout = go.Layout(
        title='Montly Amounts'
    )
data_months = [plot_data1, plot_data2]
fig = go.Figure(data=data_months, layout=plot_layout)
pyoff.iplot(fig)
# -


# Is better to see the data monthly

# #cuales son las tendencias, le va prestando cada vez más cada payer?
# #agrupar a nivel de semanas 
# #contrastar industrial con el mercado de esa industria
# #xepelin como partner de apoyo para lo que vas a necesitar tú. 
# #cuantos clientes posibles hay que ya son clientes de otros productos dentro de xep
# #ratio of paid vs failed per client

# # We now examine Payers and Receivers

#We got 200 payers
payers = data['PayerId'].unique()#.tolist()
#We got 77 receivers
receivers = data['ReceiverId'].unique()#.tolist()


#As we can see all the payers are receivers, so this is a closed network
len(np.unique(np.concatenate((payers, receivers)), axis = 0).tolist())

import networkx as nx
import matplotlib.pyplot as plt
G = nx.from_pandas_edgelist(data, source = 'PayerId',
                            target = 'ReceiverId',
                            edge_attr= 'amount')

#Ilustration of the closed network
plt.figure(2,figsize=(12,12)) 
nx.draw_random(G)

plt.figure(2,figsize=(12,12)) 
nx.draw_spring(G)

#Identify the transactions that are repetead over time with a 1
data.loc[:, 'duplicatedTransaction'] = 0
#When the PayerId and the ReceiverId are repetead duplicatedTransaction = 1
data.loc[data.duplicated(subset=['PayerId','ReceiverId'], keep=False), ['duplicatedTransaction']] = 1 #All duplicates as 1

data.groupby(['month', 'duplicatedTransaction']).count()['PayerId'].plot(kind = 'bar')#.value_counts()


# +

data.groupby(['status', 'duplicatedTransaction']).count()
data[data['duplicatedTransaction']==1]['status'].value_counts()
#less than 2% of the duplicatedTransactions Fail
# -

data[data['duplicatedTransaction']==0]['status'].value_counts()
#compared to the 4.6% failed that are not duplicated (more than double)

print('Number of Payers and Receivers at the same time are ', len(list(set(data.PayerId) & set(data.ReceiverId))))
print('Number of unique Receivers are ', len(data['ReceiverId'].unique().tolist()))
#-->All receivers are payers

#Now we observe if there is any relation if the paid is repeat and 
data.groupby(['status', 'duplicatedTransaction'])[['amount', 'amountfinancedByXepelin']].describe()

#how many times a payer is a payer
payers = data.pivot_table(columns = ['PayerId'], aggfunc = 'size').rename('repetitionPayers').to_frame()#.rename('repetitionPayers')
payers['PayerId'] = payers.index
payers = payers.reset_index(drop = True)
#How many times a receiver is a receiver
receivers = data.pivot_table(columns = ['ReceiverId'], aggfunc = 'size').rename('repetitionReceivers').to_frame()#.rename('repetitionPayers')
receivers['ReceiverId'] = receivers.index
receivers = receivers.reset_index(drop = True)
receivers
#pd.merge(data, payers, left_on='PayerId', right_index= True, how = 'inner')


data.shape

data = pd.merge(data, payers, on='PayerId', how = 'inner')
data = pd.merge(data, receivers, on='ReceiverId', how = 'inner')

data.shape

# + active=""
# We investigate how is the proportion of PAID vs FAILED on each PayerId
# -

data.loc[data['status'] == 'PROCESSING', ['status']] = 'PAID'

# +

data.groupby(['PayerId','status']).count()#.agg({'status: sum'})


# -

#Proporción de las veces que falla el Payer y el receiver
def get_proportion(data, agent):
    df = data.groupby(agent).status.value_counts(normalize = True).to_frame()#.xs('PayerId', axis = 1)
    df.columns = ['proportion'+str(agent)]
    return df.reset_index()[[agent,'proportion'+str(agent)]].drop_duplicates(subset = agent, keep = 'first')
proportion_payer = get_proportion(data, 'PayerId')
proportion_receiver = get_proportion(data, 'ReceiverId')



proportion_receiver#.drop_duplicates(subset = )
#pd.merge(data, proportion_payer, on='PayerId', how = 'inner')

data = pd.merge(data, proportion_payer, on='PayerId', how = 'inner')
data = pd.merge(data, proportion_receiver, on='ReceiverId', how = 'inner')

data.shape

corr = data[original_columns].corr()
corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)

corr = data.corr()
corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)

data.to_csv(r'‪data_ready.csv')

data.shape

#pip install pyvis
from pyvis.network import Network




net = Network(notebook = True)
net.from_nx(G)
#net.enable_physics(True)
net.show('example.html')

data['PayerId'].unique().tolist()

net = draw_network(ta['PayerId'].unique().tolist(), df, minium_weight=0, repulsion=100, spring_length=500)
net.show("match.html")
