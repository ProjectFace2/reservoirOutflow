#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import concatenate
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pickle


# In[2]:


'''
In this method, we use all the lag days and lead days and construct a dataframe, where
the lag and lead days are represented in a single row
Using these features we can predict the class of the present day with the data from the previous days
This holds good for all the rows in the dataframe
'''

#dataset preparation with lag
def series_to_supervised(data, lag_days=1, lead_days=1, dropnan=True):
    no_of_features = 1 if type(data) is list else data.shape[1]
    # print(no_of_features)    p
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(lag_days, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(no_of_features)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, lead_days):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(no_of_features)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(no_of_features)]
# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[5]:


class LocallyWeightedRegression:
    def __init__(self):
        return
        
    def setup(self,X,y):
        lag_val = int(X.shape[1]/4)
        begin_year = ''
        if(lag_val==9):
            begin_year='2011-01-10'
        else:
            begin_year='2011-01-05'
        self.train_x,self.train_y = self.init_train( lag_val, begin_year)

    def fit(self, train_x, train_y):
        self.setup(train_x,train_y)

    def init_train( self, lag_val=4, begin_year='2011-01-05'):
        df = pd.read_csv('Datasets/allYearLabeledHarangi.csv',header=0,parse_dates=True,index_col=0)
        x=df.drop(["Present Storage(TMC)",'Reservoir Level(TMC)','Outflow','Label'],axis = 1)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(x.values)
        reshaped=pd.DataFrame({'Inflow':scaled[:,0],'MADIKERI':scaled[:,1],'SOMWARPET':scaled[:,2],'VIRAJPET':scaled[:,3]})
        idx = pd.date_range('2011-01-01', '2018-12-31') 
        reshaped['Dates']=idx
        df=reshaped
        df['month'] = pd.DatetimeIndex(df["Dates"]).month
        df['year'] = pd.DatetimeIndex(df["Dates"]).year
        mask = (df['month'] <= 12)
        mask1 = (df['year'] >= 2011)&(df['year'] <= 2018)
        df = df.loc[mask]
        df = df.loc[mask1]
        df.set_index('Dates',inplace = True)
        df.drop(['month','year'],axis = 1,inplace = True)
#         lag_val = 4
        values = df.values
        values = values.astype('float32')
        reframed = series_to_supervised(values, lag_val, 1)#lag of 4 days
        reframed.drop(reframed.columns[[-1,-2,-3]], axis=1, inplace=True)
        idx = pd.date_range(begin_year, '2018-12-31') 
        reframed['Dates']=idx
        reframed['month']=pd.DatetimeIndex(reframed['Dates']).month
        reframed=reframed.sort_values(by=['month','Dates'])
        reframed.drop(columns=['month','Dates'],inplace=True)
        values = reframed.values
        train_x = values[:,:-1]
        Inflow = values[:,-1]
        train_y = Inflow.reshape((train_x.shape[0],1))
        return train_x,train_y
   
    #locally weighted regression

    def lwr1(self,x0, inp, out, k):
        m,n = np.shape(inp)
        ypred = np.zeros(m)    
        ypred = x0 * self.beta(x0, inp, out, k)
        #print("The final prediction is :",ypred)
        return ypred
    
    def beta(self,point, inp, out, k):
        wt = self.kernal(point, inp, k)
        #print("The weight of betas is",wt)
        beta_val = (inp.T * (wt*inp)).I * inp.T * wt * out
        #print("The weight is beta value is",beta_val)
        return beta_val
    
    def kernal(self,point, inp, k):
        l,b = np.shape(inp)
        weights = np.mat(np.eye((l)))
        #print(weights)    
        for i in range(l):
            #print(point.shape,inp[i].shape)
            diff = point - inp[i]
            weights[i,i] = np.exp(np.dot(diff,diff.T) / (-2.0 * (k**2)))
        return weights
    
    def call_Lwr(self, test_x,train_x,train_y,k=0):
            ypred = []
            train_X = train_x
            train_y = train_y
            for i in test_x:
                ypred.append(self.lwr1(i, train_X, train_y, 7.15))
            ypred = np.array(ypred).reshape(len(ypred),1)
            inv_yhat = np.concatenate((ypred, test_x[:, -3:]), axis=1)
            out=self.scaler.inverse_transform(inv_yhat) 
            out=out[:,0]
            return out

    def predict(self, test_x):
        return self.call_Lwr(test_x,self.train_x,self.train_y)

