#!/usr/bin/env python
# coding: utf-8

# In[18]:

import pandas as pd
import numpy as np
import pickle
from numpy import concatenate
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from LocallyWeighted import series_to_supervised
from tensorflow.keras.models import load_model
# import fbprophet as fb
# from LocallyWeighted import LocallyWeightedRegression
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM


class EnsembledRegression:
    def __init__(self, startDate, endDate = None):
        if endDate is None and isinstance(startDate, str):
            self.predictionDate = np.datetime64(startDate)
        elif isinstance(startDate, str) and isinstance(endDate, str):
            self.predictionDate = pd.date_range(startDate, endDate)
        else:
            print(startDate, endDate)
        if isinstance(self.predictionDate, np.datetime64):
            self.startDate = np.datetime64(self.predictionDate)-9
            self.endDate = np.datetime64(self.predictionDate)-1
        else:
            self.startDate = np.datetime64(self.predictionDate[0].date())-9
            self.endDate = np.datetime64(self.predictionDate[self.predictionDate.shape[0]-1].date())-1
        with open('./models/scaler.pckl', 'rb') as fin:
            self.scaler = pickle.load(fin)
        with open('./models/resLevel_forecast_model_lag2.pckl', 'rb') as fin:
            self.modelLevel = pickle.load(fin)
        with open('./models/prsStorage_forecast_model_lag2.pckl', 'rb') as fin:
            self.modelStorage = pickle.load(fin)

    def prepareSet(self, date):
        inputFrame = pd.read_csv('./Datasets/nineYearHarangi.csv', header=0, parse_dates=True, index_col = 0)
        unScaledinputSet = inputFrame.loc[self.startDate:self.endDate].copy()
        unScaledinputSet.drop(["Present Storage(TMC)", 'Reservoir Level(TMC)', 'Outflow'], axis=1, inplace=True)
        scaled=self.scaler.transform(unScaledinputSet.values)
        scaledInput = pd.DataFrame({'Inflow':scaled[:,0],'MADIKERI':scaled[:,1],'SOMWARPET':scaled[:,2],'VIRAJPET':scaled[:,3]})
        return series_to_supervised(np.vstack((scaledInput.values,np.zeros(4))),9,0)

    def getLevelAndStorage(self, date):
        inputFrame = pd.read_csv('./Datasets/nineYearHarangi.csv', header=0, parse_dates=True,
                                 index_col=0)

        if isinstance(date, np.datetime64):
            startdate = np.datetime64(date) - 2
            enddate = np.datetime64(date) - 1
        else:
            startdate = np.datetime64(date[0].date()) - 2
            enddate = np.datetime64(date[date.shape[0] - 1].date()) - 1
        unScaledinputSet = inputFrame.loc[startdate:enddate].copy()
        dataset = unScaledinputSet.iloc[:, :4]
        number = series_to_supervised(np.vstack((dataset.values, np.zeros(4))), 2, 0)
        return np.concatenate((self.modelLevel.predict(number.values), self.modelStorage.predict(number.values)),
                              axis=1)

    def sense_val_inflow(self, x):
        if(x <= 500):
            return 0
        elif((x > 500) and (x<=1000)):
            return 1
        elif((x > 1000) and (x<=2500)):
            return 2
        elif((x > 2500) and (x<=5000)):
            return 3
        elif((x > 5000) and (x <= 7000)):
            return 4
        elif((x > 7000) and (x <= 11000)):
            return 5
        elif((x > 11000) and (x <= 22000)):
            return 6
        elif(x > 22000):
             return 7
    def predict(self, startDate=None,endDate=None):
        
        '''based on query set ,and given date range predicts result for given date with multiple models and returns ensembled result 
        query set can be single row or multiple row set, in case of multiple set use dataframe range with set of prediction seeking dates 
        '''
        if((startDate is not None) and (endDate is not None)):
            self.__init__(startDate,endDate)
        elif((startDate is not None)):
            self.__init__(startDate)
        self.queryset = self.prepareSet(self.predictionDate)
        self.levelandstorage = self.getLevelAndStorage(self.predictionDate)
        # lstm prediction
        self.n_hours = 9
        self.n_features = 4
        test_x=self.queryset.values
        lstm_test_X = test_x.reshape((test_x.shape[0], self.n_hours, self.n_features))
        # with open('/home/kishora/Documents/models/lstmInf_forecast_model_lag9.pckl', 'rb') as fin:
        #     lstm_lag9_model = pickle.load(fin)
        lstm_lag9_model=load_model('./models/lstmTensorInf_forecast_model_lag9.h5')
        lstm_inv_yhat=lstm_lag9_model.predict(lstm_test_X)
        test_X = lstm_test_X.reshape((lstm_test_X.shape[0], self.n_hours*self.n_features))
        inv_yhatx = concatenate((lstm_inv_yhat, test_X[:, -3:]), axis=1)
        inv_yhaty = self.scaler.inverse_transform(inv_yhatx)
        self.lstm_res = inv_yhaty[:,0]

        # fb prophet prediction
        fb_test=self.queryset.copy()
        varList=fb_test.columns.tolist()
        varList.insert(0,'ds')
        fb_test['ds']=self.predictionDate
        fb_test_X = pd.DataFrame(fb_test[varList])
        with open('./models/fbInf_forecast_model_lag9.pckl', 'rb') as fin:
            proph_lag9_model = pickle.load(fin)
        proph_inv_yhat = proph_lag9_model.predict(fb_test_X).values
        self.proph_res = proph_inv_yhat[:, -1]

      # lwr prediction
        lwr_test=self.queryset.values
      # print(lwr_test.shape,queryset.values.shape)
        with open('./models/lwrInf_forecast_model_lag9.pckl', 'rb') as fin:
            lwr_lag9_model = pickle.load(fin)
        self.lwr_res = lwr_lag9_model.predict(lwr_test)

      # ensembled prediction
        self.inp=pd.DataFrame()
        self.inp['lstm']=self.lstm_res
        self.inp['proph']=self.proph_res
        self.inp['lwr']=self.lwr_res
        with open('./models/ensembled_forecast_model_lag9.pckl', 'rb') as fin:
            en_lag9_model = pickle.load(fin)
        self.predictions=pd.DataFrame()
        self.predictions['lower']=en_lag9_model[0].predict(self.inp[['lstm','proph','lwr']])
        self.predictions['result']=en_lag9_model[1].predict(self.inp[['lstm','proph','lwr']])
        self.predictions['upper']=en_lag9_model[2].predict(self.inp[['lstm','proph','lwr']])
        
        self.predictions['lowLabel']=self.predictions.lower.apply(self.sense_val_inflow)
        self.predictions['resultlabel']=self.predictions.result.apply(self.sense_val_inflow)
        self.predictions['uplabel']=self.predictions.upper.apply(self.sense_val_inflow)
        self.predictions['level(ft)'] =self.levelandstorage[:,0]
        self.predictions['storage(tmc)'] =self.levelandstorage[:,1]
        self.predictions['Dates']=self.predictionDate
        self.predictions.set_index(self.predictions['Dates'],inplace=True)
        self.predictions.drop(['Dates'],axis = 1,inplace = True)
        return self.predictions
    


# In[ ]:




