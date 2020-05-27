#!/usr/bin/env python
import pandas as pd
import numpy as np
import pickle
from numpy import concatenate
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from LocallyWeighted import series_to_supervised
from tensorflow.keras.models import load_model

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

    def getLevelAndStorage(self, date, second):
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

        numberInput = np.concatenate((number.values, second), axis=1)
        return np.concatenate((self.modelLevel.predict(numberInput), self.modelStorage.predict(numberInput)),
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


    def generateOutflow(self, x):
        max_water_in_cusecs = 93402.77
        tmc = 11574.874
        if x < max_water_in_cusecs and x > 82000:
            crossing_offset_inflow = x - 82000
            res = (crossing_offset_inflow) * .6
            return 1, (crossing_offset_inflow) / tmc, res
        elif x > max_water_in_cusecs:
            crossing_offset_inflow = x - 82000
            res = (crossing_offset_inflow) * .4
            return 2, (crossing_offset_inflow) / tmc, res
        else:
            return 0, 0, 0

    def outflow_provider(self, inflow, date, tmc=11574.874):
        inputFrame = pd.read_csv('./Datasets/nineYearHarangi.csv', header=0, parse_dates=True,index_col=0)

        if isinstance(date, np.datetime64):
            startdate = np.datetime64(date) - 2
            enddate = np.datetime64(date) - 1
        else:
            startdate = np.datetime64(date[0].date()) - 2
            enddate = np.datetime64(date[date.shape[0] - 1].date()) - 1
        unScaledinputSet = inputFrame.loc[startdate:enddate].copy()
        dataset = unScaledinputSet.iloc[:, :4]
        number = series_to_supervised(np.vstack((dataset.values, np.zeros(4))), 2, 0)
        number['Dates'] = date
        number = number.set_index(number['Dates'])
        number = number.drop(columns='Dates')
        max_water_in_cusecs = 93402.77
        max_in_feet = 2859
        max_in_tmc = 8.07
        tmc_inflow = number.iloc[:, -4:-3]
        dam_height = number.iloc[:, -3:-2]
        inflow_into_dam = tmc_inflow * tmc
        inflow_today = pd.DataFrame(inflow_into_dam['var1(t-1)'].values + inflow.values)
        inflow_today['Dates'] = date
        inflow_today = inflow_today.set_index(inflow_today['Dates'])
        inflow_today = inflow_today.drop(columns='Dates')
        inflow_today['state'] = inflow_today[0].apply(lambda x: self.generateOutflow(x)[0])
        inflow_today['excess(tmc)'] = inflow_today[0].apply(lambda x: self.generateOutflow(x)[1])
        inflow_today['outlet'] = inflow_today[0].apply(lambda x: self.generateOutflow(x)[2])
        return inflow_today.drop(columns=[0])


    def predict(self, startDate=None,endDate=None):
        
        '''based on query set ,and given date range predicts result for given date with multiple models and returns ensembled result 
        query set can be single row or multiple row set, in case of multiple set use dataframe range with set of prediction seeking dates 
        '''
        if((startDate is not None) and (endDate is not None)):
            self.__init__(startDate,endDate)
        elif((startDate is not None)):
            self.__init__(startDate)
        self.queryset = self.prepareSet(self.predictionDate)

        # lstm prediction
        self.n_hours = 9
        self.n_features = 4
        test_x = self.queryset.values
        lstm_test_X = test_x.reshape((test_x.shape[0], self.n_hours, self.n_features))
        # with open('/home/kishora/Documents/models/lstmInf_forecast_model_lag9.pckl', 'rb') as fin:
        #     lstm_lag9_model = pickle.load(fin)
        lstm_lag9_model = load_model('./models/lstmTensorInf_forecast_model_lag9.h5')
        lstm_inv_yhat = lstm_lag9_model.predict(lstm_test_X)
        test_X = lstm_test_X.reshape((lstm_test_X.shape[0], self.n_hours * self.n_features))
        inv_yhatx = concatenate((lstm_inv_yhat, test_X[:, -3:]), axis=1)
        inv_yhaty = self.scaler.inverse_transform(inv_yhatx)
        self.lstm_res = inv_yhaty[:, 0]

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
        self.predictions = pd.DataFrame()
        self.predictions['lower'] = en_lag9_model[0].predict(self.inp[['lstm', 'proph', 'lwr']])
        self.predictions['result'] = en_lag9_model[1].predict(self.inp[['lstm', 'proph', 'lwr']])
        self.predictions['upper'] = en_lag9_model[2].predict(self.inp[['lstm', 'proph', 'lwr']])

        # self.predictions['lowLabel']=self.predictions.lower.apply(self.sense_val_inflow)
        self.predictions['resultlabel'] = self.predictions.result.apply(self.sense_val_inflow)
        # self.predictions['uplabel']=self.predictions.upper.apply(self.sense_val_inflow)
        second = self.predictions['result'].values.reshape(self.predictions['result'].shape[0], 1)
        secondlow = self.predictions['lower'].values.reshape(self.predictions['lower'].shape[0], 1)
        secondup = self.predictions['upper'].values.reshape(self.predictions['upper'].shape[0], 1)
        self.levelandstorage = self.getLevelAndStorage(self.predictionDate, second)
        self.levelandstoragelow = self.getLevelAndStorage(self.predictionDate, secondlow)
        self.levelandstorageup = self.getLevelAndStorage(self.predictionDate, secondup)
        self.predictions['level(ft)'] = self.levelandstorage[:, 0]
        self.predictions['storage(tmc)Low'] = self.levelandstoragelow[:, 1]
        self.predictions['storage(tmc)'] = self.levelandstorage[:, 1]
        self.predictions['storage(tmc)Up'] = self.levelandstorageup[:, 1]
        self.predictions['Dates'] = self.predictionDate
        self.predictions.set_index(self.predictions['Dates'], inplace=True)
        self.predictions.drop(['Dates'], axis=1, inplace=True)
        self.outflowSuggestion = self.outflow_provider(self.predictions.result, self.predictionDate)
        self.predictions['Excess(tmc)'] = self.outflowSuggestion['excess(tmc)']
        self.predictions['Suggested Outflow'] = self.outflowSuggestion['outlet']
        self.predictions['out Status'] = self.outflowSuggestion['state']
        return self.predictions

# In[ ]:




