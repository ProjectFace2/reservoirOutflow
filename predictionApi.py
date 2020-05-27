from LocallyWeighted import LocallyWeightedRegression
from EnsembleRegression import EnsembledRegression
import base64
# tb._SYMBOLIC_SCOPE.value = True
import pandas as pd
import numpy as np
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import flask
from flask import request, jsonify,render_template,Response
app = flask.Flask(__name__,template_folder='/home/kishora/Documents/reservoirOutflow/templates',static_url_path='/home/kishora/Documents/reservoirOutflow/statics',static_folder='/home/kishora/Documents/reservoirOutflow/statics')
app.config["DEBUG"] = True
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (30, 10)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
# mpl.rcParams['xtick.rotation']=45
mpl.rcParams['legend.fontsize'] = 15
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/viz', methods=['GET'])
def visualize():
    return render_template('vizTemplate.html')

@app.route('/harangi', methods=['GET'])
def send_harangi():
    return render_template('harangiTemplate.html')

@app.route('/about', methods=['GET'])
def send_about():
    return render_template('aboutTemplate.html')


@app.route('/viewRaw',methods=['GET','POST'])
def rawData():
    year = request.form.get('year', type=str)
    month = request.form.get('month', type=str)
    if (year == '*'):
        ybegin, yend = 2011, 2019
    else:
        ybegin, yend = int(year), int(year)

    if (month == '*'):
        begin, end = 1, 12
        figval = 2
    elif (month == '1'):
        begin, end = 6, 12
        figval = 2
    elif (month == '2'):
        begin, end = 1, 5
        figval = 1
    inputFrame = pd.read_csv('./Datasets/nineYearHarangi.csv', header=0, parse_dates=True,
                             index_col=0)
    inputFrame=inputFrame.loc[(inputFrame.index.year >= ybegin)&(inputFrame.index.year <= yend)&(inputFrame.index.month >= begin)&(inputFrame.index.month <= end)]
    inputFrame.index = inputFrame.index.astype(str)
    ind = inputFrame.index.values
    inputFrame = inputFrame.round(decimals=2)
    temp = inputFrame.to_dict('records')
    columnNames = inputFrame.columns.values
    return render_template('rawdataTemplate.html', records=temp, colnames=columnNames, indexset=ind)

@app.route('/visualize',methods=['GET','POST'])
def showViz():
    year = request.form.get('year',type=str)
    month = request.form.get('month',type=str)
    figdata_png = base64.b64encode(plot_viz_png(year,month).getvalue()).decode('ascii')
    return render_template('imagetemplate.html',plot=True,name=figdata_png)

def plot_viz_png(year,month):
    fig = create_viz_figure(year,month)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    output.seek(0)
    return output

def create_viz_figure(year,month):
    pd.plotting.register_matplotlib_converters()
    if(year=='*'):
        ybegin,yend=2011,2019
    else:
        ybegin,yend=int(year),int(year)

    if (month == '*'):
        begin, end = 1, 12
        figval=2
    elif (month == '1'):
        begin, end = 6, 12
        figval=2
    elif (month == '2'):
        begin, end = 1, 5
        figval=1
    inputFrame = pd.read_csv('./Datasets/nineYearHarangi.csv', header=0, parse_dates=True,
                             index_col=0)
    fig = Figure()

    axis = fig.add_subplot(figval, 2, 1)
    axis2 = fig.add_subplot(figval, 2, 2)
    if(figval==1):
        axis.set_ylabel('Cusecs')
        axis2.set_ylabel('TMC')
        inputFrame.loc[(inputFrame.index.year >= ybegin)&(inputFrame.index.year <= yend)&(inputFrame.index.month >= begin)&(inputFrame.index.month <= end)].plot(y=['Inflow'], ax=axis, legend=True)
        inputFrame.loc[(inputFrame.index.year >= ybegin) & (inputFrame.index.year <= yend) & (inputFrame.index.month >= begin) & (inputFrame.index.month <= end)].plot(y=['Present Storage(TMC)'], ax=axis2, legend=True)
    elif(figval==2):
        axis.set_ylabel('Cusecs')
        axis2.set_ylabel('Cusecs')
        axis3 = fig.add_subplot(figval, 2, 3)
        axis4  =fig.add_subplot(figval, 2, 4)
        axis3.set_ylabel('TMC')
        axis4.set_ylabel('Rain in mm')
        inputFrame.loc[(inputFrame.index.year >= ybegin) & (inputFrame.index.year <= yend) & (inputFrame.index.month >= begin) & (
                inputFrame.index.month <= end)].plot(y=['Inflow'], ax=axis, legend=True)
        inputFrame.loc[(inputFrame.index.year >= ybegin) & (inputFrame.index.year <= yend) & (inputFrame.index.month >= begin) & (
                inputFrame.index.month <= end)].plot(y=['Outflow'], ax=axis2, legend=True)
        inputFrame.loc[(inputFrame.index.year >= ybegin) & (inputFrame.index.year <= yend) & (inputFrame.index.month >= begin) & (
                inputFrame.index.month <= end)].plot(y=['Present Storage(TMC)'], ax=axis3, legend=True)
        inputFrame.loc[(inputFrame.index.year >= ybegin) & (inputFrame.index.year <= yend) & (inputFrame.index.month >= begin) & (
                inputFrame.index.month <= end)].plot(y=['MADIKERI','SOMWARPET','VIRAJPET'], ax=axis4, legend=True)
    return fig


@app.route('/predict',methods=['GET','POST'])
def predictions():
    startDate = request.form.get('startdate',type=str)
    # print(startDate)
    model = EnsembledRegression(startDate)
    # print(type(request.args.get('enddate', type=str)), request.args.get('enddate', type=str))
    if(request.form.get('enddate',type=str)!=''):
        endDate = request.form.get('enddate',type=str)
        results = model.predict(startDate, endDate)
        results.index = results.index.astype(str)
        figdata_png = base64.b64encode(plot_png(results.copy(), startDate, endDate).getvalue()).decode('ascii')
    else:
        results = model.predict(startDate)
        results.index = results.index.astype(str)
        figdata_png = base64.b64encode(plot_png(results.copy(), startDate).getvalue()).decode('ascii')


    # jsonres = results.to_dict(orient='index')
    # y=jsonify(jsonres)
    # del model
    # y.status_code=200
    # df = pd.read_csv("DemoData.csv")
    results=results.round(decimals=2)
    results = results.drop(columns=['storage(tmc)Up','storage(tmc)Low'])
    temp = results.to_dict('records')
    ind = results.index.values
    columnNames = results.columns.values[:-1]
    # figdata_png = base64.b64encode(plot_png(startDate,endDate).getvalue()).decode('ascii')

    return render_template('tableTemplate.html', records=temp, colnames=columnNames,indexset=ind,plot=True, name=figdata_png)
# @app.route('/plot.png')
def plot_png(results, startDate,EndDate=None):
    fig = create_figure(results, startDate,EndDate)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    output.seek(0)
    return output

def create_figure(results, startDate, endDate=None):
    startDate=np.datetime64(startDate)
    fig = Figure()
    outputFrame=results
    inputFrame = pd.read_csv('./Datasets/nineYearHarangi.csv', header=0, parse_dates=True,index_col=0)
    axis = fig.add_subplot(1, 2, 1)
    axis2 = fig.add_subplot(1, 2, 2)
    beginDate = np.datetime64(startDate)-9
    iFrame=inputFrame.loc[beginDate:startDate-1]
    iFrame.index = iFrame.index.astype(str)
    label1='inflow'
    label2='storage level'
    make_subplots(fig,'upper','lower',axis,iFrame.index.values,iFrame.Inflow.values,outputFrame.index.values,outputFrame.result.values,outputFrame,label1)
    make_subplots(fig,'storage(tmc)Up','storage(tmc)Low',axis2, iFrame.index.values, iFrame['Present Storage(TMC)'].values, outputFrame.index.values, outputFrame['storage(tmc)'].values,outputFrame,label2)
    return fig

def make_subplots(fig,up,low,axis,x,y,x1,y1,outputFrame,labelname):
    X = np.concatenate((x, x1))
    Y = np.concatenate((y, y1))
    # iFrame.plot(y=['Inflow'], ax=axis, legend=True, marker='*')
    lower = np.insert(outputFrame[low].values, 0, y[-1])
    upper = np.insert(outputFrame[up].values, 0, y[-1])
    lo, up = min(np.concatenate((lower, y))), max(np.concatenate((upper, y)))
    axis.set_ylim([lo - (lo * .25), up + (up * .25)])
    axis.set_xticklabels(X, rotation=30)
    # print(X[:x.shape[0]], X[x.shape[0] - 1:])
    axis.plot(X[:x.shape[0]], Y[:x.shape[0]], label=labelname, c='blue', marker='*')
    axis.plot(X[x.shape[0] - 1:], Y[x.shape[0] - 1:], c='red', marker='*', label='Prediction')
    axis.fill_between(X[x.shape[0] - 1:], upper, lower, facecolor='yellow', label='Interval')
    axis.legend()
    # return fig

if __name__ == '__main__':
    app.run(debug=True)