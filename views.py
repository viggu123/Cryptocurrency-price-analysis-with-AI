 Failed ....!!'}from django.shortcuts import render
import pymysql
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Create your views here.
def index(request):
    return render(request,'AdminApp/index.html')
def login(request):
    return render(request,'AdminApp/Admin.html')
def LogAction(request):
    username=request.POST.get('username')
    password=request.POST.get('password')
    if username=='Admin' and password=='Admin':      
        return render(request,'AdminApp/AdminHome.html')
    else:
        context={'data':'Login
        return render(request,'AdminApp/Admin.html',context)
def home(request):
    return render(request,'AdminApp/AdminHome.html')
global df
def LoadData(request):
    global df
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df=pd.read_csv(BASE_DIR+"\\dataset\\cryptocurreny.csv")
    #data.fillna(0, inplace=True)
    context={'data':"Dataset Loaded\n"}
    
    return render(request,'AdminApp/AdminHome.html',context)
global X
global y
global X_train,X_test,y_train,y_test
def split(request):
    global X_train,X_test,y_train,y_test
    global df
    df=df.drop(columns=(['timestamp','date']))
    df.fillna(0, inplace=True)
    X=df[['open','high','low','close','volume']]
    y=df[['marketCap']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    context={"data":"Preprocess Has Done"}
    return render(request,'AdminApp/AdminHome.html',context)
global ranacc
global rfc
def runRandomForest(request):
    global ranacc
    global rfc
    rfc = RandomForestRegressor()  
    rfc.fit(X_train, y_train)
    ranacc=rfc.score(X_test, y_test)*100
    r = format(ranacc, ".2f")
    context={"data":"RandomForest Accurary: "+str(r)+"%"}
    return render(request,'AdminApp/AdminHome.html',context)

    
global adacc
global model
def runGradientboost(request):
    global adacc
    global model
    model = GradientBoostingRegressor()  
    model.fit(X_train, y_train)
    adacc=model.score(X_test, y_test)*100
    ad = format(adacc, ".2f")
    context={"data":"Gradient Boosting Accurary: "+str(ad)+"%"}
    return render(request,'AdminApp/AdminHome.html',context)
  
def runComparision(request):   
    global ranacc,adacc
    bars = ['RandomForest Accuracy','Gadient Boosting']
    height = [ranacc,adacc]
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    return render(request,'AdminApp/AdminHome.html')

def predict(request):
    return render(request,'AdminApp/Prediction.html')

def PredAction(request):
    global model
    global rfc
    o=request.POST.get('open')
    high=request.POST.get('high')
    low=request.POST.get('low')
    close=request.POST.get('close')
    volume=request.POST.get('volume')
    pred=rfc.predict([[o,high,low,close,volume]])
    context={'data':pred[0]}
    return render(request,'AdminApp/PredictedData.html',context)
        
    
        
        
    
    



    




    

