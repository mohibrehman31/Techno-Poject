from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import zipfile



global optimal_price, optimal_time, Source_city,Arrival_city, Cabin, Date,Time

optimal_price = ' '
optimal_time = ' '    
Source_city = ' '
Arrival_city = ' '
Cabin = ' '
Date = ' '
Time = ' '


# @csrf_exempt
# def neW(request):
#     if request.method == 'POST':
#         print(optimal_price,'HELLO')
#         return render(request, 'airlineticket.html',{'optimal_price':optimal_price,'optimal_time':optimal_time,'Source_city':Source_city,'Arrival_city':Arrival_city,'Cabin':Cabin,'Date':Date,'Time':Time})



@csrf_exempt
def home(request):
    return render(request, 'index.html',{})


@csrf_exempt
def predict(request):

    if request.method == 'POST':
        dcity = request.POST['Source_city'].lower()
        acity = request.POST['Arrival_city'].lower()
        cabin = request.POST['Cabin']
        date = request.POST['Date']
        time = request.POST['Time']

        Source_city = dcity
        Arrival_city = acity
        Cabin = cabin
        Date = date
        Time = time

        month = pd.to_datetime(date).month
        hour = time.split(':')[0]
        day = pd.to_datetime(date).day
        weekday = pd.to_datetime(date).dayofweek
        time = int(hour)
        if time >= 16 and time < 21:
            temp_time = 1
        if time >= 21 or time < 5:
            temp_time = 3
        if time >= 5 and time < 11:
            temp_time = 2
        if time >= 11 and time < 16:
            temp_time = 0
        day = np.asarray(day)
        hour = np.asarray(hour)

        if cabin == 'B':
            print("HII_B")
            model_time = joblib.load('prediction/B/B_time_predict.pkl')
            price_scaler = joblib.load('prediction/B/B_price_scaler.pkl')
            model_price = joblib.load('prediction/B/B_price_predict.pkl')
            scaler = joblib.load('prediction/B/B_scaler.pkl')
            airline_dict = joblib.load('prediction/B/B_airline_dict.pkl')
            city_dict = joblib.load('prediction/B/B_City_dict.pkl')
            duration = joblib.load('prediction/B/B_duration.pkl')
            acity_enc = city_dict[acity]
            dcity_enc = city_dict[dcity]
            k_0 = str(dcity_enc)+","+str(acity_enc)
            k_0 = np.asarray(duration[k_0])
            x = scaler.transform([[1 ,dcity_enc,day,acity_enc,k_0,weekday,hour,temp_time]])
            optimal_time = model_time.predict(x)
            y = price_scaler.transform([[1 ,dcity_enc,day,acity_enc,k_0,weekday,hour,temp_time,optimal_time]])
            optimal_price = model_price.predict(y)
            print('Optimal hours = ',optimal_time,'Optimal Price = ',optimal_price)
            print("HII_B_END")
            optimal_price = optimal_price[0]
            optimal_time = optimal_time[0]

        if cabin == 'PE':
            print("HII_PE")
            model_time = joblib.load('prediction/PE/PE_time_predict.pkl')
            price_scaler = joblib.load('prediction/PE/PE_price_scaler.pkl')
            model_price = joblib.load('prediction/PE/PE_price_predict.pkl')
            scaler = joblib.load('prediction/PE/PE_scaler.pkl')
            airline_dict = joblib.load('prediction/PE/PE_airline_dict.pkl')
            city_dict = joblib.load('prediction/PE/PE_City_dict.pkl')
            duration = joblib.load('prediction/PE/PE_duration.pkl')
            acity_enc = city_dict[acity]
            dcity_enc = city_dict[dcity]
            k_0 = str(dcity_enc)+","+str(acity_enc)
            k_0 = np.asarray(duration[k_0])
            x = scaler.transform([[1 ,dcity_enc,day,acity_enc,k_0,weekday,hour,temp_time]])
            optimal_time = model_time.predict(x)
            y = price_scaler.transform([[1 ,dcity_enc,day,acity_enc,k_0,weekday,hour,temp_time,optimal_time]])
            optimal_price = model_price.predict(y)
            print('Optimal hours = ',optimal_time,'Optimal Price = ',optimal_price)
            print("HII_PE_END")
            dic = [{
            'result' : 'Hello'
            }]

        if cabin == 'E':
            print("HII_E")
            model_time = joblib.load('prediction/E/E_time_predict.pkl')
            price_scaler = joblib.load('prediction/E/E_price_scaler.pkl')
            model_price = joblib.load('prediction/E/E_price_predict.pkl')
            scaler = joblib.load('prediction/E/E_scaler.pkl')
            airline_dict = joblib.load('prediction/E/E_airline_dict.pkl')
            city_dict = joblib.load('prediction/E/E_City_dict.pkl')
            duration = joblib.load('prediction/E_duration.pkl')
            acity_enc = city_dict[acity]
            dcity_enc = city_dict[dcity]
            k_0 = str(dcity_enc)+","+str(acity_enc)
            k_0 = np.asarray(duration[k_0])
            x = scaler.transform([[1 ,dcity_enc,day,acity_enc,k_0,weekday,hour,temp_time]])
            optimal_time = model_time.predict(x)
            y = price_scaler.transform([[1 ,dcity_enc,day,acity_enc,k_0,weekday,hour,temp_time,optimal_time]])
            optimal_price = model_price.predict(y)
            print('Optimal hours = ',optimal_time,'Optimal Price = ',optimal_price)
            print("HII_E_END")

        return render(request, 'airlineticket.html',{'optimal_price':optimal_price,'optimal_time':optimal_time,'Source_city':Source_city,'Arrival_city':Arrival_city,'Cabin':Cabin,'Date':Date,'Time':Time})


    return HttpResponse("ERROR")
