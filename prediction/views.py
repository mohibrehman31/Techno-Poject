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





@csrf_exempt
def home(request):
    return render(request, 'index.html', {})


@csrf_exempt
def predict(request):
    # def fun(key,Optimal_Price,Optimal_hours,Airline_and_details):
    #     l = list()
    #     final_list = list()
    #     y = np.array((Optimal_Price,Optimal_hours))
    #     for i in Airline_and_details:
    #         if(i[0]==key and i[-4]>1):
    #             x = np.array((i[-1],i[-2]))
    #             l.append((np.linalg.norm(x - y),i))
    #     l.sort(key=lambda i:i[0])
    #     x=0
    #     for i,j in l:
    #         if x<5:
    #             final_list.append(j)
    #         x=x+1
    #     return final_list
    
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
            Airline_and_details = joblib.load('prediction/B/B_Airline_list_final.pkl')
            acity_enc = city_dict[acity]
            dcity_enc = city_dict[dcity]
            key = str(dcity_enc)+","+str(acity_enc)
            k_0 = np.asarray(duration[key])
            x = scaler.transform(
                [[1, dcity_enc, day, acity_enc, k_0, weekday, hour, temp_time]])
            optimal_time = model_time.predict(x)
            y = price_scaler.transform(
                [[1, dcity_enc, day, acity_enc, k_0, weekday, hour, temp_time, optimal_time]])
            optimal_price = model_price.predict(y)
            optimal_price = optimal_price[0]
            optimal_time = optimal_time[0]

            l = list()
            final_list = list()
            y = np.array((optimal_price,optimal_time))
            for i in Airline_and_details:
                if(i[0]==str(key) and i[-4]>day):
                    x = np.array((i[-1],i[-2]))
                    l.append((np.linalg.norm(x - y),i))
            l.sort(key=lambda i:i[0])
            x=0
            for i,j in l:
                if x<5:
                    final_list.append(j)
                x=x+1
            
            new_dict = dict([(value, key) for key, value in airline_dict.items()])
            result = list()
            for i in final_list:
                Dict = {
                    'Source_city':Source_city.capitalize(),
                    'Arrival_city':Arrival_city.capitalize(),
                    'Cabin':Cabin,
                    'Date':Date,
                    'Time':Time,
                    'Airline1':new_dict[i[1]],
                    'Airline2':new_dict[i[2]],
                    'stops':i[3],
                    'departure_time':i[4],
                    'Dept_date':i[5],
                    'arrival_time':i[6],
                    'optimal_hours':i[7],
                    'Price':i[8]
                }
                result.append(Dict)
            print("HII_B_END")
        if cabin == 'PE':
            print("HII_PE")
            model_time = joblib.load('prediction/PE/PE_time_predict.pkl')
            price_scaler = joblib.load('prediction/PE/PE_price_scaler.pkl')
            model_price = joblib.load('prediction/PE/PE_price_predict.pkl')
            scaler = joblib.load('prediction/PE/PE_scaler.pkl')
            airline_dict = joblib.load('prediction/PE/PE_airline_dict.pkl')
            city_dict = joblib.load('prediction/PE/PE_City_dict.pkl')
            duration = joblib.load('prediction/PE/PE_duration.pkl')
            Airline_and_details = joblib.load('prediction/PE/PE_Airline_list_final.pkl')
            acity_enc = city_dict[acity]
            dcity_enc = city_dict[dcity]
            key = str(dcity_enc)+","+str(acity_enc)
            k_0 = np.asarray(duration[key])
            x = scaler.transform([[2, dcity_enc, day, acity_enc, k_0, weekday, hour, temp_time]])
            optimal_time = model_time.predict(x)
            y = price_scaler.transform([[2, dcity_enc, day, acity_enc, k_0, weekday, hour, temp_time, optimal_time]])
            optimal_price = model_price.predict(y)
            optimal_price = optimal_price[0]
            optimal_time = optimal_time[0]

            l = list()
            final_list = list()
            y = np.array((optimal_price,optimal_time))
            for i in Airline_and_details:
                if(i[0]==str(key) and i[-4]>day):
                    x = np.array((i[-1],i[-2]))
                    l.append((np.linalg.norm(x - y),i))
            l.sort(key=lambda i:i[0])
            x=0
            for i,j in l:
                if x<5:
                    final_list.append(j)
                x=x+1
            
            new_dict = dict([(value, key) for key, value in airline_dict.items()])
            result = list()
            for i in final_list:
                Dict = {
                    'Source_city':Source_city.capitalize(),
                    'Arrival_city':Arrival_city.capitalize(),
                    'Cabin':Cabin,
                    'Date':Date,
                    'Time':Time,
                    'Airline1':new_dict[i[1]],
                    'Airline2':new_dict[i[2]],
                    'stops':i[3],
                    'departure_time':i[4],
                    'Dept_date':i[5],
                    'arrival_time':i[6],
                    'optimal_hours':i[7],
                    'Price':i[8]
                }
                result.append(Dict)
            print("HII_PE_END")
        if cabin == 'E':
            print("HII_E")
            model_time = joblib.load('prediction/E/E_time_predict.pkl')
            price_scaler = joblib.load('prediction/E/E_price_scaler.pkl')
            model_price = joblib.load('prediction/E/E_price_predict.pkl')
            scaler = joblib.load('prediction/E/E_scaler.pkl')
            airline_dict = joblib.load('prediction/E/E_airline_dict.pkl')
            city_dict = joblib.load('prediction/E/E_City_dict.pkl')
            duration = joblib.load('prediction/E/E_duration.pkl')
            Airline_and_details = joblib.load('prediction/E/E_Airline_list_final.pkl')
            acity_enc = city_dict[acity]
            dcity_enc = city_dict[dcity]
            key = str(dcity_enc)+","+str(acity_enc)
            k_0 = np.asarray(duration[key])
            x = scaler.transform([[0, dcity_enc, day, acity_enc, k_0, weekday, hour, temp_time]])
            optimal_time = model_time.predict(x)
            y = price_scaler.transform([[0, dcity_enc, day, acity_enc, k_0, weekday, hour, temp_time, optimal_time]])
            optimal_price = model_price.predict(y)
            optimal_price = optimal_price[0]
            optimal_time = optimal_time[0]

            l = list()
            final_list = list()
            y = np.array((optimal_price,optimal_time))
            for i in Airline_and_details:
                if(i[0]==str(key) and i[-4]>day):
                    x = np.array((i[-1],i[-2]))
                    l.append((np.linalg.norm(x - y),i))
            l.sort(key=lambda i:i[0])
            x=0
            for i,j in l:
                if x<5:
                    final_list.append(j)
                x=x+1
            
            new_dict = dict([(value, key) for key, value in airline_dict.items()])
            result = list()
            for i in final_list:
                Dict = {
                    'Source_city':Source_city.capitalize(),
                    'Arrival_city':Arrival_city.capitalize(),
                    'Cabin':Cabin,
                    'Date':Date,
                    'Time':Time,
                    'Airline1':new_dict[i[1]],
                    'Airline2':new_dict[i[2]],
                    'stops':i[3],
                    'departure_time':i[4],
                    'Dept_date':i[5],
                    'arrival_time':i[6],
                    'optimal_hours':i[7],
                    'Price':i[8]
                }
                result.append(Dict)
            print("HII_E_END")

        return render(request, 'airlineticket.html',{'listDict': result})


    return HttpResponse("ERROR")
