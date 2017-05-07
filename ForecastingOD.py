
# -*- coding: utf-8 -*-
"""

@author: Alfonso Fernandez Bes
"""


# Cargamos las librerias necesarias
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import urllib3
import time
import datetime

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
ts=robjects.r('ts')
forecast=importr('forecast')

from rpy2.robjects import pandas2ri
pandas2ri.activate()

t0 = time.clock()
http = urllib3.PoolManager()

# Leemos la pagina principal
url = 'http://ausmacrodata.org/Data/'
response = http.request('GET', url)
soup = BeautifulSoup(response.data, 'lxml')

# Parseamos las carpetas en las que se encuentran los datos
table = soup.find_all('table')[0] # Grab the first table
s = pd.Series()
for i in range(3,len(table.findAll('tr'))-1):
    s = s.append(pd.Series(table.findAll('tr')[i].findAll('td')[1].getText()),ignore_index=True)

s[33] = 'D13%20/'
s1 = pd.Series()
for j in range(0,len(s)):
    url1 = 'http://ausmacrodata.org/Data/'+s[j]
    response1 = http.request('GET', url1)
    soup1 = BeautifulSoup(response1.data, 'lxml')

    table1 = soup1.find_all('table')[0]
    table1_string = str(table1)
    wordList = table1_string.split()
    aux = pd.Series(wordList)
    aux1 = aux[aux.str.contains(".csv")]
    aux1 = aux1.reset_index()
    for i in range(0,len(aux1)):
        aux1.ix[i,0] = aux1.ix[i,0].partition('"')[-1].rpartition('"')[0]
    aux1=aux1[0]
    aux1 = s[j]+aux1
    s1 = s1.append(aux1,ignore_index=True)

del(s,aux,aux1,i,j,table1,table1_string,response1,table,response,url,url1,wordList)
t1 = time.clock()
print(round(t1 - t0,2)," segundos")


# Ejecutamos un 15ª parte de los datos 2653
t0_pred = time.clock()
newDF = pd.DataFrame(columns=["Archivo","Date","Forecast","LowerC","UpperC"])
for j in range(0,2653):
    url2='http://ausmacrodata.org/Data/'+s1[j]
    df = pd.read_csv(url2, skipinitialspace=True, usecols=['date','value'])
    if type(df.index[0]) != tuple:
        if type(df.ix[0,0])!=str:
            df = df.reset_index()
            df.columns = ['date','value','inf']
            df = df[['date','value']]
        FMT = '%m/%Y'
        d = (datetime.datetime.strptime(df.date[1], FMT) - datetime.datetime.strptime(df.date[0], FMT)).days
        # Estacionalidad de la serie (suele ser 12 = mensual, 4=Trimestral, 3=Cuatrimestral, 2=Semestral, 1=Anual)
        freq = round(12/(d/30))
        
        # Calculamos el siguiente elemento de la serie que será la predicción
        i_d = datetime.timedelta(days=(12/freq)*30+7)
        l_d = datetime.datetime.strptime(df.date[len(df)-1], FMT)
        h_pred = l_d+i_d
        h_pred_sal = datetime.datetime.strftime(h_pred,format=FMT)
    
        df=df["value"]
        rdata=ts(df.values,frequency=freq)
        fit=forecast.auto_arima(rdata)
        forecast_output=forecast.forecast(fit,h=1,level=(95.0))
        forecast1=pd.Series(forecast_output[3])[0][0]
        lowerpi1=np.array(pd.Series(forecast_output[4]))[0][0]
        upperpi1=np.array(pd.Series(forecast_output[5]))[0][0]
        #salida compuesta por nombre fichero, horizonte, prediccion e IC
        newDF.loc[len(newDF)]=list([s1[j],h_pred_sal,forecast1,lowerpi1,upperpi1])
        #salida
        del(df,d,forecast1,freq,h_pred,h_pred_sal,l_d,lowerpi1,upperpi1,url2)
