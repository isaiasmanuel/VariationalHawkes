from mpmath import mp
import numpy as np
import pandas as pd
import scipy as sp
import geopy.distance
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from multiprocessing import Pool, cpu_count, get_context, Process
from scipy import interpolate
import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default='browser'
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import os
from functools import partial
from scipy.interpolate import griddata


plt.rcParams['figure.dpi'] = 300

############# Define  functions


def fintegrated(i,alpha,d,Lat,Lon):
    cov=(d*mp.exp(alpha*(Datos["Magnitude"][i]-M0)))*np.diag((1,1))
    cum1=multivariate_normal.cdf((latmax,lonmax), mean=(Lat,Lon), cov=cov)
    cum2=multivariate_normal.cdf((latmin,lonmax), mean=(Lat,Lon), cov=cov)
    cum3=multivariate_normal.cdf((latmax,lonmin), mean=(Lat,Lon), cov=cov)
    cum4=multivariate_normal.cdf((latmin,lonmin), mean=(Lat,Lon), cov=cov)
    return cum1-cum2-cum3+cum4


def fintegrated2(i,alpha,d,Lat,Lon,Lim):
    cov=(d*mp.exp(alpha*(Datos["Magnitude"][i]-M0)))*np.diag((1,1))
    cum1=multivariate_normal.cdf((Lat+Lim,Lon+Lim), mean=(Lat,Lon), cov=cov)
    cum2=multivariate_normal.cdf((Lat+Lim,Lon-Lim), mean=(Lat,Lon), cov=cov)
    cum3=multivariate_normal.cdf((Lat-Lim,Lon+Lim), mean=(Lat,Lon), cov=cov)
    cum4=multivariate_normal.cdf((Lat-Lim,Lon-Lim), mean=(Lat,Lon), cov=cov)
    return cum1-cum2-cum3+cum4


def klam(M,A,alpha,M0):
    return A*mp.exp(alpha*(M-M0))

def fint(x,y,M, d,alpha,M0):
    return 1/(2*np.pi*d*mp.exp(alpha*(M-M0)))*mp.exp(-(x**2+y**2)/(2*d*np.exp(alpha*(M-M0))))

def g(t,p,c):
    return (p-1)*mp.power(c,p-1)*mp.power(t+c,-p)*(t>0)

def likelihhod(thetalike):
    A,alpha,c,p,d=np.copy(thetalike)
    print(A)
  
    if (A*bvalue*np.log(10))/(bvalue*np.log(10)-alpha)<1:
        global lik        
        def lik(j):
            value=0
            xObs=Datos["Latitude"].loc[j]
            yObs=Datos["Longitude"].loc[j]
            t= mindate+ timedelta(seconds=Diferencias[j])
            Mag=Datos["Magnitude"].loc[j]
            intg=(1-mp.power(c,(p-1))*(((maxdate-t).total_seconds())/sectoday+c)**(1-p))
            intf=fintegrated(j,alpha,d,xObs,yObs)
            for i in np.arange(j,len(Datos)-1)+1:    
                if PIJ[i,j]>0.05:
                    Lat=Datos["Latitude"].loc[i]
                    Lon=Datos["Longitude"].loc[i]
                    # Mag2=Datos["Magnitude"].loc[i]
                    value+= PIJ[i,j]*mp.log(klam(Mag,A,alpha,M0)*fint(xObs-Lat,yObs-Lon,Mag, d,alpha,M0)*g((Fechas[i]-t).total_seconds()/sectoday,p,c))
            value=value-klam(Mag,A,alpha,M0)*intg*intf#- 10000*(fintegrated2(j,alpha,d,xObs,yObs,0.5)-1)**2 #-(intf-1)**2-(intg-1)**2
            return value
        paralell= get_context("fork").Pool(cpu_count())
        results = paralell.map(lik, range(len(Datos)))
        paralell.close()
        paralell.join()
        # results=np.vectorize(lik)( range(len(Datos)))    
        value=np.sum(results)
        # print(thetalike,"\n", value)
        global thetaIteracion
        thetaIteracion=np.copy(thetalike)
    else:
        value=np.log(0)
    print(value,alpha)
    return value



from shapely.geometry import Point, Polygon

Region=np.loadtxt('./region1.txt')
Region= Polygon(Region)

def inside(row):
    point = Point(row['Longitude'], row['Latitude'])
    return Region.contains(point)  # Use polygon.covers(point) if you want to include edge points




################ Hyperparameter
mindate=datetime(2000, 1,1)
sectoday=86400
# anio=2012  #2009 #2012 # Fix to 2017
anio=2017  #2009 #2012 # Fix to 2017 max 2021
maxdate=datetime(anio, 1,1)
################ Read data GNSS
file_path = './CAYA_2000-2021_GAMIT.dat'


################ Read data Epicenter
M0=4.3
##########
########## New database
Datos=pd.read_csv('./Earthquakes.csv',delimiter=",")

Datos=Datos[Datos.apply(inside, axis=1)]



Datos=Datos.loc[ Datos["Year"]<anio]

print(np.min(Datos["Magnitude"]))
Datos=Datos.loc[ Datos["Magnitude"]>=M0]
Datos=Datos.reset_index()
print(len(Datos))

bvalue=1/np.mean(Datos["Magnitude"]-M0)*np.log10(np.exp(1))

Datos.loc[Datos["Magnitude"]>7]

for i in range(len(Datos)):
    if i==0:
        Fechas=datetime(Datos["Year"][i], Datos["Month"][i], Datos["Day"][i], Datos["Hour"][i], Datos["Minute"][i], int(Datos["Second"][i]), int(str(Datos["Second"][i])[-1])*100000)
    else :
        Fechas=np.hstack((Fechas,datetime(Datos["Year"][i], Datos["Month"][i], Datos["Day"][i], Datos["Hour"][i], Datos["Minute"][i], int(Datos["Second"][i]), int(str(Datos["Second"][i])[-1])*100000)))

Diferencias=np.zeros(len(Fechas))
for i in range(len(Diferencias)):
    Diferencias[i]=(Fechas[i]-mindate).total_seconds()

############### Exploring data
S=1 #Extra size of the square
latmin=np.min(Datos["Latitude"])-S
latmax=np.max(Datos["Latitude"])+S
lonmin=np.min(Datos["Longitude"])-S
lonmax=np.max(Datos["Longitude"])+S    
#############################



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################