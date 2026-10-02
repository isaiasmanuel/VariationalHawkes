
from multiprocessing import Pool, cpu_count, get_context, Process
import pandas as pd
import matplotlib.pyplot as plt
import os 
import numpy as np
from scipy.stats import invwishart,multivariate_normal,gamma,dirichlet,poisson,truncnorm
import mpmath as mp
from datetime import datetime, timedelta
import scipy as sp

try:
    os.chdir('/home/isaias.ramirez/')
except:
    os.chdir('/Users/isaias/Documents/ArturoHawkes')

try :
    os. mkdir("./figures")
except:
    pass



def klam(M,A,alpha,M0):
    return A*mp.exp(alpha*(M-M0))


############################## Synthetic sample generation


np.random.seed(1)
c,p=.021,1.363
A,alpha,M0=0.118,1.112,4  ###### If A is 0, it is the Poisson case
# A,alpha,M0=0,1.112,4  ###### If A is 0, it is the Poisson case
d=0.0048
sectoday=86400
mindate=datetime(2000, 1,1)
anio=2010
maxdate=datetime(anio, 1,1)
Tmax=(maxdate-mindate).days/365

gamma=70 # Previous example with 15
#########
x=0
y=0
t=0
### Simulation using thinning

def cambioT(t):
    return 0.5#1/(1+np.exp(-(t-Tmax/2)))
# plt.plot(np.arange(0,Tmax,0.01),cambioT(np.arange(0,Tmax,0.01)))
# plt.show()
def cambiogamma(t):
    # return (1/(1+np.exp(-50*(t-Tmax/2)))+0.5)*10
    return gamma #10*(t<Tmax/2)*50+(t>=Tmax/2)*100
# plt.plot(np.arange(0,Tmax,0.01),cambiogamma(np.arange(0,Tmax,0.01)))
# plt.show()
def intensity(X):
    x,y,t=X
    return cambiogamma(t)*(cambioT(t)*(2/3*multivariate_normal.pdf((x,y), mean=(0,0),cov=1)+1/3*multivariate_normal.pdf((x,y), mean=(2,2),cov=1))+(1-cambioT(t))*(1/3*multivariate_normal.pdf((x,y), mean=(4.,6.),cov=1)+2/3*multivariate_normal.pdf((x,y), mean=(6,2),cov=1)))


Const=[[-5, 5], #x
  [-5, 5],#y
  [0.0, Tmax]]#t


Opt=sp.optimize.minimize(lambda theta: -intensity(theta),(0.5,0.5,Tmax/2+1), bounds=Const, method= "L-BFGS-B")
intensmax=intensity(Opt["x"])

Eventos=poisson.rvs(intensity(Opt["x"])*15*15*Tmax,size=1)[0]


Candidatos=np.vstack((np.random.uniform(-5,10,Eventos),np.random.uniform(-5,10,Eventos),np.sort(np.random.uniform(0,Tmax,Eventos)))).T

Aceptados=np.zeros(Eventos)
for i in range(Eventos):
    paccept=intensity(Candidatos[i,:])/intensmax
    Aceptados[i]=np.random.choice(2,p=(1-paccept,paccept))

len(Aceptados)
SimPoisson=Candidatos[Aceptados==1]
np.sum(SimPoisson[:,2]<2.5)/2.5
np.sum(SimPoisson[:,2]>7.5)/2.5
len(SimPoisson)

Tmin=0
TmaxUnidades=(maxdate-mindate).total_seconds()
SimPoisson[:,2]=SimPoisson[:,2]*TmaxUnidades/Tmax


M0=4
bvalue=1.2
def SimulaGR(Simulados):
    return np.reshape(M0-np.log10(1-np.random.uniform(size=len(Simulados)))/bvalue,(len(Simulados),1))



SimPoisson=np.hstack((SimPoisson,SimulaGR(SimPoisson), np.zeros((len(SimPoisson),1))))

SimPoisson.shape


def SimulaPL(Simulados):
    # return (c/(np.random.uniform(size=len(Simulados)))**(p-1)-c)*sectoday
    return c * ((1 - np.random.uniform(size=len(Simulados))) ** (-1 / (p - 1)) - 1)*sectoday




SimPoisson[:,4]=1
 


SimPoisson=SimPoisson[SimPoisson[:, 2].argsort()]
recuperafecha=np.vectorize(lambda j :mindate+ timedelta(seconds=np.min((SimPoisson[j,2],(maxdate-mindate).total_seconds()))))

recuperafecha=recuperafecha(np.arange(len(SimPoisson)))

ArregloFechas=np.zeros((len(SimPoisson),6))
for j in range(len(SimPoisson)):
    ArregloFechas[j,:]=recuperafecha[j].year,recuperafecha[j].month,recuperafecha[j].day,recuperafecha[j].hour,recuperafecha[j].minute,recuperafecha[j].second

SimPoisson=np.hstack((SimPoisson,ArregloFechas))

    
SimPoisson=pd.DataFrame(SimPoisson)
SimPoisson.columns=["Longitude","Latitude","A","Magnitude","B","Year","Month","Day","Hour","Minute","Second"]
SimPoisson=SimPoisson.drop(columns=['A', 'B'])
SimPoisson=SimPoisson.astype({'Year': 'int', "Month": 'int', "Day": 'int',"Hour": 'int', "Minute": 'int',"Second": 'int'})

SimPoisson=SimPoisson.loc[ SimPoisson["Year"]<anio]

SimPoisson=SimPoisson.reset_index(drop=True)




for i in range(len(SimPoisson)):
    if i==0:
        Fechas=datetime(SimPoisson["Year"][i], SimPoisson["Month"][i], SimPoisson["Day"][i], SimPoisson["Hour"][i], SimPoisson["Minute"][i], int(SimPoisson["Second"][i]), int(str(SimPoisson["Second"][i])[-1])*100000)
    else :
        Fechas=np.hstack((Fechas,datetime(SimPoisson["Year"][i], SimPoisson["Month"][i], SimPoisson["Day"][i], SimPoisson["Hour"][i], SimPoisson["Minute"][i], int(SimPoisson["Second"][i]), int(str(SimPoisson["Second"][i])[-1])*100000)))

Diferencias=np.zeros(len(Fechas))
for i in range(len(Diferencias)):
    Diferencias[i]=(Fechas[i]-mindate).total_seconds()
    
    
plt.scatter(Diferencias,SimPoisson["Magnitude"])
plt.show()

plt.scatter(SimPoisson["Longitude"],SimPoisson["Latitude"])
plt.show()

Datos=SimPoisson