# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:13:45 2017

@author: bling
"""
from mpl_toolkits.basemap import Basemap  
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
import numpy as np
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
from matplotlib.dates import date2num,num2date
def haversine(lon1, lat1, lon2, lat2): 
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """   
    #print 'lon1, lat1, lon2, lat21',lon1, lat1, lon2, lat2
    #print 'lon1, lat1, lon2, lat22',lon1, lat1, lon2, lat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    #print 34
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * atan(sqrt(a)/sqrt(1-a))   
    r = 6371 
    d=c * r
    #print 'd',d
    return d
def calculate_SD(dmlon,dmlat):
    '''compare the model_points and drifter point(time same as model point)
    (only can pompare one day)!!!'''
    #print modelpoints,dmlon,dmlat,drtime
    #print len(dmlon)
    
    dd=0
    
    for a in range(len(dmlon)-1):
        
        d=haversine(dmlon[a+1],dmlat[a+1],dmlon[a],dmlat[a])
        dd=dd+d
    #print 'dd',dd
    return dd
lon=np.load('lonmm1.npy')
lat=np.load('latmm1.npy')
dzn=[]
dyn=[]
length=60*0.009009009
latc=np.linspace(44.65,45.02,10)
lonc=np.linspace(-66.6,-65.93,10)

p1 = Path.circle(((lonc[5]+lonc[4])/2,(latc[5]+latc[4])/2),radius=length)
fig,axes=plt.subplots(1,1,figsize=(10,10))#figure()

p2 = Path.circle((-66.9,44.8),radius=15*0.009009009)
"""
c2=plt.Circle((-66.9,44.8),15*0.009009009,alpha=0.6,color='red')
axes.add_patch(c2)
"""
p3 = Path.circle((-66.7,44.4),radius=30*0.009009009)
#k=np.load('k.npy')
kk=[]
zdy=0
ydz=0
n1=0
for xx in np.arange(len(lon)):
    points = np.vstack((np.array(lon[xx]).flatten(),np.array(lat[xx]).flatten())).T
    insidep11=[]
    insidep22=[]
    s1=[]
    s2=[]
    for i in xrange(len(points)):
        if p1.contains_point(points[i]):
            s1.append(i)
            insidep11.append(points[i])
        else:
            s2.append(i)
            insidep22.append(points[i])
    if len(insidep11)>0 and len(insidep22)>0:
        n1=n1+1
        
        
        lonx=np.linspace(-80,-50,1000)
        x1=-66.6
        y1=45.02
        x2=-65.93
        y2=44.65
        laty=[]
        for a in np.arange(len(lonx)):
            laty.append(((y2-y1)/(x2-x1))*(lonx[a]-x1)+y1)
        #plt.plot(lonx,laty,'^-')
        insidep1=[]
        insidep2=[]
        
        for ii in xrange(len(points)):
            if p1.contains_point(points[ii]):
                insidep1.append(points[ii])
            else:
                insidep2.append(points[ii])
        
         
        zuo=0
        you=0
        xn=[]
        #for a in np.arange(len(insidep2)):
        w=np.argmin(abs(lon[xx][s2[-1]]-list(lonx)))
        if laty[w]>lat[xx][s2[-1]]:
            zuo=zuo+1
        else:
            you=you+1
        
        xn.append(len(insidep2))
        xn.append(zuo)
        xn.append(you)
        kk.append(xn)
        
        
        if zuo>you and len(insidep2)!=0:
            #plt.plot(lon[xx][s1[-1]:-1],lat[xx][s1[-1]:-1],color='red')
            #plt.plot(lon[k[xx][0]][k[xx][1]:-1],lat[k[xx][0]][k[xx][1]:-1])
            zdy=zdy+1
            
            dz=0
            dy=0
            for nei in np.arange(len(points)):
                if p2.contains_point(points[nei]):
                    dz=dz+1
                
                else:
                    dy=dy+1
            if dz!=0:
                plt.plot(lon[xx][s1[-1]:-1],lat[xx][s1[-1]:-1],color='pink')
            else:
                plt.plot(lon[xx][s1[-1]:-1],lat[xx][s1[-1]:-1],color='red')
            dzn.append(dz)
            dyn.append(dy)
             
            
        if zuo<you and len(insidep2)!=0:
            plt.plot(lon[xx][s1[-1]:-1],lat[xx][s1[-1]:-1],color='blue')
            #plt.plot(lon[k[xx][0]][k[xx][1]:-1],lat[k[xx][0]][k[xx][1]:-1])
        
            ydz=ydz+1
            
m = Basemap(projection='cyl',llcrnrlat=43,urcrnrlat=46,\
            llcrnrlon=-69,urcrnrlon=-64,resolution='h')#,fix_aspect=False)
    #  draw coastlines
m.drawcoastlines()
m.ax=axes
m.fillcontinents(color='white',alpha=1,zorder=2)

#draw major rivers
m.drawmapboundary()
#draw major rivers
m.drawrivers()
parallels = np.arange(43,46,1.)
m.drawparallels(parallels,labels=[1,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
meridians = np.arange(-70.,-64.,1.)
m.drawmeridians(meridians,labels=[0,0,0,1],dashes=[1,1000],fontsize=10,zorder=0)
cl=plt.Circle(((lonc[5]+lonc[4])/2,(latc[5]+latc[4])/2),length,alpha=0.6,color='yellow')
axes.add_patch(cl)
n=0
for a in np.arange(len(dzn)):
    if dzn[a]!=0:
        n=n+1
print 'Total number of drifter',n1
print 'From the west side of Grand Manan Island to the southwest ',n
print 'From the east side of Grand Manan Island to the southwest ',zdy-n
print 'To the northeast',ydz
plt.savefig('Fig3',dpi=300)