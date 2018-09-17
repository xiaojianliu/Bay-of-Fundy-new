# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:13:45 2017

@author: bling
"""
from mpl_toolkits.basemap import Basemap  
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, atan, sqrt  

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
        #print 12
        #dla=(dmlat[a+1]-dmlat[a])*111
        #dlo=(dmlon[a+1]-dmlon[a])*(111*np.cos(dmlat[a]*np.pi/180))
        #d=sqrt(dla**2+dlo**2)#Calculate the distance between two points 
        #print model_points['lon'][a][j],model_points['lat'][a][j],dmlon[a][j],dmlat[a][j],d           
        #print 'd',d
        d=haversine(dmlon[a+1],dmlat[a+1],dmlon[a],dmlat[a])
        dd=dd+d
    #print 'dd',dd
    return dd
lon=np.load('lonmm1.npy')
lat=np.load('latmm1.npy')
FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])

length=60*0.009009009
latc=np.linspace(44.65,45.02,10)
lonc=np.linspace(-66.6,-65.93,10)

p1 = Path.circle(((lonc[5]+lonc[4])/2,(latc[5]+latc[4])/2),radius=length)
fig,axes=plt.subplots(1,1,figsize=(10,10))#figure()
cl=plt.Circle(((lonc[5]+lonc[4])/2,(latc[5]+latc[4])/2),length,alpha=0.6,color='yellow')
axes.add_patch(cl)

#################################################
p2 = Path.circle((-66.9,44.8),radius=15*0.009009009)
#fig,axes=plt.subplots(1,1,figsize=(10,10))#figure()
c2=plt.Circle((-66.9,44.8),15*0.009009009,alpha=0.6,color='red')
axes.add_patch(c2)

p3 = Path.circle((-66.7,44.4),radius=30*0.009009009)
#fig,axes=plt.subplots(1,1,figsize=(10,10))#figure()
#c3=plt.Circle((-66.7,44.4),30*0.009009009,alpha=0.6,color='green')
#axes.add_patch(c3)
"""
cl1=plt.Circle(((lonc[5]+lonc[4])/2,(latc[5]+latc[4])/2),length*100,alpha=0.4,color='green')
axes.add_patch(cl1)
"""
#plt.plot(CL['lon'],CL['lat'])
#plt.axis([-70,-65,42,47])
k=np.load('k.npy')
kk=[]
zdy=0
ydz=0
for xx in np.arange(len(k)):
    if k[xx][0]!=10000000000:
        points = np.vstack((np.array(lon[xx]).flatten(),np.array(lat[xx]).flatten())).T  

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
        
        for i in xrange(len(points)):
            if p1.contains_point(points[i]):
                insidep1.append(points[i])
            else:
                insidep2.append(points[i])
                
        zuo=0
        you=0
        xn=[]
        for a in np.arange(len(insidep2)):
            w=np.argmin(abs(insidep2[a][0]-list(lonx)))
            if laty[w]>insidep2[a][1]:
                zuo=zuo+1
                #nihao.append(1)
            else:
                you=you+1
                #if
                #nihao.append(0)
            #wo.append(nihao)
        xn.append(len(insidep2))
        xn.append(zuo)
        xn.append(you)
        kk.append(xn)
        west=[1,1]
        east=[1,1]
        if zuo>you and len(insidep2)!=0:
            plt.plot(lon[k[xx][0]][k[xx][1]:-1],lat[k[xx][0]][k[xx][1]:-1])
            zdy=zdy+1
            for nei in np.arange(len(lon[k[xx][0]][k[xx][1]:-1])):
                if p2.contains_point([lon[k[xx][0]][nei],lat[k[xx][0]][nei]]):
                    west.append(1)
            if len(west)>2:
                print '###############################33'
        if zuo<you and len(insidep2)!=0:
            plt.plot(lon[k[xx][0]][k[xx][1]:-1],lat[k[xx][0]][k[xx][1]:-1])
        
            ydz=ydz+1
            for nei1 in np.arange(len(lon[k[xx][0]][k[xx][1]:-1])):
                if p3.contains_point([lon[k[xx][0]][nei1],lat[k[xx][0]][nei1]]):
                    east.append(1)
            if len(east)>2:
                print '******************************'


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
m.drawmeridians(meridians,labels=[0,0,1,0],dashes=[1,1000],fontsize=10,zorder=0)

plt.plot([-66.3,-65.9],[44.9,44.45],zorder=2)
plt.text(-65.8,44.4,'Bay of Fundy',fontsize=12)
plt.plot([-66,-65.5],[45.5,45],'b-',zorder=2)
plt.text(-67,45.55,'North of Bay of Fundy',fontsize=12)
plt.plot([-67,-67.5],[44.3,45],'b-',zorder=2)
plt.text(-68.5,45.2,'South of Bay of Fundy',fontsize=12)
plt.plot(CL['lon'],CL['lat'],'b-')
#plt.axis([-67.875,-64.75,43.915,45.33])
plt.savefig('xxxxin',dpi=300)

    
