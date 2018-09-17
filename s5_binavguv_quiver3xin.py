#http://www.ngdc.noaa.gov/mgg/coast/
# coast line data extractor

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:27:09 2012

@author: vsheremet
"""
import numpy as np
from SeaHorseLib import *
from datetime import *
from SeaHorseTide import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
from netCDF4 import Dataset
import os
from math import sqrt
from datetime import datetime, timedelta

def sh_bindata(x, y, z, xbins, ybins):
    """
    Bin irregularly spaced data on a rectangular grid.

    """
    ix=np.digitize(x,xbins)
    iy=np.digitize(y,ybins)
    xb=0.5*(xbins[:-1]+xbins[1:]) # bin x centers
    yb=0.5*(ybins[:-1]+ybins[1:]) # bin y centers
    zb_mean=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_median=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_std=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_num=np.zeros((len(xbins)-1,len(ybins)-1),dtype=int)    
    for iix in range(1,len(xbins)):
        for iiy in range(1,len(ybins)):
#            k=np.where((ix==iix) and (iy==iiy)) # wrong syntax
            k,=np.where((ix==iix) & (iy==iiy))
            zb_mean[iix-1,iiy-1]=np.mean(z[k])
            zb_median[iix-1,iiy-1]=np.median(z[k])
            zb_std[iix-1,iiy-1]=np.std(z[k])
            zb_num[iix-1,iiy-1]=len(z[k])
            
    return xb,yb,zb_mean,zb_median,zb_std,zb_num
"""
from netCDF4 import Dataset

# read in etopo5 topography/bathymetry.
url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
etopodata = Dataset(url)

topoin = etopodata.variables['ROSE'][:]
lons = etopodata.variables['ETOPO05_X'][:]
lats = etopodata.variables['ETOPO05_Y'][:]
# shift data so lons go from -180 to 180 instead of 20 to 380.
topoin,lons = shiftgrid(180.,topoin,lons,start=False)
"""



"""
BATHY=np.genfromtxt('necscoast_noaa.dat',dtype=None,names=['coast_lon', 'coast_lat'])
coast_lon=BATHY['coast_lon']
coast_lat=BATHY['coast_lat']
"""

#BATHY=np.genfromtxt('coastlineNE.dat',names=['coast_lon', 'coast_lat'],dtype=None,comments='>')
#coast_lon=BATHY['coast_lon']
#coast_lat=BATHY['coast_lat']


# www.ngdc.noaa.gov
# world vector shoreline ascii
FNCL='necscoast_worldvec.dat'
# lon lat pairs
# segments separated by nans
"""
nan nan
-77.953942	34.000067
-77.953949	34.000000
nan nan
-77.941035	34.000067
-77.939568	34.001241
-77.939275	34.002121
-77.938688	34.003001
-77.938688	34.003881
"""
CL=np.genfromtxt(FNCL,names=['lon','lat'])


FN='binned_drifter12078.npz'
#FN='binned_model.npz'
Z=np.load(FN) 
xb=Z['xb']
yb=Z['yb']
ub_mean=Z['ub_mean']
ub_median=Z['ub_median']
ub_std=Z['ub_std']
ub_num=Z['ub_num']
vb_mean=Z['vb_mean']
vb_median=Z['vb_median']
vb_std=Z['vb_std']
vb_num=Z['vb_num']
Z.close()

#cmap = matplotlib.cm.jet
#cmap.set_bad('w',1.)
xxb,yyb = np.meshgrid(xb, yb)
cc=np.arange(-1.5,1.500001,0.03)
#cc=np.array([-1., -.75, -.5, -.25, -0.2, -.15, -.1, -0.05, 0., 0.05, .1, .15, .2, .25, .5, .75, 1.])
fig,axes=plt.subplots(2,2,figsize=(15,10))
#plt.figure()
ub = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_mean, mask=np.isnan(vb_mean))
Q=axes[0,0].quiver(xxb,yyb,ub.T,vb.T,scale=5.)
qk=axes[0,0].quiverkey(Q,0.9,0.6,0.5, r'$0.1m/s$', fontproperties={'weight': 'bold'})

#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')

#plt.plot(coast_lon,coast_lat,'b.')
axes[0,0].plot(CL['lon'],CL['lat'],'b-')
axes[0,0].set_xlabel('a')
axes[0,0].text(-65.8,44.5,'Bay of Fundy',fontsize=12)
#axes[1].text(-67.5,41.5,'Georges Bank',fontsize=7)
axes[0,0].plot([-68.8,-68.5],[44.4,44.7],'y-')
axes[0,0].plot([-65.8,-66.3],[44.5,44.7],'y-')
axes[0,0].plot([-67,-68],[43.4,43.5],'y-')
axes[0,0].plot([-70.8,-70],[44,42.8],'y-')
axes[0,0].plot([-70.2,-70],[43.85,43.75],'y-')
axes[0,0].plot([-69.8,-69.8],[44.25,44.1],'y-')
axes[0,0].plot([-67.5,-67.6],[45.25,45.15],'y-')
axes[0,0].plot([-66.1,-66.2],[45.25,45.25],'y-')
axes[0,0].plot([-68.85,-70],[44.6,44.7],'y-')
#axes[1].text(-70.5,44.7,'Penobscot River',fontsize=7)
axes[0,0].text(-66.7,45.25,'St. John River',fontsize=7)
axes[0,0].text(-67.8,45.1,'St. Croix River',fontsize=7)
axes[0,0].text(-67.7,44.8,'Eastern',fontsize=7)
axes[0,0].text(-67.7,44.7,'Maine',fontsize=7)
#axes[1].text(-69.4,44.2,'Western',fontsize=7)
#axes[1].text(-69.4,44.1,'Maine',fontsize=7)
#axes[1].text(-70.4,44.3,'Kennebec River',fontsize=7)
#axes[1].text(-70.7,45,'Maine',fontsize=7)
#axes[0].text(-70.6,43.9,'Casco Bay',fontsize=7)
#axes[1].text(-67,43.4,'Jordan Basin',fontsize=7)
axes[0,0].text(-67.7,45,'Grand Manan Island',fontsize=7)
axes[0,0].plot([-67.5,-66.8],[44.99,44.7],'y-')
#axes[1].text(-69,44.7,'Penobscot Bay',fontsize=7)
axes[0,0].text(-66,44.0,'Nova Scotia',fontsize=7)
#axes[1].text(-70.9,44,'Wikkson Basin',fontsize=7)
axes[0,0].axis([-67.875,-64.75,43.915,45.33])#axes[0].axis([-71,-64.75,42.5,45.33])-67.875,-64.75,43.915,45.33
axes[0,0].xaxis.tick_top() 
#plt.savefig('drifter120',dpi=700)
#plt.show()
###################################################################################33


FN='binned_model12078.npz'
Z=np.load(FN) 
xb=Z['xb']
yb=Z['yb']
ub_mean=Z['ub_mean']
ub_median=Z['ub_median']
ub_std=Z['ub_std']
ub_num=Z['ub_num']
vb_mean=Z['vb_mean']
vb_median=Z['vb_median']
vb_std=Z['vb_std']
vb_num=Z['vb_num']
Z.close()

#cmap = matplotlib.cm.jet
#cmap.set_bad('w',1.)
xxb,yyb = np.meshgrid(xb, yb)
cc=np.arange(-1.5,1.500001,0.03)
#cc=np.array([-1., -.75, -.5, -.25, -0.2, -.15, -.1, -0.05, 0., 0.05, .1, .15, .2, .25, .5, .75, 1.])


#fig,axes=plt.subplots(1,1)
ub = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_mean, mask=np.isnan(vb_mean))
Q=axes[0,1].quiver(xxb,yyb,ub.T,vb.T,scale=5.)
qk=axes[0,1].quiverkey(Q,0.9,0.6,0.5, r'$0.1m/s$', fontproperties={'weight': 'bold'})

#plt.title('''Mean current derived from FVCOM (1-20m)''')

#plt.plot(coast_lon,coast_lat,'b.')
axes[0,1].plot(CL['lon'],CL['lat'],'b-')
axes[0,1].set_xlabel('b')
#axes[1].text(-65.8,44.5,'Bay of Fundy',fontsize=12)
#axes[1].text(-67.5,41.5,'Georges Bank',fontsize=7)

axes[0,1].axis([-67.875,-64.75,43.915,45.33])#axis([-71,-64.75,42.5,45.33])
#axes[1].axis([-71,-64.75,42.5,45.33])
axes[0,1].set_yticklabels([])
axes[0,1].xaxis.tick_top() 

def sh_bindata(x, y, z, xbins, ybins):
    """
    Bin irregularly spaced data on a rectangular grid.

    """
    ix=np.digitize(x,xbins)
    iy=np.digitize(y,ybins)
    xb=0.5*(xbins[:-1]+xbins[1:]) # bin x centers
    yb=0.5*(ybins[:-1]+ybins[1:]) # bin y centers
    zb_mean=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_median=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_std=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_num=np.zeros((len(xbins)-1,len(ybins)-1),dtype=int)    
    for iix in range(1,len(xbins)):
        for iiy in range(1,len(ybins)):
#            k=np.where((ix==iix) and (iy==iiy)) # wrong syntax
            k,=np.where((ix==iix) & (iy==iiy))
            zb_mean[iix-1,iiy-1]=np.mean(z[k])
            zb_median[iix-1,iiy-1]=np.median(z[k])
            zb_std[iix-1,iiy-1]=np.std(z[k])
            zb_num[iix-1,iiy-1]=len(z[k])
            
    return xb,yb,zb_mean,zb_median,zb_std,zb_num
def nearest_point(lon, lat, lons, lats, length):  #0.3/5==0.06
    '''Find the nearest point to (lon,lat) from (lons,lats),
     return the nearest-point (lon,lat)
     author: Bingwei'''
    p = Path.circle((lon,lat),radius=length)
    #numpy.vstack(tup):Stack arrays in sequence vertically
    points = np.vstack((lons.flatten(),lats.flatten())).T  
        
    insidep = []
    #collect the points included in Path.
    ii=[]
    for i in xrange(len(points)):
        if p.contains_point(points[i]):# .contains_point return 0 or 1
            insidep.append(points[i]) 
            ii.append(i)
    # if insidep is null, there is no point in the path.
    if not insidep:
        #print 'lon,lat',lon,lat
        print 'There is no model-point near the given-point.'
        raise Exception()
    #calculate the distance of every points in insidep to (lon,lat)
    distancelist = []
    for i in insidep:
        ss=sqrt((lon-i[0])**2+(lat-i[1])**2)
        distancelist.append(ss)
    # find index of the min-distance
    mindex = np.argmin(distancelist)
    # location the point
    
        
    return mindex,ii
def rot2d(x, y, ang):
    '''rotate vectors by geometric angle'''
    xr = x*np.cos(ang) - y*np.sin(ang)
    yr = x*np.sin(ang) + y*np.cos(ang)
    return xr, yr
def RataDie(yr,mo,da):
    """

RD = RataDie(yr,mo=1,da=1,hr=0,mi=0,se=0)
RD = RataDie(yr,mo=1,da=1)

returns the serial day number in the (proleptic) Gregorian calendar
or elapsed time in days since 0001-01-00.

Vitalii Sheremet, SeaHorse Project, 2008-2013.
"""
#
#    yr+=(mo-1)//12;mo=(mo-1)%12+1; # this extends mo values beyond the formal range 1-12
    RD=367*yr-(7*(yr+((mo+9)//12))//4)-(3*(((yr+(mo-9)//7)//100)+1)//4)+(275*mo//9)+da-396;
    return RD    
import os
FNs=[]
for filename in os.listdir(r'C:\Users\xiaojian\Desktop\Drift\driftx\Drift\driftfvcom_data51'):
    if filename!='FList.csv':
        
        FNs.append(filename)
SOURCEDIR='driftfvcom_data51/'
#DESTINDIR='driftfvcom_data4/'

#FList = np.genfromtxt(SOURCEDIR+'FList.csv',dtype=None,names=['FNs'],delimiter=',')
#FNs=list(FList['FNs'])
kdr=0
k=RataDie(1858,11,17)
url='''current_04hind_hourly.nc'''
ds=Dataset(url,'r').variables
url1='''gom6-grid.nc'''
ds1=Dataset(url1,'r').variables
FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])

t0=datetime(1858,11,17,0,0,0)
xi = np.arange(-76.,-56.000001,0.08)
yi = np.arange(35.,47.000001,0.08)
#tt=np.array([t0+timedelta(days=kk[kt]) for kt in np.arange(len(kk))])
t=np.array([t0+timedelta(days=ds['ocean_time'][kt]/float(60*60*24)) for kt in np.arange(len(ds['ocean_time']))])
s=0
x=[]
y=[]
u=[]
v=[]
romt=[]
while kdr in range(len(FNs)):
#while kdr in range(0,284):
#while kdr in range(9,10):
    
    FN=FNs[kdr]
    FN1=SOURCEDIR+FN
    Z=np.load(FN1)
    tdh=Z['tdh'];londh=Z['londh'];latdh=Z['latdh'];umoh=Z['umoh'];vmoh=Z['vmoh']
    #plt.plot(londh,latdh,'.')
    umom=sh_rmtide(umoh,ends=np.NaN)
    vmom=sh_rmtide(vmoh,ends=np.NaN)
    if len(londh)>1:
        for a in np.arange(len(londh)):
            if abs(umom[a])<10 or abs(vmom[a])<10:
                x.append(londh[a])
                y.append(latdh[a])
                u.append(umom[a])
                v.append(vmom[a])
                romt.append(tdh[a])
    kdr=kdr+1
np.save('romst',romt)

np.save('romslon',x)
np.save('romslat',y)
np.save('romsu',u)
np.save('romsv',v)
xb,yb,ub_mean,ub_median,ub_std,ub_num = sh_bindata(np.array(x), np.array(y), np.array(u), xi, yi)
xb,yb,vb_mean,vb_median,vb_std,vb_num = sh_bindata(np.array(x), np.array(y), np.array(v), xi, yi)
FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])
xxb,yyb = np.meshgrid(xb, yb)

#plt.figure()
ub = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_mean, mask=np.isnan(vb_mean))
Q=axes[1,0].quiver(xxb,yyb,ub.T,vb.T,scale=5.)
qk=axes[1,0].quiverkey(Q,0.9,0.6,0.5, r'$0.1m/s$', fontproperties={'weight': 'bold'})
axes[1,0].set_xlabel('c')
#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')

#plt.plot(coast_lon,coast_lat,'b.')
axes[1,0].plot(CL['lon'],CL['lat'],'b-')
axes[1,0].axis([-67.875,-64.75,43.915,45.33])
#axes[0].axis([-71,-64.75,42.5,45.33])-67.875,-64.75,43.915,45.33
axes[1,0].set_xticklabels([])#axes[2].xaxis.tick_top()  
#################################################################################
loo=np.load('lonmm.npy')
laa=np.load('latmm.npy')
for a in np.arange(len(loo)):
    plt.plot(loo[a],laa[a])
FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])
axes[1,1].plot(CL['lon'],CL['lat'],'b-')
axes[1,1].set_xticklabels([])
axes[1,1].set_yticklabels([])
axes[1,1].axis([-67.875,-64.75,43.915,45.33])
axes[1,1].set_xlabel('d')
################################################33
plt.savefig('model12011111111111112222222222222222277777777777777777788888888888xin1',dpi=400)
plt.show()

plt.figure()
ub = np.ma.array(ub_std, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_std, mask=np.isnan(vb_mean))
Q=plt.quiver(xxb,yyb,ub.T,vb.T,scale=5.)
qk=plt.quiverkey(Q,0.9,0.6,0.5, r'$0.1m/s$', fontproperties={'weight': 'bold'})

plt.title(FN[:-4]+', std')

#plt.plot(coast_lon,coast_lat,'b.')
plt.plot(CL['lon'],CL['lat'])
plt.axis([-71,-64.75,42.5,45.33])
plt.savefig('model_std120',dpi=700)
plt.show()

fig,axes=plt.subplots(1,1)#plt.figure()
for a in np.arange(len(xxb[0])):
    for b in np.arange(len(yyb)):
        if -68<xxb[0][a]<-64.75 and 44<yyb[b][0]<45.33 and ub_num[a][b]!=0:
            #plt.text(xxb[0][a],yyb[b][0],ubn[a][b],fontsize='smaller')
            plt.text(xxb[0][a],yyb[b][0],ub_num[a][b],fontsize=4)
            #axes[1,1].scatter(xxb[0][a],yyb[b][0],s=ubn[a][b]/float(100),color='red',marker='o')
plt.plot(CL['lon'],CL['lat'])
plt.axis([-68,-64.75,44,45.33])
axes.xaxis.tick_top() 
#plt.title('binned_drifter_num')
plt.savefig('binned_drifter_num1207777777777778888888888888',dpi=700)
plt.show()

plt.figure()
#plt.plot(xxb+0.015,yyb+0.015,'r-')
#plt.plot((xxb+0.015).T,(yyb+0.015).T,'r-')
#Q=plt.quiver(xxb,yyb,ub.T,vb.T,scale=5.)
#qk=plt.quiverkey(Q,0.8,0.1,0.5, r'$50cm/s$', fontproperties={'weight': 'bold'})
for a in np.arange(len(xxb[0])):
    for b in np.arange(len(yyb)):
        if -71<xxb[0][a]<-67.875 and 43.915<yyb[b][0]<45.33 and  ub_num[a][b]!=0 :
            plt.text(xxb[0][a],yyb[b][0],ub_num[a][b],fontsize=6)
plt.plot(CL['lon'],CL['lat'])
plt.axis([-71,-67.875,43.915,45.33])
plt.title('binned_drifter_num_1')
plt.savefig('binned_num120_1')
plt.show()

plt.figure()
#plt.plot(xxb+0.015,yyb+0.015,'r-')
#plt.plot((xxb+0.015).T,(yyb+0.015).T,'r-')
#Q=plt.quiver(xxb,yyb,ub.T,vb.T,scale=5.)
#qk=plt.quiverkey(Q,0.8,0.1,0.5, r'$50cm/s$', fontproperties={'weight': 'bold'})
for a in np.arange(len(xxb[0])):
    for b in np.arange(len(yyb)):
        if -67.875<xxb[0][a]<-64.75 and 43.915<yyb[b][0]<45.33 and  ub_num[a][b]!=0 :
            plt.text(xxb[0][a]-0.06,yyb[b][0],ub_num[a][b],fontsize=4.2)
plt.plot(CL['lon'],CL['lat'])
plt.axis([-67.875,-64.75,43.915,45.33])
plt.title('binned_drifter_num')
plt.savefig('binned_num120_2',dpi=700)
plt.show()
