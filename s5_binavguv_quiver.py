#http://www.ngdc.noaa.gov/mgg/coast/
# coast line data extractor

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:27:09 2012

@author: vsheremet
"""
from SeaHorseLib import *
from SeaHorseTide import *
import numpy as np
from scipy import interpolate
import sys
import matplotlib.pyplot as plt
from matplotlib.path import Path
from netCDF4 import Dataset
import os
from math import sqrt,radians,sin,cos,atan
from datetime import datetime, timedelta
import matplotlib.mlab as mlab
import matplotlib.cm as cm
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

# www.ngdc.noaa.gov
# world vector shoreline ascii
FNCL='necscoast_worldvec.dat'
CL=np.genfromtxt(FNCL,names=['lon','lat'])
FN='binned_drifter12078.npz'
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
xxb,yyb = np.meshgrid(xb, yb)
fig,axes=plt.subplots(2,2,figsize=(15,10))
ub = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_mean, mask=np.isnan(vb_mean))
Q=axes[0,0].quiver(xxb,yyb,ub.T,vb.T,scale=5.)
qk=axes[0,0].quiverkey(Q,0.9,0.6,0.5, r'$0.5m/s$', fontproperties={'weight': 'bold'})
axes[0,0].plot(CL['lon'],CL['lat'],'b-')
axes[0,0].set_title('a')
axes[0,0].set_xticklabels([])
axes[0,0].text(-65.8,44.5,'Bay of Fundy',fontsize=12)
axes[0,0].plot([-68.8,-68.5],[44.4,44.7],'y-')
axes[0,0].plot([-65.8,-66.3],[44.5,44.7],'y-')
axes[0,0].plot([-67,-68],[43.4,43.5],'y-')
axes[0,0].plot([-70.8,-70],[44,42.8],'y-')
axes[0,0].plot([-70.2,-70],[43.85,43.75],'y-')
axes[0,0].plot([-69.8,-69.8],[44.25,44.1],'y-')
axes[0,0].plot([-67.5,-67.6],[45.25,45.15],'y-')
axes[0,0].plot([-66.1,-66.2],[45.25,45.25],'y-')
axes[0,0].plot([-68.85,-70],[44.6,44.7],'y-')
axes[0,0].text(-66.7,45.25,'St. John River',fontsize=7)
axes[0,0].text(-67.8,45.1,'St. Croix River',fontsize=7)
axes[0,0].text(-67.7,44.8,'Eastern',fontsize=7)
axes[0,0].text(-67.7,44.7,'Maine',fontsize=7)
axes[0,0].text(-67.7,45,'Grand Manan Island',fontsize=7)
axes[0,0].plot([-67.5,-66.8],[44.99,44.7],'y-')
axes[0,0].text(-66,44.0,'Nova Scotia',fontsize=7)
axes[0,0].axis([-67.875,-64.75,43.915,45.33])
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
xxb,yyb = np.meshgrid(xb, yb)
ub = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_mean, mask=np.isnan(vb_mean))
Q=axes[0,1].quiver(xxb,yyb,ub.T,vb.T,scale=5.)
qk=axes[0,1].quiverkey(Q,0.9,0.6,0.5, r'$0.5m/s$', fontproperties={'weight': 'bold'})
axes[0,1].plot(CL['lon'],CL['lat'],'b-')
axes[0,1].set_title('b')
axes[0,1].axis([-67.875,-64.75,43.915,45.33])
axes[0,1].set_yticklabels([])
axes[0,1].set_xticklabels([])
############################################################################################ 
FNs=[]
for filename in os.listdir(r'C:\Users\xiaojian\Desktop\Bay-of-Fundy-new-fig_4\Bay-of-Fundy-new-fig_4\driftfvcom_data51'):
    if filename!='FList.csv':
        FNs.append(filename)
kdr=0
SOURCEDIR='driftfvcom_data51/'
xi = np.arange(-76.,-56.000001,0.08)
yi = np.arange(35.,47.000001,0.08)
"""
url='''current_05hind_hourly.nc'''
ds=Dataset(url,'r').variables
url1='''gom6-grid.nc'''
t0=datetime(1858,11,17,0,0,0)
t=np.array([t0+timedelta(days=ds['ocean_time'][kt]/float(60*60*24)) for kt in np.arange(len(ds['ocean_time']))])
"""
s=0
x=[]
y=[]
u=[]
v=[]
romt=[]
while kdr in range(len(FNs)):
    FN=FNs[kdr]
    FN1=SOURCEDIR+FN
    Z=np.load(FN1)
    tdh=Z['tdh'];londh=Z['londh'];latdh=Z['latdh'];umoh=Z['umoh'];vmoh=Z['vmoh']
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
xxb,yyb = np.meshgrid(xb, yb)
ub = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_mean, mask=np.isnan(vb_mean))
Q=axes[1,0].quiver(xxb,yyb,ub.T,vb.T,scale=5.)
qk=axes[1,0].quiverkey(Q,0.9,0.6,0.5, r'$0.5m/s$', fontproperties={'weight': 'bold'})
axes[1,0].set_title('c')
axes[1,0].plot(CL['lon'],CL['lat'],'b-')
axes[1,0].axis([-67.875,-64.75,43.915,45.33])
#################################################################################
loo=np.load('lon_rom_drifter.npy')
laa=np.load('lat_rom_drifter.npy')
for a in np.arange(len(loo)):
    plt.plot(loo[a],laa[a])
axes[1,1].plot(CL['lon'],CL['lat'],'b-')
axes[1,1].set_yticklabels([])
axes[1,1].axis([-67.875,-64.75,43.915,45.33])
axes[1,1].set_title('d')
plt.savefig('Fig4',dpi=400)
plt.show()

