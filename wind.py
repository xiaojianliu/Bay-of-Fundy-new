# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 22:21:37 2017

@author: xiaojian
"""
import numpy as np
#from pydap.client import open_url
import matplotlib.pyplot as plt

from datetime import *
#from scipy import interpolate
import sys

import shutil
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import os
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
FN='necscoast_worldvec.dat'
from mpl_toolkits.basemap import Basemap 
CL=np.genfromtxt(FN,names=['lon','lat'])
fig=plt.figure(figsize=(15,10))
ax1=fig.add_subplot(2,2,1)
#fig,axes=plt.subplots(3,1,figsize=(7,15))
plt.subplots_adjust(wspace=0.05,hspace=0.05)
m4=np.load('roms2008_4_30_to_5_15.npy')#'m_ps2011-2010_630.npy'roms2008_4_30_to_5_15.npy
p4=m4.tolist()
for a in np.arange(len(p4)):
    ax1.scatter(p4[a]['lon'][0],p4[a]['lat'][0],color='green',s=10)
    ax1.scatter(p4[a]['lon'][-1],p4[a]['lat'][-1],color='red',s=10)
    ax1.plot([p4[a]['lon'][0],p4[a]['lon'][-1]],[p4[a]['lat'][0],p4[a]['lat'][-1]],'y-',linewidth=1)
ax1.scatter(p4[a]['lon'][0],p4[a]['lat'][0],color='green',label='start point')
ax1.scatter(p4[a]['lon'][-1],p4[a]['lat'][-1],color='red',label='end point')
ax1.legend(loc='best')
m = Basemap(projection='cyl',llcrnrlat=43,urcrnrlat=46,\
            llcrnrlon=-69,urcrnrlon=-64,resolution='h')#,fix_aspect=False)
    #  draw coastlines
m.drawcoastlines()
m.ax=ax1
m.fillcontinents(color='white',alpha=1,zorder=2)

#draw major rivers
m.drawmapboundary()
#draw major rivers
m.drawrivers()
parallels = np.arange(43,46,1.)
m.drawparallels(parallels,labels=[1,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
meridians = np.arange(-70.,-64.,1.)
m.drawmeridians(meridians,labels=[0,0,1,0],dashes=[1,1000],fontsize=10,zorder=0)

'''
axes[0].axis([-68.875,-64.75,43.515,45.33])
axes[0].xaxis.tick_top() 
axes[0].set_xlabel('2008_4_30_to_5_15')
'''
ax1.plot(CL['lon'],CL['lat'],linewidth=1)
m8=np.load('roms2008_5_25_to_6_9.npy')#'m_ps2011-2010_630.npy'
p8=m8.tolist()
ax2=fig.add_subplot(2,2,2)

for a in np.arange(len(p8)):
    ax2.scatter(p8[a]['lon'][0],p8[a]['lat'][0],color='green',s=10)
    ax2.scatter(p8[a]['lon'][-1],p8[a]['lat'][-1],color='red',s=10)
    ax2.plot([p8[a]['lon'][0],p8[a]['lon'][-1]],[p8[a]['lat'][0],p8[a]['lat'][-1]],'y-',linewidth=1)
#axes[1].axis([-68.875,-64.75,43.515,45.33])
ax2.xaxis.tick_top() 
ax2.set_xlabel('b    2008_5_25_to_6_9')
m.drawcoastlines()
m.ax=ax2
m.fillcontinents(color='white',alpha=1,zorder=2)

#draw major rivers
m.drawmapboundary()
#draw major rivers
m.drawrivers()
#parallels = np.arange(43,46,1.)
#m.drawparallels(parallels,labels=[1,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
meridians = np.arange(-70.,-64.,1.)
m.drawmeridians(meridians,labels=[0,0,1,0],dashes=[1,1000],fontsize=10,zorder=0)
ax2.set_xticklabels([])
ax2.plot(CL['lon'],CL['lat'],linewidth=1)
m15=np.load('roms2008_7_4_to_7_19.npy')#'m_ps2011-2010_630.npy'
p15=m15.tolist()
ax3=fig.add_subplot(2,2,3)

for a in np.arange(len(p15)):
    ax3.scatter(p15[a]['lon'][0],p15[a]['lat'][0],color='green',s=10)
    ax3.scatter(p15[a]['lon'][-1],p15[a]['lat'][-1],color='red',s=10)
    ax3.plot([p15[a]['lon'][0],p15[a]['lon'][-1]],[p15[a]['lat'][0],p15[a]['lat'][-1]],'y-',linewidth=1)
#axes[2].axis([-68.875,-64.75,43.515,45.33])
ax3.set_xticklabels([])
m.drawcoastlines()
m.ax=ax3
m.fillcontinents(color='white',alpha=1,zorder=2)
ax1.set_xlabel('c    2008_4_30_to_5_15')

#draw major rivers
m.drawmapboundary()
#draw major rivers
m.drawrivers()
parallels = np.arange(43,46,1.)
m.drawparallels(parallels,labels=[1,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
#meridians = np.arange(-70.,-64.,1.)
#m.drawmeridians(meridians,labels=[0,0,1,0],dashes=[1,1000],fontsize=10,zorder=0)
ax3.set_xlabel('c    2008_7_4_to_7_19')
ax3.plot(CL['lon'],CL['lat'],linewidth=1)
plt.savefig('2008-2008-2008xin',dpi=400)

fig,axes=plt.subplots(3,2,figsize=(14,15))
plt.subplots_adjust(wspace=0.1,hspace=0.1)

num=np.load('num.npy')
latc=np.load('gom3.latc.npy')
lonc=np.load('gom3.lonc.npy')
ustress=np.load('uwind_stress20085_25.npy')
vstress=np.load('vwind_stress20085_25.npy')
"""
ustress1=np.load('uwind_stress1.npy')
vstress1=np.load('vwind_stress1.npy')
ustress2=np.load('uwind_stress2.npy')
vstress2=np.load('vwind_stress2.npy')
"""
y=[]
x=[]
for a in np.arange(len(num)):
    y.append(latc[num[a]])
    x.append(lonc[num[a]])
uw=[]
vw=[]
for a in np.arange(len(ustress)):
    uw.append(ustress[a])
    vw.append(vstress[a])
uw20085=uw
vw20085=vw
"""
for a in np.arange(len(ustress1)):
    uw.append(ustress1[a])
    vw.append(vstress1[a])
for a in np.arange(len(ustress2)):
    uw.append(ustress2[a])
    vw.append(vstress2[a])
"""
xi = np.arange(-76.,-56.000001,0.08)
yi = np.arange(35.,47.000001,0.08)

xb,yb,ub_mean,ub_median,ub_std,ub_num = sh_bindata(np.array(x), np.array(y), np.array(uw), xi, yi)
xb,yb,vb_mean,vb_median,vb_std,vb_num = sh_bindata(np.array(x), np.array(y), np.array(vw), xi, yi)

xxb,yyb = np.meshgrid(xb, yb)

ub = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_mean, mask=np.isnan(vb_mean))
Q=axes[1,0].quiver(xxb,yyb,ub.T,vb.T,scale=2.)
qk=axes[1,0].quiverkey(Q,0.9,0.6,0.2, r'$0.1pa$', fontproperties={'weight': 'bold'})

#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')

#plt.plot(coast_lon,coast_lat,'b.')
axes[0,0].plot(CL['lon'],CL['lat'])
axes[1,0].set_xlabel('c    2008_5_25_to_6_9_wind')
#axes[0,0].set_xticklabels([])
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
############################################################################
ustress2008=np.load('uwind_stress2008.npy')
vstress2008=np.load('vwind_stress2008.npy')
ustress20081=np.load('uwind_stress20081.npy')
vstress20081=np.load('vwind_stress20081.npy')
ustress20082=np.load('uwind_stress20082.npy')
vstress20082=np.load('vwind_stress20082.npy')
y=[]
x=[]
for a in np.arange(len(num)):
    y.append(latc[num[a]])
    x.append(lonc[num[a]])
uw=[]
vw=[]
for a in np.arange(len(ustress2008)):
    uw.append(ustress2008[a])
    vw.append(vstress2008[a])
for a in np.arange(len(ustress20081)):
    uw.append(ustress20081[a])
    vw.append(vstress20081[a])
for a in np.arange(len(ustress20082)):
    uw.append(ustress20082[a])
    vw.append(vstress20082[a])
xi = np.arange(-76.,-56.000001,0.08)
yi = np.arange(35.,47.000001,0.08)

xb1,yb1,ub_mean1,ub_median1,ub_std1,ub_num1 = sh_bindata(np.array(x), np.array(y), np.array(uw), xi, yi)
xb1,yb1,vb_mean1,vb_median1,vb_std1,vb_num1 = sh_bindata(np.array(x), np.array(y), np.array(vw), xi, yi)

xxb1,yyb1 = np.meshgrid(xb1, yb1)

ub1 = np.ma.array(ub_mean1, mask=np.isnan(ub_mean1))
vb1 = np.ma.array(vb_mean1, mask=np.isnan(vb_mean1))
Q1=axes[0,0].quiver(xxb1,yyb1,ub1.T,vb1.T,scale=2.)
qk=axes[0,0].quiverkey(Q1,0.9,0.6,0.2, r'$0.1pa$', fontproperties={'weight': 'bold'})
axes[0,0].set_xlabel('a    2008_4_30_to_5_15_wind')
axes[1,0].set_xticklabels([])
#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')

#plt.plot(coast_lon,coast_lat,'b.')
axes[1,0].plot(CL['lon'],CL['lat'])

#axes[1].text(-70.9,44,'Wikkson Basin',fontsize=7)
axes[1,0].axis([-67.875,-64.75,43.915,45.33])#axes[0].axis([-71,-64.75,42.5,45.33])-67.875,-64.75,43.915,45.33
axes[0,0].xaxis.tick_top() 
##################################################################################
ustress2015=np.load('uwind_stress20087_4.npy')
vstress2015=np.load('vwind_stress20087_4.npy')
"""
ustress20151=np.load('uwind_stress20151.npy')
vstress20151=np.load('vwind_stress20151.npy')
ustress20152=np.load('uwind_stress20152.npy')
vstress20152=np.load('vwind_stress20152.npy')
"""
y=[]
x=[]
for a in np.arange(len(num)):
    y.append(latc[num[a]])
    x.append(lonc[num[a]])
uw=[]
vw=[]
for a in np.arange(len(ustress2015)):
    uw.append(ustress2015[a])
    vw.append(vstress2015[a])
"""
for a in np.arange(len(ustress20151)):
    uw.append(ustress20151[a])
    vw.append(vstress20151[a])
for a in np.arange(len(ustress20152)):
    uw.append(ustress20152[a])
    vw.append(vstress20152[a])
"""
xi = np.arange(-76.,-56.000001,0.08)
yi = np.arange(35.,47.000001,0.08)

xb2,yb2,ub_mean2,ub_median2,ub_std2,ub_num2 = sh_bindata(np.array(x), np.array(y), np.array(uw), xi, yi)
xb2,yb2,vb_mean2,vb_median2,vb_std2,vb_num2 = sh_bindata(np.array(x), np.array(y), np.array(vw), xi, yi)

xxb2,yyb2 = np.meshgrid(xb2, yb2)

ub2 = np.ma.array(ub_mean2, mask=np.isnan(ub_mean2))
vb2 = np.ma.array(vb_mean2, mask=np.isnan(vb_mean2))
Q1=axes[2,0].quiver(xxb2,yyb2,ub2.T,vb2.T,scale=2.)
qk=axes[2,0].quiverkey(Q1,0.9,0.6,0.2, r'$0.1pa$', fontproperties={'weight': 'bold'})

#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')

#plt.plot(coast_lon,coast_lat,'b.')
axes[2,0].plot(CL['lon'],CL['lat'])


axes[2,0].set_xlabel('e    2008_7_4_to_7_19 wind')
axes[2,0].set_xticklabels([])
#axes[1].text(-67.5,41.5,'Georges Bank',fontsize=7)
axes[2,0].axis([-67.875,-64.75,43.915,45.33])#axes[0].axis([-71,-64.75,42.5,45.33])-67.875,-64.75,43.915,45.33
#axes[2,0].xaxis.tick_top() 
######################################################################
xx2004=np.load('roms_lon_point.npy')
yy2004=np.load('roms_lat_point.npy')
lu2004=np.load('lu2008_5_25.npy')
lv2004=np.load('lv2008_5_25.npy')
"""
lu20041=np.load('lu1.npy')
lv20041=np.load('lv1.npy')
"""
#lu2=np.load('lu20152.npy')
#lv2=np.load('lv20152.npy')
uu2004=[]
vv2004=[]
for a in np.arange(len(lu2004)):
    uu2004.append(lu2004[a])
    vv2004.append(lv2004[a])
'''
for a in np.arange(len(lu20041)):
    uu2004.append(lu20041[a])
    vv2004.append(lv20041[a])

for a in np.arange(len(lu2)):
    uu.append(lu2[a])
    vv.append(lv2[a])
'''
x=[]
y=[]
u=[]
v=[]
for a in np.arange(len(uu2004)):
    if abs(uu2004[a])<10 or abs(vv2004[a])<10:
        x.append(xx2004[a])
        y.append(yy2004[a])
        u.append(uu2004[a])
        v.append(vv2004[a])
xi = np.arange(-76.,-56.000001,0.08)
yi = np.arange(35.,47.000001,0.08)
u2004=u
v2004=v
xb3,yb3,ub_mean3,ub_median3,ub_std3,ub_num3 = sh_bindata(np.array(x), np.array(y), np.array(u), xi, yi)
xb3,yb3,vb_mean3,vb_median3,vb_std3,vb_num3 = sh_bindata(np.array(x), np.array(y), np.array(v), xi, yi)
FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])
xxb3,yyb3 = np.meshgrid(xb3, yb3)

ub3 = np.ma.array(ub_mean3, mask=np.isnan(ub_mean3))
vb3 = np.ma.array(vb_mean3, mask=np.isnan(vb_mean3))
Q3=axes[1,1].quiver(xxb3,yyb3,ub3.T,vb3.T,scale=3.)
qk=axes[1,1].quiverkey(Q3,0.9,0.6,0.3, r'$0.1m/s$', fontproperties={'weight': 'bold'})

#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')
axes[1,1].axis([-67.875,-64.75,43.915,45.33])#axes[0].axis([-71,-64.75,42.5,45.33])-67.875,-64.75,43.915,45.33
axes[0,1].xaxis.tick_top() 
axes[1,1].set_yticklabels([])
axes[1,1].set_xticklabels([])
axes[1,1].set_xlabel('d    2008_5_25_to_6_9 ocean current')
#plt.plot(coast_lon,coast_lat,'b.')
axes[1,1].plot(CL['lon'],CL['lat'])
###############################################################33
xx2008=np.load('roms_lon_point.npy')
yy2008=np.load('roms_lat_point.npy')
lu2008=np.load('lu2008.npy')
lv2008=np.load('lv2008.npy')
lu20081=np.load('lu20081.npy')
lv20081=np.load('lv20081.npy')
#lu2=np.load('lu20152.npy')
#lv2=np.load('lv20152.npy')
uu2008=[]
vv2008=[]
for a in np.arange(len(lu2008)):
    uu2008.append(lu2008[a])
    vv2008.append(lv2008[a])
for a in np.arange(len(lu20081)):
    uu2008.append(lu20081[a])
    vv2008.append(lv20081[a])
'''
for a in np.arange(len(lu2)):
    uu.append(lu2[a])
    vv.append(lv2[a])
'''
x=[]
y=[]
u=[]
v=[]
for a in np.arange(len(uu2008)):
    if abs(uu2008[a])<10 or abs(vv2008[a])<10:
        x.append(xx2008[a])
        y.append(yy2008[a])
        u.append(uu2008[a])
        v.append(vv2008[a])
xi = np.arange(-76.,-56.000001,0.08)
yi = np.arange(35.,47.000001,0.08)
xb4,yb4,ub_mean4,ub_median4,ub_std4,ub_num4 = sh_bindata(np.array(x), np.array(y), np.array(u), xi, yi)
xb4,yb4,vb_mean4,vb_median4,vb_std4,vb_num4 = sh_bindata(np.array(x), np.array(y), np.array(v), xi, yi)
FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])
xxb4,yyb4 = np.meshgrid(xb4, yb4)

ub4 = np.ma.array(ub_mean4, mask=np.isnan(ub_mean4))
vb4 = np.ma.array(vb_mean4, mask=np.isnan(vb_mean4))
Q4=axes[0,1].quiver(xxb4,yyb4,ub4.T,vb4.T,scale=3.)
qk=axes[0,1].quiverkey(Q4,0.9,0.6,0.3, r'$0.1m/s$', fontproperties={'weight': 'bold'})

#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')
axes[0,1].axis([-67.875,-64.75,43.915,45.33])#axes[0].axis([-71,-64.75,42.5,45.33])-67.875,-64.75,43.915,45.33
axes[0,1].xaxis.tick_top() 
axes[0,1].set_yticklabels([])
#axes[0,1].set_xticklabels([])
axes[0,1].set_xlabel('b    2008_4_30_to_5_15 ocean current')
#plt.plot(coast_lon,coast_lat,'b.')
axes[0,1].plot(CL['lon'],CL['lat'])
##################################################################################
xx2015=np.load('roms_lon_point.npy')
yy2015=np.load('roms_lat_point.npy')
lu2015=np.load('lu2008_7_4.npy')
lv2015=np.load('lv2008_7_4.npy')
"""
lu20151=np.load('lu20151.npy')
lv20151=np.load('lv20151.npy')
lu20152=np.load('lu20152.npy')
lv20152=np.load('lv20152.npy')
"""
uu2015=[]
vv2015=[]
for a in np.arange(len(lu2015)):
    uu2015.append(lu2015[a])
    vv2015.append(lv2015[a])
'''
for a in np.arange(len(lu20151)):
    uu2015.append(lu20151[a])
    vv2015.append(lv20151[a])

for a in np.arange(len(lu20152)):
    uu2015.append(lu20152[a])
    vv2015.append(lv20152[a])
'''
x=[]
y=[]
u=[]
v=[]
for a in np.arange(len(uu2015)):
    if abs(uu2015[a])<10 or abs(vv2015[a])<10:
        x.append(xx2015[a])
        y.append(yy2015[a])
        u.append(uu2015[a])
        v.append(vv2015[a])
u2015=u
v2015=v
xi = np.arange(-76.,-56.000001,0.08)
yi = np.arange(35.,47.000001,0.08)
xb5,yb5,ub_mean5,ub_median5,ub_std5,ub_num5 = sh_bindata(np.array(x), np.array(y), np.array(u), xi, yi)
xb5,yb5,vb_mean5,vb_median5,vb_std5,vb_num5 = sh_bindata(np.array(x), np.array(y), np.array(v), xi, yi)
FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])
xxb5,yyb5 = np.meshgrid(xb5, yb5)

ub5 = np.ma.array(ub_mean5, mask=np.isnan(ub_mean5))
vb5 = np.ma.array(vb_mean5, mask=np.isnan(vb_mean5))
Q5=axes[2,1].quiver(xxb5,yyb5,ub5.T,vb5.T,scale=3.)
qk=axes[2,1].quiverkey(Q5,0.9,0.6,0.3, r'$0.1m/s$', fontproperties={'weight': 'bold'})

#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')
axes[2,1].axis([-67.875,-64.75,43.915,45.33])#axes[0].axis([-71,-64.75,42.5,45.33])-67.875,-64.75,43.915,45.33
axes[2,1].xaxis.tick_top() 
axes[2,1].set_yticklabels([])
axes[2,1].set_xticklabels([])
axes[2,1].set_xlabel('f    2008_7_4_to_7_19 ocean current')
#plt.plot(coast_lon,coast_lat,'b.')
axes[2,1].plot(CL['lon'],CL['lat'])
plt.savefig('2015-5-15xin',dpi=200)


fig,axes=plt.subplots(1,2,figsize=(14,5))
xb6,yb6,ub_mean6,ub_median6,ub_std6,ub_num6 = sh_bindata(np.array(x), np.array(y), np.array(uw)-np.array(uw20085), xi, yi)
xb6,yb6,vb_mean6,vb_median6,vb_std6,vb_num6 = sh_bindata(np.array(x), np.array(y), np.array(vw)-np.array(vw20085), xi, yi)
xxb6,yyb6 = np.meshgrid(xb6, yb6)
ub6 = np.ma.array(ub_mean6, mask=np.isnan(ub_mean6))
vb6 = np.ma.array(vb_mean6, mask=np.isnan(vb_mean6))
Q6=axes[0].quiver(xxb6,yyb6,ub6.T,vb6.T,scale=0.25)
axes[0].plot(CL['lon'],CL['lat'])
qk=axes[0].quiverkey(Q6,0.9,0.6,0.025, r'$0.1pa$', fontproperties={'weight': 'bold'})
axes[0].axis([-67.875,-64.75,43.915,45.33])#[-67.3,-66.3,44.6,45.2]

xb7,yb7,ub_mean7,ub_median7,ub_std7,ub_num7 = sh_bindata(np.array(x), np.array(y), np.array(u2015)-np.array(u2004), xi, yi)
xb7,yb7,vb_mean7,vb_median7,vb_std7,vb_num7 = sh_bindata(np.array(x), np.array(y), np.array(v2015)-np.array(v2004), xi, yi)
xxb7,yyb7 = np.meshgrid(xb7, yb7)
ub7 = np.ma.array(ub_mean7, mask=np.isnan(ub_mean7))
vb7 = np.ma.array(vb_mean7, mask=np.isnan(vb_mean7))
Q6=axes[1].quiver(xxb7,yyb7,ub7.T,vb7.T,scale=1)
axes[1].plot(CL['lon'],CL['lat'])
qk=axes[1].quiverkey(Q6,0.9,0.6,0.1, r'$0.1m/s$', fontproperties={'weight': 'bold'})
axes[1].axis([-67.875,-64.75,43.915,45.33])#[-67.3,-66.3,44.6,45.2]





"""
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].contour(xxb6,yyb6,ub6.T,500)
#axes[0].colorbar()
axes[0].plot(CL['lon'],CL['lat'])
axes[0].axis([-67.875,-64.75,43.915,45.33])
axes[0].contour(xxb6,yyb6,ub6.T,500)
#axes[0].colorbar()
axes[0].plot(CL['lon'],CL['lat'])
axes[0].axis([-67.875,-64.75,43.915,45.33])
"""








