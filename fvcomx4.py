# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 10:59:31 2017

@author: xiaojian
"""
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
from matplotlib.dates import date2num,num2date
starttime=dt.datetime(2008,6,24,0,0,0,0)
endtime=dt.datetime(2008,6,24,0,0,0,0)++timedelta(hours=15*24)
hourss = int(round((endtime-starttime).total_seconds()/60/60))
        
timeurl = """http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?time[0:1:333551]"""#316008]"""
url = '''http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?uwind_stress[{0}:1:{1}][{2}],vwind_stress[{0}:1:{1}][{2}]'''
            
try:
    mtime = netCDF4.Dataset(timeurl).variables['time'][:]
except:
    print '"30yr" database is unavailable!'
    raise Exception
            # get model's time horizon(UTC).
'''fmodtime = datetime(1858,11,17) + timedelta(float(mtime[0]))
emodtime = datetime(1858,11,17) + timedelta(float(mtime[-1]))
mstt = fmodtime.strftime('%m/%d/%Y %H:%M')
mett = emodtime.strftime('%m/%d/%Y %H:%M') #'''
# get number of days from 11/17/1858
#print starttime
t1 = (starttime - datetime(1858,11,17)).total_seconds()/86400 
t2 = (endtime - datetime(1858,11,17)).total_seconds()/86400
if not mtime[0]<t1<mtime[-1] or not mtime[0]<t2<mtime[-1]:
    #print 'Time: Error! Model(massbay) only works between %s with %s(UTC).'%(mstt,mett)
    print 'Time: Error! Model(30yr) only works between 1978-1-1 with 2014-1-1(UTC).'
    raise Exception()
            
tm1 = mtime-t1; #tm2 = mtime-t2
#print mtime,tm1
index1 = np.argmin(abs(tm1)); #index2 = np.argmin(abs(tm2)); print index1,index2
index2 = index1 + hourss
num=np.load('num.npy') 
ustress=[]
vstress=[]
for a in np.arange(len(num)):
    print a
    url1 = url.format(index1, index2,num[a])
    data = netCDF4.Dataset(url1).variables
    
    """                      num=[]
    for a in np.arange(len(data['lonc'][:])):
        print a
        if data['lonc'][a]>-67.875 and data['lonc'][a]<-64.75 and data['latc'][a]>43.915 and data['latc'][a]<45.33:
            num.append(a)
    """
    
    ss=np.mean(np.hstack(data['uwind_stress'][:]))
    ss1=np.mean(np.hstack(data['vwind_stress'][:]))
    ustress.append(ss)
    vstress.append(ss1)
np.save('uwind_stress2008624',ustress)
np.save('vwind_stress2008624',vstress)
    
