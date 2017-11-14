# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:57:02 2017

@author: xiaojian
"""
import datetime as dt
import pytz
import pandas as pd
from math import sqrt,radians,sin,cos,atan
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pytz import timezone
import numpy as np
import csv
from matplotlib.path import Path
from netCDF4 import Dataset
from scipy import  interpolate
from matplotlib.dates import date2num,num2date
t0=datetime(2004,3,31,0,0,0)#2004,
tt=[]
for a in np.arange(22):
    tt.append(t0+timedelta(days=a*5))
t=['3/31','4/5','4/10','4/15','4/20','4/25','4/30','5/5','5/10','5/15','5/20','5/25','5/30','6/4','6/9','6/14','6/19','6/24','6/29','7/4','7/9','7/14']
r2004=[36,21,58,83,87,96,99,65,71,86,79,84,98,98,93,55,64,66,73,64,58,42]
r2004s=[64,79,42,17,13,4,1,35,29,14,20,16,1,2,7,43,36,34,27,36,11,9]
r2004n=[0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,2,0,0,0,0,31,49]

r2005=[93,98,94,45,16,5,9,21,43,62,84,91,97,98,82,80,78,87,75,86,67,70]
r2005s=[7,2,6,55,84,95,91,79,57,38,16,9,3,2,10,20,22,13,24,14,33,30]
r2005n=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,1,0,0,0]

r2006=[79,65,73,50,57,63,56,92,96,90,89,90,84,69,57,94,88,64,58,76,82,77]
r2006s=[21,35,27,50,42,37,44,8,4,10,11,10,16,5,0,2,0,11,25,24,18,23]
r2006n=[0,0,0,0,1,0,0,0,0,0,0,0,0,26,43,4,12,25,17,0,0,0]

r2007=[75,70,61,71,86,94,95,99,94,25,22,33,32,46,74,84,99,98,96,98,94,59]
r2007s=[25,30,39,29,14,6,5,1,6,75,78,67,68,54,26,16,1,1,4,2,5,41]
r2007n=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0]

r2008=[78,61,37,43,22,12,9,15,29,76,39,45,57,66,75,91,98,99,83,53,38,16]
r2008s=[21,39,63,57,78,88,91,85,71,24,61,55,43,34,25,9,2,1,17,47,62,84]
r2008n=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

r2009=[86,79,86,100,99,100,98,81,86,89,79,46,64,44,53,73,72,63,81,75,90,96]
r2009s=[13,4,6,0,1,0,1,17,14,11,21,54,36,56,47,27,28,37,18,15,4,1]
r2009n=[1,17,8,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,1,10,6,3]

r2010=[40,29,40,45,59,76,93,85,67,80,73,61,70,83,77,92,88,86,87,82,76,71]
r2010s=[60,71,60,55,41,24,7,15,33,20,26,37,30,16,21,8,9,9,12,15,24,29]
r2010n=[0,0,0,0,0,0,0,0,0,0,1,2,0,1,2,0,3,5,1,3,0,0]
"""
r2009=[90,72,53,100,97,60,54,53,12,87,71,76,65,91,28,49]
r2009s=[10,5,7,0,1,0,0,6,4,0,0,6,0,5,72,50]
r2009n=[0,23,40,0,2,40,46,41,84,13,29,18,35,4,0,1]

r2010=[73,61,46,32,54,82,52,97,75,91,88,66,81,95]
r2010s=[27,39,53,68,46,16,0,0,8,9,6,16,4,0]
r2010n=[0,0,1,0,0,2,48,3,17,0,6,18,15,5]

r2011=[94,70,82,96,66,63,8,2,12,28,81,100]
r2011s=[4,29,18,4,34,37,92,98,88,72,15,0]
r2011n=[2,1,0,0,0,0,0,2,0,0,4,0]

r2012=[77,88,96,96,97,89,95,99,90,79,89,39,64,52,60,78]
r2012s=[0,0,0,0,0,0,4,1,0,0,11,61,46,48,27,10]
r2012n=[23,12,4,4,3,11,1,0,10,21,0,0,0,0,13,12]

r2013=[88,93,91,81,34,26,63,83,60,54,81,87,50,45,48,50]
r2013s=[0,0,9,19,66,74,36,7,40,46,19,3,44,52,45,50]
r2013n=[12,7,0,0,0,0,1,10,0,0,0,10,6,3,7,0]

r2014=[40,89,42,14,32,43,54,69,24,21,6,26,65,70,71,67,81,52,64]
r2014s=[60,2,48,86,68,57,46,31,76,79,94,74,34,30,29,19,13,27,0]
r2014n=[0,9,10,0,0,0,0,0,0,0,0,0,1,0,0,14,6,21,36]

r2015=[98,100,100,88,44,62,87,89,92,5,13,62,77,76,91,85]
r2015s=[0,0,0,11,56,35,1,1,0,0,0,5,22,11,1,3]
r2015n=[2,0,0,1,0,3,12,10,8,95,87,33,1,13,8,12]
"""
'''
[73,34,57,82,96,98,100,98,99,83,73,93,94,67,49,58,87,93,83,76,84,75,52,61,45]
[90,77,82,97,97,95,69,47,42,42,44,77,73,56,85,70,83,90,65,68,77,61,60,66,68]
[78,65,64,30,53,73,34,68,78,99,73,72,69,81,97,51,43,47,46,64,71,86,75,66,85]
[89,73,60,43,98,90,79,90,85,73,81,97,41,49,51,61,39,55,72,99,74,62,77,67,65]
[96,89,87,48,63,58,9,11,29,84,76,100,90,73,73,90,99,98,71,38,58,74,83,41,22]
[89,67,68,100,90,63,45,57,25,84,75,73,73,86,45,47,62,63,81,75,75,93,78,48,43]
[68,67,66,36,62,81,54,89,75,83,79,57,78,94,84,81,83,55,54,82,81,76,67,65,70]
'''
r2010=[40,29,40,45,59,76,93,85,67,80,73,61,70,83,77,92,88,86,87,82,76,71]

m1=[36,93,79,75,78,86,40]
m2=[21,98,65,70,61,79,29]
m3=[58,94,73,61,37,86,40]
m4=[83,45,50,71,43,100,45]
m5=[87,16,57,86,22,99,59]
m6=[96,5,63,94,12,100,76]
m7=[99,9,56,95,9,98,93]
m8=[65,21,92,99,15,81,85]
m9=[71,43,96,94,29,86,67]
m10=[86,62,90,25,76,89,80]
m11=[79,84,89,22,39,79,73]
m12=[84,91,90,33,45,46,61]
m13=[98,97,84,32,57,64,70]
m14=[98,98,69,46,66,44,83]
m15=[93,82,57,74,75,53,77]
m16=[55,80,94,84,91,73,92]
m17=[64,78,88,99,98,72,88]
m18=[66,87,64,98,99,63,86]
m19=[73,75,58,96,83,81,87]
m20=[64,86,76,98,53,75,82]
m21=[58,67,82,94,38,90,76]
m22=[42,70,77,59,16,96,71]


m=[]

m.append(m1)
m.append(m2)
m.append(m3)
m.append(m4)
m.append(m5)
m.append(m6)
m.append(m7)
m.append(m8)
m.append(m9)
m.append(m10)
m.append(m11)
m.append(m12)
m.append(m13)
m.append(m14)
m.append(m15)
m.append(m16)
m.append(m17)
m.append(m18)
m.append(m19)
m.append(m20)
m.append(m21)
m.append(m22)

"""
plt.figure(figsize=(12,4))
plt.plot(tt,r2004,'*-',label='2004')
plt.plot(tt,r2005,'*-',label='2005')
plt.plot(tt,r2006,'*-',label='2006')
plt.plot(tt,r2007,'*-',label='2007')
plt.plot(tt,r2008,'*-',label='2008')
#plt.plot(tt[0:16],r2009,'*-',label='2009')
plt.title('retention of particle')
#plt.xticks(['3/31','4/5','4/10','4/15','4/20','4/25','4/30','5/5','5/10','5/15','5/20','5/25','5/30','6/4','6/9','6/14','6/19','6/24','6/29','7/4','7/9','7/14','7/19','7/24','7/29'])
plt.legend(loc='best')
plt.ylim([0,110])
"""
fig,axes=plt.subplots(1,1,sharex=True,figsize=(12,5))
'''
data2015=pd.Series(r2015,index=list(t[0:16]))
data2015.plot(marker='*',ax=axes[0],color='#000000',label='2015')
data2014=pd.Series(r2014,index=list(t[0:19]))
data2014.plot(marker='*',ax=axes[0],color='#A9A9A9',label='2014')
data2013=pd.Series(r2013,index=list(t[0:16]))
data2013.plot(marker='*',ax=axes[0],color='red',label='2013')
data2012=pd.Series(r2012,index=list(t[0:16]))
data2012.plot(marker='*',ax=axes[0],color='#FF7F50',label='2012')
data2011=pd.Series(r2011,index=list(t[0:12]))
data2011.plot(marker='*',ax=axes[0],color='#FFA500',label='2011')
data2010=pd.Series(r2010,index=list(t[0:14]))
data2010.plot(marker='*',ax=axes[0],color='#FFFF00',label='2010')
data2009=pd.Series(r2009,index=list(t[0:16]))
data2009.plot(marker='*',ax=axes[0],color='#7FFF00',label='2009')
'''

data2010=pd.Series(r2010,index=list(t))
data2010.plot(marker='o',linestyle='--',ax=axes,label='2010')
data2009=pd.Series(r2009,index=list(t))
data2009.plot(marker='x',linestyle='--',ax=axes,label='2009')

data2008=pd.Series(r2008,index=list(t))
data2008.plot(marker='x',linestyle='--',ax=axes,label='2008')#color='#006400',
data2007=pd.Series(r2007,index=list(t))
data2007.plot(marker='o',linestyle='--',ax=axes,label='2007')#color='#B0E0E6',
data2006=pd.Series(r2006,index=list(t))
data2006.plot(marker='.',linestyle='--',ax=axes,label='2006')#color='#4682B4',
data2005=pd.Series(r2005,index=list(t))
data2005.plot(marker='*',linestyle='--',ax=axes,label='2005')#color='#000080',
data2004=pd.Series(r2004,index=list(t))
data2004.plot(marker='^',linestyle='--',ax=axes,label='2004')#color='#800080',

txin=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
#axes.plot(txin,r2010,marker='o',linestyle='--',label='2010')

axes.set_title('retention of particle(after 30 days) below the sea level 15 meters')
axes.set_ylim([0,119])
#axes.set_xlim(['3/31','4/5','4/10','4/15','4/20','4/25','4/30','5/5','5/10','5/15','5/20','5/25','5/30','6/4','6/9','6/14','6/19','6/24','6/29','7/4','7/9','7/14','7/19','7/24','7/29'])
#axes.set_xlabel('a')
data=[]
for a in np.arange(len(m)):
    xx=[]
    xx.append(np.mean(m[a]))
    xx.append(np.std(m[a]))
    data.append(xx)
"""
df=pd.DataFrame(data,index=['3/31','4/5','4/10','4/15','4/20','4/25','4/30','5/5','5/10','5/15','5/20','5/25','5/30','6/4','6/9','6/14','6/19','6/24','6/29','7/4','7/9','7/14','7/19','7/24','7/29'],columns=pd.Index(['mean','std']))#,name='Genus'))
df.plot(kind='bar',ax=axes[1],alpha=0.5)
axes[1].set_title('mean and std')
#axes[1].set_xlabel('b')
plt.savefig('retentionxin',dpi=200)
"""
#axes.plot(txin,np.array(data).T[0],'.-',color='black',linewidth=3,label='mean')

axes.errorbar(txin,np.array(data).T[0],yerr=np.array(data).T[1],fmt='.-',color='black',linewidth=2,label='mean')
axes.legend(loc='best',fontsize=10)
plt.savefig('retentionxin15',dpi=200)
s=[]
for a in np.arange(len(r2004s)):
    s.append(r2004s[a])
    
    
fig,axes=plt.subplots(2,1,figsize=(12,8))#sharex=True,
plt.subplots_adjust(wspace=0.22,hspace=0.22)

data2010s=pd.Series(r2010s,index=list(t))
data2010s.plot(marker='o',linestyle='--',ax=axes[0],label='2010')
data2009s=pd.Series(r2009s,index=list(t))
data2009s.plot(marker='x',linestyle='--',ax=axes[0],label='2009')

data2008s=pd.Series(r2008s,index=list(t))
data2008s.plot(marker='x',linestyle='--',ax=axes[0],label='2008')
data2007s=pd.Series(r2007s,index=list(t))
data2007s.plot(marker='o',linestyle='--',ax=axes[0],label='2007')
data2006s=pd.Series(r2006s,index=list(t))
data2006s.plot(marker='.',linestyle='--',ax=axes[0],label='2006')
data2005s=pd.Series(r2005s,index=list(t))
data2005s.plot(marker='*',linestyle='--',ax=axes[0],label='2005')
data2004s=pd.Series(r2004s,index=list(t))
data2004s.plot(marker='^',linestyle='--',ax=axes[0],label='2004')


axes[0].set_title('The number of particles moving to the south(after 30 days) below the sea level 15 meters')
axes[0].set_ylim([0,119])

data2010n=pd.Series(r2010n,index=list(t))
data2010n.plot(marker='o',linestyle='--',ax=axes[1],label='2010')
data2009n=pd.Series(r2009n,index=list(t))
data2009n.plot(marker='x',linestyle='--',ax=axes[1],label='2009')

data2008n=pd.Series(r2008n,index=list(t))
data2008n.plot(marker='x',linestyle='--',ax=axes[1],label='2008')
data2007n=pd.Series(r2007n,index=list(t))
data2007n.plot(marker='o',linestyle='--',ax=axes[1],label='2007')
data2006n=pd.Series(r2006n,index=list(t))
data2006n.plot(marker='.',linestyle='--',ax=axes[1],label='2006')
data2005n=pd.Series(r2005n,index=list(t))
data2005n.plot(marker='*',linestyle='--',ax=axes[1],label='2005')
data2004n=pd.Series(r2004n,index=list(t))
data2004n.plot(marker='^',linestyle='--',ax=axes[1],label='2004')

axes[0].set_xlabel('a')
axes[1].set_xlabel('b')
axes[1].set_title('The number of particles moving to the north(after 30 days) below the sea level 15 meters')
axes[1].set_ylim([0,119])

s=[]
if len(r2004s)<22:
    for a in np.arange(22-len(r2004s)):
        r2004s.append(0)
        r2004n.append(0)
if len(r2005s)<22:
    for a in np.arange(22-len(r2005s)):
        r2005s.append(0)
        r2005n.append(0)
if len(r2006s)<22:
    for a in np.arange(22-len(r2006s)):
        r2006s.append(0)
        r2006n.append(0)
if len(r2007s)<22:
    for a in np.arange(22-len(r2007s)):
        r2007s.append(0)
        r2007n.append(0)
if len(r2008s)<22:
    for a in np.arange(22-len(r2008s)):
        r2008s.append(0)
        r2008n.append(0)
if len(r2009s)<22:
    for a in np.arange(22-len(r2009s)):
        r2009s.append(0)
        r2009n.append(0)
if len(r2010s)<22:
    for a in np.arange(22-len(r2010s)):
        r2010s.append(0)
        r2010n.append(0)

s=[]
s.append(r2004s)
s.append(r2005s)
s.append(r2006s)
s.append(r2007s)
s.append(r2008s)
s.append(r2009s)
s.append(r2010s)


n=[]
n.append(r2004n)
n.append(r2005n)
n.append(r2006n)
n.append(r2007n)
n.append(r2008n)
n.append(r2009n)
n.append(r2010n)

south=np.array(s).T
north=np.array(n).T
"""
fig,axes=plt.subplots(2,1,sharex=True,figsize=(30,16))
df=pd.DataFrame(south,index=t,columns=pd.Index(['2004','2005','2006','2007','2008','2009','2010']))#,name='Genus'))
df.plot(kind='bar',ax=axes[0])
df1=pd.DataFrame(north,index=t,columns=pd.Index(['2004','2005','2006','2007','2008','2009','2010']))#,name='Genus'))
df1.plot(kind='bar',ax=axes[1])
plt.savefig('gggxin',pdi=200)
"""
S=[]
if len(r2004)<25:
    for a in np.arange(25-len(r2004)):
        r2004.append(0)
if len(r2005)<25:
    for a in np.arange(25-len(r2005)):
        r2005.append(0)
if len(r2006)<25:
    for a in np.arange(25-len(r2006)):
        r2006.append(0)
if len(r2007)<25:
    for a in np.arange(25-len(r2007)):
        r2007.append(0)
if len(r2008)<25:
    for a in np.arange(25-len(r2008)):
        r2008.append(0)
if len(r2009)<25:
    for a in np.arange(25-len(r2009)):
        r2009.append(0)
if len(r2010)<25:
    for a in np.arange(25-len(r2010)):
        r2010.append(0)

S=[]
S.append(r2004)
S.append(r2005)
S.append(r2006)
S.append(r2007)
S.append(r2008)
S.append(r2009)
S.append(r2010)

SN=np.array(S).T
summ=[]
for a in np.arange(len(SN)):
    xx=0
    for b in np.arange(len(SN[0])):
        if SN[a][b]!=0:
            xx=xx+1
    summ.append(xx)
msouth=[]
mnorth=[]
stdsouth=[]
stdnorth=[]
for a in np.arange(len(south)):
    msouth.append(sum(south[a])/float(summ[a]))
    mnorth.append(sum(north[a])/float(summ[a]))
    stdsouth.append(np.std(south[a][0:summ[a]]))
    stdnorth.append(np.std(north[a][0:summ[a]]))
#fig,axes=plt.subplots(2,1,sharex=True,figsize=(12,8))
ssou=[]
ssou.append(msouth)
ssou.append(stdsouth)
ssou1=[]
ssou1.append(mnorth)
ssou1.append(stdnorth)

"""
df=pd.DataFrame(np.array(ssou).T,index=t,columns=pd.Index(['mean','std']))#,name='Genus'))
df.plot(kind='bar',ax=axes[0],alpha=0.5)
axes[0].set_ylim([0,50])
axes[0].set_title('The mean and std( the number of particles moving to the south after 15 days)')
df1=pd.DataFrame(np.array(ssou1).T,index=t,columns=pd.Index(['mean','std']))#,name='Genus'))
df1.plot(kind='bar',ax=axes[1],alpha=0.5)
axes[1].set_title('The mean and std( the number of particles moving to the north after 15 days)')
axes[1].set_ylim([0,50])
plt.savefig('mean_south_northxin',pdi=200)
"""
axes[0].errorbar(txin,msouth,yerr=stdsouth,fmt='.-',color='black',linewidth=2,label='mean')

#axes[0].plot(txin,msouth,color='black',linewidth=4,label='mean')
axes[0].legend(loc='best',fontsize=10)

axes[1].errorbar(txin,mnorth,yerr=stdnorth,fmt='.-',color='black',linewidth=2,label='mean')

#axes[1].plot(txin,mnorth,color='black',linewidth=4,label='mean')

axes[1].legend(loc='best',fontsize=10)
plt.savefig('south_northxin15',dpi=200)