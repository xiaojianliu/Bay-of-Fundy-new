# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:32:31 2017

@author: xiaojian
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:57:02 2017
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
r2004=[28,33,57,83,98,98,70,74,80,78,55,71,66,78,56,69,52,64,67,64,55,59]
r2004s=[72,66,42,17,2,2,30,26,20,22,27,7,0,0,6,10,41,36,33,30,28,21]
r2004n=[0,1,2,0,0,0,0,0,0,0,18,22,34,22,38,31,7,0,0,6,17,20]

r2005=[85,89,86,74,62,35,33,28,38,42,44,73,75,76,84,52,65,65,60,63,68,60]
r2005s=[15,4,13,26,37,65,67,72,59,57,56,25,15,8,14,48,33,30,22,23,10,0]
r2005n=[0,7,1,0,1,0,0,0,3,1,0,2,10,16,2,0,2,5,18,14,22,40]

r2006=[37,38,43,19,45,56,38,68,74,85,88,88,54,44,60,45,43,51,46,63,69,78]
r2006s=[54,59,57,81,55,43,62,38,26,15,11,12,32,19,3,2,0,1,2,1,3,11]
r2006n=[9,3,0,0,0,1,0,0,0,0,1,0,14,37,37,53,57,48,52,36,28,11]

r2007=[73,75,60,43,97,92,86,81,86,30,54,53,41,48,44,35,42,55,79,99,77,31]
r2007s=[26,15,39,57,3,5,5,19,14,70,46,45,59,52,48,17,2,4,0,0,1,41]
r2007n=[1,10,1,0,0,3,9,0,0,0,0,2,0,0,8,48,56,41,21,1,22,28]

r2008=[43,58,33,29,26,17,9,11,29,76,35,45,60,71,73,67,64,67,70,40,50,30]
r2008s=[53,42,67,71,74,83,89,89,71,24,65,55,39,29,25,8,1,5,4,11,14,52]
r2008n=[4,0,0,0,0,0,2,0,0,0,0,0,1,0,2,25,35,28,26,49,36,18]

r2009=[94,78,67,91,94,45,44,49,41,72,77,65,73,55,47,57,62,55,81,75,68,83]
r2009s=[6,1,5,0,1,0,1,9,6,1,0,35,20,45,53,42,38,45,19,9,2,0]
r2009n=[0,21,28,9,5,55,55,42,53,27,23,0,7,0,0,1,0,0,0,16,30,17]

r2010=[37,50,54,13,51,56,71,90,77,66,77,57,87,96,70,56,80,62,62,80,74,73]
r2010s=[63,50,44,52,16,12,1,0,9,18,7,21,8,0,4,7,0,5,6,0,0,1]
r2010n=[0,0,2,35,33,32,28,10,14,16,16,22,5,4,26,37,20,33,32,20,26,26]
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
m1=[28,85,37,73,43,94,37]
m2=[33,89,38,75,58,78,50]
m3=[57,86,43,60,33,67,54]
m4=[83,74,19,43,29,91,13]
m5=[98,62,45,97,26,94,51]
m6=[98,35,56,92,17,45,56]
m7=[70,33,38,86,9,44,71]
m8=[74,28,68,81,11,49,90]
m9=[80,38,74,86,29,41,77]
m10=[78,42,85,30,76,72,66]
m11=[55,44,88,54,35,77,77]
m12=[71,73,88,53,45,65,57]
m13=[66,75,54,41,60,73,87]
m14=[78,76,44,48,71,55,96]
m15=[56,84,60,44,73,47,70]
m16=[69,52,45,35,67,57,56]
m17=[52,65,43,42,64,62,80]
m18=[64,65,51,55,67,55,62]
m19=[67,60,46,79,70,81,62]
m20=[64,63,63,99,40,75,80]
m21=[55,68,69,77,50,68,74]
m22=[59,60,78,31,30,83,73]

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

axes.set_title('retention of particle(after 30 days) below the sea level 1 meters')
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
plt.savefig('retentionxin1',dpi=200)
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


axes[0].set_title('The number of particles moving to the south(after 30 days) below the sea level 1 meters')
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
axes[1].set_title('The number of particles moving to the north(after 30 days) below the sea level 1 meters')
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
plt.savefig('south_northxin1',dpi=200)