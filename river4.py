# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 06:13:11 2017

@author: xiaojian
"""

import numpy as np
import datetime as dt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import csv
data2008='20082.csv'
#mend_list='bar1.csv'
d2008= np.genfromtxt(data2008,dtype=None,names=['ids'],delimiter=',')
l2008=[]
t2008=[]

#t=np.linspace(t1,t2,3933)
for a in np.arange(len(d2008['ids'])):
    l2008.append(float(d2008['ids'][a][35:-2]))
    t2008.append(dt.datetime(int(d2008['ids'][a][14:18]),int(d2008['ids'][a][19:21]),int(d2008['ids'][a][22:24]),int(d2008['ids'][a][25:27]),int(d2008['ids'][a][28:30]),0,0))
######################################
data2009='2009.csv'
#mend_list='bar1.csv'
d2009= np.genfromtxt(data2009,dtype=None,names=['ids'],delimiter=',')
l2009=[]
t2009=[]

#t=np.linspace(t1,t2,3933)
for a in np.arange(len(d2009['ids'])):
    l2009.append(float(d2009['ids'][a][35:-2]))
    t2009.append(dt.datetime(int(d2009['ids'][a][14:18]),int(d2009['ids'][a][19:21]),int(d2009['ids'][a][22:24]),int(d2009['ids'][a][25:27]),int(d2009['ids'][a][28:30]),0,0))
##################################################
data2010='2010.csv'
#mend_list='bar1.csv'
d2010= np.genfromtxt(data2010,dtype=None,names=['ids'],delimiter=',')
l2010=[]
t2010=[]

#t=np.linspace(t1,t2,3933)
for a in np.arange(len(d2010['ids'])):
    l2010.append(float(d2010['ids'][a][35:-2]))
    t2010.append(dt.datetime(int(d2010['ids'][a][14:18]),int(d2010['ids'][a][19:21]),int(d2010['ids'][a][22:24]),int(d2010['ids'][a][25:27]),int(d2010['ids'][a][28:30]),0,0))
##################################################
data2004='2004.csv'
#mend_list='bar1.csv'
d2004= np.genfromtxt(data2004,dtype=None,names=['ids'],delimiter=',')
l2004=[]
t2004=[]

#t=np.linspace(t1,t2,3933)
for a in np.arange(len(d2004['ids'])):
    l2004.append(float(d2004['ids'][a][35:-7]))
    t2004.append(dt.datetime(int(d2004['ids'][a][14:18]),int(d2004['ids'][a][19:21]),int(d2004['ids'][a][22:24]),int(d2004['ids'][a][25:27]),int(d2004['ids'][a][28:30]),0,0))
##################################################
data2005='2005.csv'
#mend_list='bar1.csv'
d2005= np.genfromtxt(data2005,dtype=None,names=['ids'],delimiter=',')
l2005=[]
t2005=[]

#t=np.linspace(t1,t2,3933)
for a in np.arange(len(d2005['ids'])):
    l2005.append(float(d2005['ids'][a][35:-7]))
    t2005.append(dt.datetime(int(d2005['ids'][a][14:18]),int(d2005['ids'][a][19:21]),int(d2005['ids'][a][22:24]),int(d2005['ids'][a][25:27]),int(d2005['ids'][a][28:30]),0,0))
##################################################
data2006='2006.csv'
#mend_list='bar1.csv'
d2006= np.genfromtxt(data2006,dtype=None,names=['ids'],delimiter=',')
l2006=[]
t2006=[]

#t=np.linspace(t1,t2,3933)
for a in np.arange(len(d2006['ids'])):
    l2006.append(float(d2006['ids'][a][35:-7]))
    t2006.append(dt.datetime(int(d2006['ids'][a][14:18]),int(d2006['ids'][a][19:21]),int(d2006['ids'][a][22:24]),int(d2006['ids'][a][25:27]),int(d2006['ids'][a][28:30]),0,0))
##################################################
data2007='2007.csv'
#mend_list='bar1.csv'
d2007= np.genfromtxt(data2007,dtype=None,names=['ids'],delimiter=',')
l2007=[]
t2007=[]

#t=np.linspace(t1,t2,3933)
for a in np.arange(len(d2007['ids'])):
    l2007.append(float(d2007['ids'][a][35:-7]))
    t2007.append(dt.datetime(int(d2007['ids'][a][14:18]),int(d2007['ids'][a][19:21]),int(d2007['ids'][a][22:24]),int(d2007['ids'][a][25:27]),int(d2007['ids'][a][28:30]),0,0))
##################################################

#time=np.load('time.npy')
s=np.load('s.npy')
fm=np.load('fm.npy')

t1=dt.datetime(2008,4,1,0,0,0)
t=t1
time=[]
time.append(t1)
while t<dt.datetime(2008,7,31,0,0,0):
    t=t+timedelta(minutes=15)
    time.append(t)
tt=[]    
for a in np.arange(len(s)):
    tt.append(time[s[a]])
plt.figure(figsize=(10,5))
plt.plot(t2008,l2008)
plt.plot(tt,fm)
ttt=[]
fmm=[]
for a in np.arange(len(tt)):
    if tt[a]>dt.datetime(2008,4,25,23,0,0):
        ttt.append('''%s/%s'''%(tt[a].month,tt[a].day))
        fmm.append(fm[a])
fmmm=[]
tttt=[]
for a in np.arange(len(t2008)):
    if t2008[a]>dt.datetime(2008,4,25,23,0,0):
        tttt.append('''%s/%s'''%(t2008[a].month,t2008[a].day))
        fmmm.append(l2008[a])
tttt2009=[]
fmmm2009=[]
for a in np.arange(len(t2009)):
    if t2009[a]>dt.datetime(2009,4,25,23,0,0):
        tttt2009.append('''%s/%s'''%(t2009[a].month,t2009[a].day))
        fmmm2009.append(l2009[a])
tttt2010=[]
fmmm2010=[]
for a in np.arange(len(t2010)):
    if t2010[a]>dt.datetime(2010,4,25,23,0,0):
        tttt2010.append('''%s/%s'''%(t2010[a].month,t2010[a].day))
        fmmm2010.append(l2010[a])
tttt2007=[]
fmmm2007=[]
for a in np.arange(len(t2007)):
    if t2007[a]>dt.datetime(2007,4,25,23,0,0):
        tttt2007.append('''%s/%s'''%(t2007[a].month,t2007[a].day))
        fmmm2007.append(l2007[a])
tttt2006=[]
fmmm2006=[]
for a in np.arange(len(t2006)):
    if t2006[a]>dt.datetime(2006,4,25,23,0,0):
        tttt2006.append('''%s/%s'''%(t2006[a].month,t2006[a].day))
        fmmm2006.append(l2006[a])
tttt2005=[]
fmmm2005=[]
for a in np.arange(len(t2005)):
    if t2005[a]>dt.datetime(2005,4,25,23,0,0):
        tttt2005.append('''%s/%s'''%(t2005[a].month,t2005[a].day))
        fmmm2005.append(l2005[a])
tttt2004=[]
fmmm2004=[]
for a in np.arange(len(t2004)):
    if t2004[a]>dt.datetime(2004,4,25,0,0,0):
        tttt2004.append('''%s/%s'''%(t2004[a].month,t2004[a].day))
        fmmm2004.append(l2004[a])

r2004=[73,34,57,82,96,98,100,98,99,83,73,93,94,67,49,58,87,93,83,76,84,75,52,61,45]
r2004s=[27,65,42,18,2,2,0,2,1,17,27,7,0,0,1,0,1,7,17,24,16,24,15,14,4]
r2004n=[0,1,1,0,2,0,0,0,0,0,0,0,6,33,50,42,12,0,0,0,0,1,33,25,51]

r2005=[90,77,82,97,97,95,69,47,42,42,44,77,73,56,85,70,83,90,65,68,77,61,60,66,68]
r2005s=[10,2,4,3,2,5,31,53,55,57,56,21,0,26,15,30,2,6,27,23,8,0,1,4,9]
r2005n=[0,21,14,0,1,0,0,0,3,1,0,2,27,18,0,0,15,4,8,9,15,39,39,30,23]

r2006=[78,65,64,30,53,73,34,68,78,99,73,72,69,81,97,51,43,47,46,64,71,86,75,66,85]
r2006s=[0,32,36,70,47,25,66,32,22,0,0,14,31,19,3,2,0,0,0,0,1,2,5,5,8]
r2006n=[22,3,0,0,0,2,0,0,0,1,27,14,0,0,0,47,57,53,54,36,28,12,20,29,7]

r2007=[89,73,60,43,98,90,79,90,85,73,81,97,41,49,51,61,39,55,72,99,74,62,77,67,65]
r2007s=[7,17,39,57,2,1,0,10,15,27,3,0,59,51,49,17,2,0,0,0,4,2,1,2,28]
r2007n=[4,10,1,0,0,9,21,0,0,0,16,3,0,0,0,22,59,45,28,1,22,36,22,31,7]

r2008=[96,89,87,48,63,58,9,11,29,84,76,100,90,73,73,90,99,98,71,38,58,74,83,41,22]
r2008s=[0,11,13,52,37,42,89,89,71,11,0,0,9,27,25,10,1,2,1,4,5,8,17,59,78]
r2008n=[4,0,0,0,0,0,2,0,0,5,24,0,1,0,2,0,0,0,28,58,37,18,0,0,0]

r2009=[89,67,68,100,90,63,45,57,25,84,75,73,73,86,45,47,62,63,81,75,75,93,78,48,43]
r2009s=[6,2,5,0,1,0,1,9,6,1,0,10,0,9,55,52,38,37,19,9,2,0,0,1,0]
r2009n=[5,31,27,0,9,37,54,34,69,15,25,17,27,5,0,1,0,0,0,16,23,7,22,51,57]

r2010=[68,67,66,36,62,81,54,89,75,83,79,57,78,94,84,81,83,55,54,82,81,76,67,65,70]
r2010s=[32,33,32,64,38,14,1,0,9,17,7,21,8,0,4,7,0,5,4,0,0,1,3,2,5]
r2010n=[0,0,2,0,0,5,45,11,16,0,14,22,14,6,12,12,17,40,42,18,19,23,30,33,25]

plt.figure(figsize=(10,5))
#plt.plot(t2008,l2008)
plt.plot(tt,fm)

fig,axes=plt.subplots(7,1,figsize=(10,20))
plt.subplots_adjust(wspace=0.4,hspace=0.4)
'''
for label in axes[0].get_xticklabels():
    label.set_visible(False)
'''
##########################################################

t00=datetime(2004,3,31,0,0,0)#2004,
tt0=[]
for a in np.arange(25):
    tt0.append(t00+timedelta(days=a*5))
width1=t00=datetime(2004,4,1,0,0,0)-datetime(2004,3,31,0,0,0)
axes[0].bar(tt0,r2004,width=1.2,label='retention')
axes[0].bar(np.array(tt0)+width1,r2004s,width=1.2,label='south')
axes[0].bar(np.array(tt0)+2*width1,r2004n,width=1.2,label='north')
axes[0].set_title('2004')

ax02=axes[0].twinx()
ax02.plot(t2004,l2004,'r-')
axes[0].legend()
ax02.set_ylabel('discharge, cubic feet per second')

#########################################################
t05=datetime(2005,3,31,0,0,0)#2004,
tt5=[]
for a in np.arange(25):
    tt5.append(t05+timedelta(days=a*5))
width1=t00=datetime(2005,4,1,0,0,0)-datetime(2005,3,31,0,0,0)
axes[1].bar(tt5,r2005,width=1.2,label='retention')
axes[1].bar(np.array(tt5)+width1,r2005s,width=1.2,label='south')
axes[1].bar(np.array(tt5)+2*width1,r2005n,width=1.2,label='north')
axes[1].set_title('2005')

ax05=axes[1].twinx()
ax05.plot(t2005,l2005,'r-')
#ax05.set_ylabel('discharge, cubic feet per second')

#########################################################333
t06=datetime(2006,3,31,0,0,0)#2004,
tt6=[]
for a in np.arange(25):
    tt6.append(t06+timedelta(days=a*5))
width1=t00=datetime(2005,4,1,0,0,0)-datetime(2005,3,31,0,0,0)
axes[2].bar(tt6,r2006,width=1.2,label='retention')
axes[2].bar(np.array(tt6)+width1,r2006s,width=1.2,label='south')
axes[2].bar(np.array(tt6)+2*width1,r2006n,width=1.2,label='north')
axes[2].set_title('2006')

ax06=axes[2].twinx()
ax06.plot(t2006,l2006,'r-')
#ax06.set_ylabel('discharge, cubic feet per second')
################################################3#axes[1].plot(t2005,l2005)
t07=datetime(2007,3,31,0,0,0)#2004,
tt7=[]
for a in np.arange(25):
    tt7.append(t07+timedelta(days=a*5))
width1=t00=datetime(2005,4,1,0,0,0)-datetime(2005,3,31,0,0,0)
axes[3].bar(tt7,r2007,width=1.2,label='retention')
axes[3].bar(np.array(tt7)+width1,r2007s,width=1.2,label='south')
axes[3].bar(np.array(tt7)+2*width1,r2007n,width=1.2,label='north')
axes[3].set_title('2007')

ax07=axes[3].twinx()
ax07.plot(t2007,l2007,'r-')
#ax07.set_ylabel('discharge, cubic feet per second')

#########################################################
t08=datetime(2008,3,31,0,0,0)#2004,
tt8=[]
for a in np.arange(25):
    tt8.append(t08+timedelta(days=a*5))
width1=t00=datetime(2005,4,1,0,0,0)-datetime(2005,3,31,0,0,0)
axes[4].bar(tt8,r2008,width=1.2,label='retention')
axes[4].bar(np.array(tt8)+width1,r2008s,width=1.2,label='south')
axes[4].bar(np.array(tt8)+2*width1,r2008n,width=1.2,label='north')
axes[4].set_title('2008')

ax08=axes[4].twinx()
ax08.plot(t2008,l2008,'r-')
#ax08.set_ylabel('discharge, cubic feet per second')

#######################################################
t09=datetime(2009,3,31,0,0,0)#2004,
tt9=[]
for a in np.arange(25):
    tt9.append(t09+timedelta(days=a*5))
width1=t00=datetime(2005,4,1,0,0,0)-datetime(2005,3,31,0,0,0)
axes[5].bar(tt9,r2009,width=1.2,label='retention')
axes[5].bar(np.array(tt9)+width1,r2009s,width=1.2,label='south')
axes[5].bar(np.array(tt9)+2*width1,r2009n,width=1.2,label='north')
axes[5].set_title('2009')

ax09=axes[5].twinx()
ax09.plot(t2009,l2009,'r-')
#ax09.set_ylabel('discharge, cubic feet per second')

#######################################################
t10=datetime(2010,3,31,0,0,0)#2004,
tt10=[]
for a in np.arange(25):
    tt10.append(t10+timedelta(days=a*5))
width1=t00=datetime(2005,4,1,0,0,0)-datetime(2005,3,31,0,0,0)
axes[6].bar(tt10,r2010,width=1.2,label='retention')
axes[6].bar(np.array(tt10)+width1,r2010s,width=1.2,label='south')
axes[6].bar(np.array(tt10)+2*width1,r2010n,width=1.2,label='north')
axes[6].set_title('2010')

ax10=axes[6].twinx()
ax10.plot(t2010,l2010,'r-')
#ax10.set_ylabel('discharge, cubic feet per second')

#######################################################
plt.legend()
plt.savefig('rivernewxin',dpi=400)

rp=(np.array(r2004)+np.array(r2005)+np.array(r2006)+np.array(r2007)+np.array(r2008)+np.array(r2009)+np.array(r2010))/7.0
sp=(np.array(r2004s)+np.array(r2005s)+np.array(r2006s)+np.array(r2007s)+np.array(r2008s)+np.array(r2009s)+np.array(r2010s))/7.0
np=(np.array(r2004n)+np.array(r2005n)+np.array(r2006n)+np.array(r2007n)+np.array(r2008n)+np.array(r2009n)+np.array(r2010n))/7.0
#######################################################
fig,axes=plt.subplots(2,1,figsize=(10,7))
plt.subplots_adjust(wspace=0.4,hspace=0.4)
t08=datetime(2008,3,31,0,0,0)#2004,

width1=datetime(2005,4,1,0,0,0)-datetime(2005,3,31,0,0,0)
axes[0].bar(tt8,r2008,width=1.2,label='retention')
axes[0].bar(np.array(tt8)+width1,r2008s,width=1.2,label='south')
axes[0].bar(np.array(tt8)+2*width1,r2008n,width=1.2,label='north')
axes[0].set_title('2008')

ax081=axes[0].twinx()
ax081.plot(t2008,l2008,'r-')