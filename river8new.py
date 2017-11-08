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
north2004=np.load('2004north.npy')
north2005=np.load('2005north.npy')
north2006=np.load('2006north.npy')
north2007=np.load('2007north.npy')
north2008=np.load('2008north.npy')
north2009=np.load('2009north.npy')
north2010=np.load('2010north.npy')

south2004=np.load('2004south.npy')
south2005=np.load('2005south.npy')
south2006=np.load('2006south.npy')
south2007=np.load('2007south.npy')
south2008=np.load('2008south.npy')
south2009=np.load('2009south.npy')
south2010=np.load('2010south.npy')

north20041=np.load('2004north1.npy')
north20051=np.load('2005north1.npy')
north20061=np.load('2006north1.npy')
north20071=np.load('2007north1.npy')
north20081=np.load('2008north1.npy')
north20091=np.load('2009north1.npy')
north20101=np.load('2010north1.npy')

south20041=np.load('2004south1.npy')
south20051=np.load('2005south1.npy')
south20061=np.load('2006south1.npy')
south20071=np.load('2007south1.npy')
south20081=np.load('2008south1.npy')
south20091=np.load('2009south1.npy')
south20101=np.load('2010south1.npy')

north20082=np.load('2008north2.npy')
north20092=np.load('2009north2.npy')

south20082=np.load('2008south2.npy')
south20092=np.load('2009south2.npy')
s2008=[]
s2009=[]
ns2009=np.load('2009sn.npy')
ns2008=np.load('2008sn.npy')

#s2008.append(south2008[6])
s2008.append(south20081[6])
s2008.append(south20082[6])

#s2009.append(north2009[6])
s2009.append(north20091[6])
s2009.append(north20092[6])
s=[]
s.append(s2008)
s.append(s2009)
tttt20051=[]
fmmm20051=[]
for a in np.arange(len(t2005)):
    if t2005[a]>dt.datetime(2005,5,5,0,0,0) and t2005[a]<dt.datetime(2005,5,20,0,0,0):
        tttt20051.append('''%s/%s'''%(t2005[a].month,t2005[a].day))
        fmmm20051.append(l2005[a])

fmmm1=[]
tttt1=[]
for a in np.arange(len(t2008)):
    if t2008[a]>dt.datetime(2008,5,5,0,0,0) and t2008[a]<dt.datetime(2008,5,20,0,0,0):
        tttt1.append('''%s/%s'''%(t2008[a].month,t2008[a].day))
        fmmm1.append(l2008[a])
tttt20091=[]
fmmm20091=[]
for a in np.arange(len(t2009)):
    if t2009[a]>dt.datetime(2009,5,15,0,0,0) and t2009[a]<dt.datetime(2009,5,30,0,0,0):
        tttt20091.append('''%s/%s'''%(t2009[a].month,t2009[a].day))
        fmmm20091.append(l2009[a])

fig,axes=plt.subplots(2,2,figsize=(15,10))
plt.subplots_adjust(wspace=0.4,hspace=0.4)
df=pd.DataFrame(np.array([[south2005[7],south2008[7],south2009[9]],[north2005[7],north2008[7],north2009[9]]]).T,index=['2005/5/5','2008/5/5','2009/5/15'],columns=pd.Index(['windstress to south','windstess to north']))
df.plot(kind='bar',ax=axes[0,0])
axes[0,0].set_ylim([0,300])
axes[0,0].set_ylabel('hours')
axes[0,0].set_xlabel('a')
axes[0,1].set_xlabel('b')
axes[1,0].set_xlabel('c')
axes[1,1].set_xlabel('d    (day)')

df=pd.DataFrame(np.array([[-south20051[7],-south20081[7],-south20091[9]],[north20051[7],north20081[7],north20091[9]]]).T,index=['2005/5/5','2008/5/5','2009/5/15'],columns=pd.Index(['the sum of windstress(to south)','the sum of windstess(to north)']))
df.plot(kind='bar',ax=axes[0,1])
axes[0,1].set_ylim([0,18])
axes[0,1].set_ylabel('pa')

df=pd.DataFrame(np.array([[47,11,84],[53,89,1],[0,0,15]]).T,index=['2005/5/5','2008/5/5','2009/5/15'],columns=pd.Index(['retention','the mumber of particles(to south)','the mumber of particles(to north)']))
df.plot(kind='bar',ax=axes[1,0])
axes[1,0].set_ylim([0,100])
axes[1,0].set_title('the number of particles in three regions')
axes[1,1].set_ylabel('discharge, cubic feet per second')
axes[1,1].set_title('discharge (St. Join River)')
xxin=np.linspace(0,1439,1439)
xxxin=[]
for a in np.arange(len(xxin)):
    xxxin.append(xxin[a]*15/(float(60)*24))
axes[1,1].plot(xxxin,fmmm20051,label='2005/5/5')
axes[1,1].plot(xxxin,fmmm1,label='2008/5/5')
axes[1,1].plot(xxxin,fmmm20091,label='2009/5/15')
#axes[1,1].set_xlabel('day')
axes[1,1].legend()
plt.savefig('lunwen',dpi=300)


'''
axes[0,0].bar(0.5,south2008[6],width=0.2,label='to south(2008)')
axes[0,0].bar(1,south2005[6],width=0.2,label='to north(2005)')
axes[0,0].legend()


axes[0,1].bar(0.5,-south20081[6],width=0.2,label='to south(2008)')
axes[0,1].bar(1,north20091[6],width=0.2,label='to north(2009)')
axes[0,1].legend()
#axes[1].legend()
axes[1,0].bar(0.5,north20081[6],width=0.2,label='to north(2008)')
axes[1,0].bar(1,-south20091[6],width=0.2,label='to south(2009)')

axes[1,0].legend()
axes[1,1].bar(0.5,ns2008[6],width=0.2,label='2008')
axes[1,1].bar(1,ns2009[6],width=0.2,label='2009')
axes[1,1].legend()

axes[3].bar(0.5,south20082[6],width=0.3,label='to south')
axes[3].bar(1,north20092[9],width=0.3,label='to north')
'''
#axes[1].legend()

'''
df=pd.DataFrame(np.array(s).T,index=['mean','std'],columns=pd.Index(['to south','to north']))
df.plot(kind='bar',ax=axes[1])
'''
print np.argmin(abs(np.array(north2004)-south2008[6]))
print np.argmin(abs(np.array(north2005)-south2008[6]))
print np.argmin(abs(np.array(north2006)-south2008[6]))
print np.argmin(abs(np.array(north2007)-south2008[6]))
print np.argmin(abs(np.array(north2008)-south2008[6]))
print np.argmin(abs(np.array(north2009)-south2008[6]))
print np.argmin(abs(np.array(north2010)-south2008[6]))
print '''#########################################################3'''

print np.argmin(abs(np.array(north20041)+south20081[6]))
print np.argmin(abs(np.array(north20051)+south20081[6]))
print np.argmin(abs(np.array(north20061)+south20081[6]))
print np.argmin(abs(np.array(north20071)+south20081[6]))
print np.argmin(abs(np.array(north20081)+south20081[6]))
print np.argmin(abs(np.array(north20091)+south20081[6]))
print np.argmin(abs(np.array(north20101)+south20081[6]))

print '''#########################################################3'''

print np.argmin(abs(np.array(south20041)+north20081[6]))
print np.argmin(abs(np.array(south20051)+north20081[6]))
print np.argmin(abs(np.array(south20061)+north20081[6]))
print np.argmin(abs(np.array(south20071)+north20081[6]))
print np.argmin(abs(np.array(south20081)+north20081[6]))
print np.argmin(abs(np.array(south20091)+north20081[6]))
print np.argmin(abs(np.array(south20101)+north20081[6]))
v2008=np.load('v2008.npy')
v2009=np.load('v2009.npy')
v2010=np.load('v2010.npy')
v2004=np.load('v2004.npy')
v2005=np.load('v2005.npy')
v2006=np.load('v2006.npy')
v2007=np.load('v2007.npy')
'''
k=np.argmin(abs(v2008[0]-datetime(2008,4,30)))
k1=np.argmin(abs(v2009[0]-datetime(2009,5,15)))
plt.figure()
plt.plot(v2008[1][k:k+360])
for a in np.arange(0,1,1):
    
    plt.plot(-v2009[1][a*120:a*120+360],label='2009_%s_%s'%(v2009[0][a*120].month,v2009[0][a*120].day))
plt.legend()
'''
plt.figure(figsize=(15,5))

plt.plot(v2008[1][1200:1200+15*24],label='2004')

plt.plot(v2010[1][1200:1200+15*24],label='2010')
'''
plt.plot(v2006[1],label='2006')
plt.plot(v2007[1],label='2007')
plt.plot(v2008[1],label='2008')
plt.plot(v2009[1],label='2009')
plt.plot(v2010[1],label='2010')


plt.legend()
'''