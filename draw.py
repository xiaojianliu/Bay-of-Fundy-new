# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:18:02 2016

@author: xiaojian
"""
import numpy as np
import matplotlib.pyplot as plt

drifter_list='drifter_vs_roms use hourly data15.csv'
#mend_list='bar1.csv'
day=15
drifters = np.genfromtxt(drifter_list,dtype=None,names=['ids','lon','lat','s','d'],delimiter=',',skip_header=1)
#mend = np.genfromtxt(mend_list,dtype=None,names=['ids','lon','lat','s','d'],delimiter=',',skip_header=1)   
num=[]
x=[]
for b in np.arange(6,round(max(drifters['s']))+6,6):
    #print b,'##################################'
    aa=0
    x.append(b-6)
    x.append(b)
    for a in np.arange(len(drifters['s'])):
        if drifters['s'][a]>=b-6 and drifters['s'][a]<b:
            aa=aa+1
            #print drifters['s'][a]
    num.append(aa)
    num.append(aa)
num.append(0)
x.append(b)
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].set_title('Separation distance density distribution(%s days)'%day) 
axes[0].plot(x,num,'b-',label='roms',linewidth=1)
axes[0].set_xlabel('km')
axes[0].set_ylabel('number')

num1=[]
x1=[]
for b in np.arange(0.02,round(max(drifters['d'])*100)/float(100),0.02):
    #print b
    aa=0
    x1.append(b-0.02)
    x1.append(b)
    for a in np.arange(len(drifters['d'])):
        if drifters['d'][a]>=b-0.02 and drifters['d'][a]<b:
            aa=aa+1
            
    num1.append(aa)
    num1.append(aa)
num1.append(0)
x1.append(b)
axes[1].set_title('distance ratio density distribution') 
axes[1].set_xlabel('km/km')
axes[1].set_ylabel('number')

#plt.title('drifter_meandis_and_mend_meandis') 
axes[1].plot(x1,num1,'b-',label='roms',linewidth=1)
print '''roms'''
print 'mean',np.mean(drifters['s'])/day
print 'max',np.max(drifters['s'])
print 'min',np.min(drifters['s'])
print 'std',np.std(drifters['s'])
#print np.mean(drifters['s'][drifters['d']<1])
print 'mean',np.mean(drifters['d'])
print 'max',np.max(drifters['d'])
print 'min',np.min(drifters['d'])
print 'std',np.std(drifters['d'])
print '###############################################'
drifter_list='drifter_vs_model use hourly data15.csv'
#mend_list='bar1.csv'
drifters = np.genfromtxt(drifter_list,dtype=None,names=['ids','lon','lat','s','d'],delimiter=',',skip_header=1)
#mend = np.genfromtxt(mend_list,dtype=None,names=['ids','lon','lat','s','d'],delimiter=',',skip_header=1)   
num=[]
x=[]
for b in np.arange(6,round(max(drifters['s']))+6,6):
    #print b,'##################################'
    aa=0
    x.append(b-6)
    x.append(b)
    for a in np.arange(len(drifters['s'])):
        if drifters['s'][a]>=b-6 and drifters['s'][a]<b:
            aa=aa+1
            #print drifters['s'][a]
    num.append(aa)
    num.append(aa)
num.append(0)
x.append(b)
axes[0].plot(x,num,'r-',label='fvcom',linewidth=1)
num1=[]
x1=[]
for b in np.arange(0.02,round(max(drifters['d'])*100)/float(100),0.02):
    #print b
    aa=0
    x1.append(b-0.02)
    x1.append(b)
    for a in np.arange(len(drifters['d'])):
        if drifters['d'][a]>=b-0.02 and drifters['d'][a]<b:
            aa=aa+1
            
    num1.append(aa)
    num1.append(aa)
num1.append(0)
x1.append(b)
axes[0].legend()
axes[1].plot(x1,num1,'r-',label='fvcom',linewidth=1)
axes[1].legend(loc='best')
plt.savefig('roms_vs_fvcomx%shah'%day,dpi=200)
plt.show()
