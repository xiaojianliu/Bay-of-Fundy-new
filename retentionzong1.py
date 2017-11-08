# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 01:17:28 2017

@author: xiaojian
"""
from mpl_toolkits.basemap import Basemap  
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
import numpy as np
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
from matplotlib.dates import date2num,num2date
r20041=np.load('r20041.npy')
r20051=np.load('r20051.npy')
r20061=np.load('r20061.npy')
r20071=np.load('r20071.npy')
r20081=np.load('r20081.npy')
r20091=np.load('r20091.npy')
r20101=np.load('r20101.npy')

r20042=np.load('r20042.npy')
r20052=np.load('r20052.npy')
r20062=np.load('r20062.npy')
r20072=np.load('r20072.npy')
r20082=np.load('r20082.npy')
r20092=np.load('r20092.npy')
r20102=np.load('r20102.npy')

n20041=np.load('n20041.npy')
n20051=np.load('n20051.npy')
n20061=np.load('n20061.npy')
n20071=np.load('n20071.npy')
n20081=np.load('n20081.npy')
n20091=np.load('n20091.npy')
n20101=np.load('n20101.npy')

n20042=np.load('n20042.npy')
n20052=np.load('n20052.npy')
n20062=np.load('n20062.npy')
n20072=np.load('n20072.npy')
n20082=np.load('n20082.npy')
n20092=np.load('n20092.npy')
n20102=np.load('n20102.npy')

s20041=np.load('s20041.npy')
s20051=np.load('s20051.npy')
s20061=np.load('s20061.npy')
s20071=np.load('s20071.npy')
s20081=np.load('s20081.npy')
s20091=np.load('s20091.npy')
s20101=np.load('s20101.npy')

s20042=np.load('s20042.npy')
s20052=np.load('s20052.npy')
s20062=np.load('s20062.npy')
s20072=np.load('s20072.npy')
s20082=np.load('s20082.npy')
s20092=np.load('s20092.npy')
s20102=np.load('s20102.npy')

fig,axes=plt.subplots(3,2,figsize=(14,15))
axes[0,0].plot(r20041,label='2004')
axes[0,0].plot(r20051,label='2005')
axes[0,0].plot(r20061,label='2006')
axes[0,0].plot(r20071,label='2007')
axes[0,0].plot(r20081,label='2007')
axes[0,0].plot(r20091,label='2008')
axes[0,0].plot(r20101,label='2009')
r1m=[]
for a in np.arange(len(r20041)):
    r1m.append((r20041[a]+r20051[a]+r20061[a]+r20071[a]+r20081[a]+r20091[a]+r20101[a])/7.0)
axes[0,0].plot(r1m,color='black',linewidth=3,label='mean')
axes[0,0].legend()
axes[0,0].set_title('particle retention(3/31 to 6/14) at 15 meters')
axes[0,1].set_title('particle retention(6/14 to 7/14) at 15 meters')
axes[1,0].set_title('the number of partitle to south(3/31 to 6/14) at 15 meters')
axes[1,1].set_title('the number of partitle to south(6/14 to 7/14) at 15 meters')

axes[2,0].set_title('the number of partitle to north(3/31 to 6/14) at 15 meters')
axes[2,1].set_title('the number of partitle to north(6/14 to 7/14) at 15 meters')

axes[0,1].plot(r20042,label='2004')
axes[0,1].plot(r20052,label='2005')
axes[0,1].plot(r20062,label='2006')
axes[0,1].plot(r20072,label='2007')
axes[0,1].plot(r20082,label='2007')
axes[0,1].plot(r20092,label='2008')
axes[0,1].plot(r20102,label='2009')
axes[1,0].set_ylim([0,35])
axes[1,1].set_ylim([0,35])
axes[2,0].set_ylim([0,35])
axes[2,1].set_ylim([0,35])
axes[0,0].set_xlabel('a    (hour)')
axes[0,1].set_xlabel('b    (hour)')

axes[1,0].set_xlabel('c    (hour)')

axes[2,0].set_xlabel('e    (hour)')

axes[1,1].set_xlabel('d    (hour)')

axes[2,1].set_xlabel('f    (hour)')

r2m=[]
for a in np.arange(len(r20042)):
    r2m.append((r20042[a]+r20052[a]+r20062[a]+r20072[a]+r20082[a]+r20092[a]+r20102[a])/7.0)
axes[0,1].plot(r2m,color='black',linewidth=3,label='mean')
axes[0,1].legend()

axes[1,0].plot(s20041,label='2004')
axes[1,0].plot(s20051,label='2005')
axes[1,0].plot(s20061,label='2006')
axes[1,0].plot(s20071,label='2007')
axes[1,0].plot(s20081,label='2007')
axes[1,0].plot(s20091,label='2008')
axes[1,0].plot(s20101,label='2009')
s1m=[]
for a in np.arange(len(r20041)):
    s1m.append((s20041[a]+s20051[a]+s20061[a]+s20071[a]+s20081[a]+s20091[a]+s20101[a])/7.0)
axes[1,0].plot(s1m,color='black',linewidth=3,label='mean')
axes[1,0].legend()

axes[1,1].plot(s20042,label='2004')
axes[1,1].plot(s20052,label='2005')
axes[1,1].plot(s20062,label='2006')
axes[1,1].plot(s20072,label='2007')
axes[1,1].plot(s20082,label='2007')
axes[1,1].plot(s20092,label='2008')
axes[1,1].plot(s20102,label='2009')
s2m=[]
for a in np.arange(len(r20042)):
    s2m.append((s20042[a]+s20052[a]+s20062[a]+s20072[a]+s20082[a]+s20092[a]+s20102[a])/7.0)
axes[1,1].plot(s2m,color='black',linewidth=3,label='mean')
axes[1,1].legend()

axes[2,0].plot(n20041,label='2004')
axes[2,0].plot(n20051,label='2005')
axes[2,0].plot(n20061,label='2006')
axes[2,0].plot(n20071,label='2007')
axes[2,0].plot(n20081,label='2007')
axes[2,0].plot(n20091,label='2008')
axes[2,0].plot(n20101,label='2009')
n1m=[]
for a in np.arange(len(r20041)):
    n1m.append((n20041[a]+n20051[a]+n20061[a]+n20071[a]+n20081[a]+n20091[a]+n20101[a])/7.0)
axes[2,0].plot(n1m,color='black',linewidth=3,label='mean')
axes[2,0].legend()

axes[2,1].plot(n20042,label='2004')
axes[2,1].plot(n20052,label='2005')
axes[2,1].plot(n20062,label='2006')
axes[2,1].plot(n20072,label='2007')
axes[2,1].plot(n20082,label='2007')
axes[2,1].plot(n20092,label='2008')
axes[2,1].plot(n20102,label='2009')
n2m=[]
for a in np.arange(len(r20042)):
    n2m.append((n20042[a]+n20052[a]+n20062[a]+n20072[a]+n20082[a]+n20092[a]+n20102[a])/7.0)
axes[2,1].plot(n2m,color='black',linewidth=3,label='mean')
axes[2,1].legend()
plt.savefig('rsn',dpi=300)
    
    