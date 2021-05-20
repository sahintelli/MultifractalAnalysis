import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

def convertprice(price):
    return format(float(price), '.10f')

def mfdfa(close, scmax, m):
    #scmax = 250
    #m = 1
    scmin = 16
    scres = 19
    exponents = np.linspace(np.log2(scmin), np.log2(scmax), scres)

    scale = np.exp2(exponents).astype(int)

    q = np.linspace(-5, 5, num=101)
    q = [round(x,1) for x in q]

    close = np.array(close)
    X = np.cumsum(close-close.mean())
    segments = []
    RMS_scale = [[] for _ in range(len(scale))] 
    qRMS = [[[] for _ in range(len(scale))] for _ in range(len(q))]
    Fq = [[[] for _ in range(len(scale))] for _ in range(len(q))]

    for ns in range(0,len(scale)):
        segments.append(int(len(X)/scale[ns]))
        for v in range(0,segments[ns]):
            Index = range(v*scale[ns],(v+1)*scale[ns])
            C = np.polyfit(Index,X[Index],m)
            fit = np.polyval(C,Index)
            RMS_scale[ns].append(np.sqrt(np.square(X[Index]-fit).mean())) 
        for nq in range(0,len(q)):
            qRMS[nq][ns] = np.power(RMS_scale[ns],q[nq])
            if q[nq]==0:
                Fq[nq][ns] = np.exp(0.5*np.log(np.power(RMS_scale[ns],2)).mean())
            else:
                Fq[nq][ns] = np.power((qRMS[nq][ns]).mean(),1/q[nq])


    Hq = []
    qRegLine = [[] for _ in range(len(q))]

    for nq in range(0,len(q)):
        C = np.polyfit(np.log2(scale),np.log2(Fq[nq]),1)
        Hq.append(C[0])
        qRegLine[nq].append(np.polyval(C,np.log2(scale)))

    #[ x*y for x in a for y in b] # [[x*y for x in a] for y in b]
    tq = np.array([val1*val2 for val1,val2 in zip(Hq,q)])-1
    hq = np.diff(tq)/(q[1]-q[0])
    Dq = np.array([val1*val2 for val1,val2 in zip(q[0:len(q)-1],hq)])-tq[0:len(tq)-1]

    I = np.argmax(Dq)-1
    alpha0 = hq[I]
    R = (np.max(hq)-alpha0)/(alpha0-np.min(hq))
    W = np.max(hq) - np.min(hq)

    complexity = [alpha0, R, W]
    complexity = [round(i,4) for i in complexity]
    #plt.plot(np.log2(scale), np.log2(Fq))
    #plt.plot(hq,Dq)
    #plt.show()
    return complexity
