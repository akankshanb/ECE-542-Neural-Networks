import numpy as np
from scipy.stats import kurtosis, skew


def generate_features(X):
    X  = np.array(X)
    f1 = np.mean(X, axis=0)
    c = np.cov(np.transpose(X))
    f2 = np.concatenate((c[0, 0:3],c[1, 1:3],c[2, 2:3]), axis=0)
    f3 = skew(X)
    f4 = kurtosis(X,axis=0, fisher=False)
    f5 = np.zeros(3)
    f6 = np.zeros(3)
    for i in range(3):
        g = abs(np.fft.fft(X[:, i]))
        g = g[0:int(np.ceil(len(g)/2))]
        g[0] = 0
        w = 50*np.arange(len(g))/(2*len(g))
        v = max(g)
        idx = int(np.where(g==v)[0])
        f5[i] = v
        f6[i] = w[idx]
        
    f=np.concatenate((f5, f6), axis=0)
    f=np.concatenate((f3, f4, f), axis=0)
    f=np.concatenate((f1, f2, f), axis=0)
    return f

# xftrain = generate_features([[1,2,3],[4,5,6],[7,8,9],[1,5,8]])
