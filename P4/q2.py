from sklearn import linear_model
import lassoData as ld
import numpy as np
import math
import matplotlib.pyplot as plt

def compute_sinX(X,M):
	r = np.zeros((X.size,M))
	for i in range(X.size):
		r[i,0] = X[i]
		for j in range(1,M):
			r[i,j] = math.sin(0.4*j*math.pi*X[i])
	return r	

x,y = ld.lassoTrainData()
xv,yv = ld.lassoValData()
xt,yt = ld.lassoTestData()
m = 13
l = 0.001	
X = compute_sinX(x,m)
clf = linear_model.Lasso(alpha = l)
clf.fit(X,y)
clf1 = linear_model.Lasso(alpha = 0.1)
clf1.fit(X,y)
clf2 = linear_model.Lasso(alpha = 1)
clf2.fit(X,y)

plt.plot(x,y,'bo',label='training')
plt.plot(xv,yv,'ro',label='validation')
plt.plot(xt,yt,'go',label='test')
xnew = np.linspace(min(x.min(),xt.min(),xv.min()),max(x.max(),xt.max(),xv.max()),100)
x1 = compute_sinX(xnew,m)
plt.plot(xnew,clf.predict(x1),'b',label='0.001')
plt.plot(xnew,clf1.predict(x1),'g',label='0.1')
plt.plot(xnew,clf2.predict(x1),'r',label='1')
plt.legend()
plt.title("Plot of estimated function with LASSO")
plt.ylabel("y")
plt.xlabel("x")
plt.axhline(0, color='black', lw=2)
plt.show()

