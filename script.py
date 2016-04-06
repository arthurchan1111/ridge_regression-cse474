import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi, exp
from scipy import linalg
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    k = np.unique(y).shape[0]
    means = np.zeros((k,X.shape[1]))
    counts = np.zeros((k,1))

    for i in range(0, y.shape[0]):
        counts[int(y[i]-1)]+= 1
        means[int(y[i]-1)]= means[int(y[i]-1)]+X[i]

    poolmeans= np.transpose(np.mean(X, axis=0, keepdims=True))
    means= np.transpose(means/counts)
    covelem= means-poolmeans

    covmat=np.dot(covelem,np.transpose(covelem))/y.shape[0]
    det= np.linalg.det(covmat)

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD

    k = np.unique(y).shape[0]
    means = np.zeros((k,X.shape[1]))
    counts = np.zeros((k,1))
    covmats = []

    for i in range(0, y.shape[0]):
        counts[int(y[i]-1)]+= 1
        means[int(y[i]-1)]= means[int(y[i]-1)]+X[i]

    div= []
    for i in range(0,k):
        a= np.zeros((counts[i],X.shape[1]))
        c=0
        for j in range(0, y.shape[0]):
            if y[j]-1 == i:
                a[c]= X[j]
                c+= 1
        div.append(a)

    for i in range(0,k):
        covmats.append(np.cov(np.transpose(div[i])))


    means= np.transpose(means/counts)

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    all= np.zeros((Xtest.shape[0],means.shape[1]))
    
    for i in range(0, means.shape[1]):
        for j in range(0, Xtest.shape[0]):
            xu= Xtest[j]-np.transpose(means)[i]
            po= np.dot(np.dot(np.transpose(xu),np.linalg.inv(covmat)),xu)/-2
            fin= exp(po)/((2*pi)**(means.shape[1]/2)*sqrt(np.linalg.det(covmat)))
            all[j][i]= fin

    #print all
    ypred= np.zeros((Xtest.shape[0],1))
    c=0
    for x in all:
        maxindex=0
        max=0
        count=1
        for i in x:
            if i >= max:
                max=i
                maxindex=count
            count+= 1
        ypred[c]= maxindex
        c+= 1

    correct=0
    for p in range(0, Xtest.shape[0]):
        print(ypred[p], ytest[p])
        if ypred[p] == ytest[p]:
            correct+= 1
    
    acc= (correct*100.0/ytest.shape[0])
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    all= np.zeros((Xtest.shape[0],means.shape[1]))

    for i in range(0, means.shape[1]):
        for j in range(0, Xtest.shape[0]):
            xu= Xtest[j]-np.transpose(means)[i]
            po= np.dot(np.dot(np.transpose(xu),scipy.linalg.inv(covmats[i])),xu)/-2
            fin= exp(po)/((2*pi)**(means.shape[1]/2)*sqrt(np.linalg.det(covmats[i])))
            all[j][i]= fin

    #print all
    ypred= np.zeros((Xtest.shape[0],1))
    c=0
    for x in all:
        maxindex=0
        max=0
        count=1
        for i in x:
            if i >= max:
                max=i
                maxindex=count
            count+= 1
        ypred[c]= maxindex
        c+= 1

    correct=0
    for p in range(0, Xtest.shape[0]):
        if ypred[p] == ytest[p]:
            correct+= 1
    
    acc= (correct*100.0/ytest.shape[0])

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD                                                   

    e1= np.linalg.inv(np.dot(np.transpose(X),X))
    e2= np.dot(e1,np.transpose(X))
    w= np.dot(e2,y)
    
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD      
    id = np.identity(X.shape[1]);
    e1= np.linalg.inv(np.add(np.dot(np.transpose(X),X), lambd*id))
    e2= np.dot(e1,np.transpose(X))
    w= np.dot(e2,y)                                             
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    e1= ytest-np.dot(Xtest,w)
    e2= np.dot(np.transpose(e1),e1)/Xtest.shape[0]
    rmse= sqrt(e2)

    # IMPLEMENT THIS METHOD
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    w = np.array(w,ndmin=2)
    w = np.transpose(w)
    
    error = 0.5 * (np.dot(np.transpose(y-np.dot(X,w)),y-np.dot(X,w)) + lambd*(np.dot(np.transpose(w),w)))
    #error = 0.5 * (np.sum(np.power(y-np.dot(X,w),2)) + lambd*(np.dot(np.transpose(w),w)))
    #error_grad= np.dot(np.dot(np.transpose(X),X),w) - np.dot(np.transpose(X),y)  +  lambd * np.transpose(w)
    error_grad = -2 * np.dot( np.transpose(X),(y-np.dot(X,w)))  + 2*lambd * w
    #error_grad=np.transpose(np.array(error_grad[:,0],ndmin=2))
    error_grad=error_grad[:,0]
    error = error[0][0]
                                              
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    print("Xd in shape")
    #print(x)
    print(p)
    Xd = np.zeros((x.shape[0],p+1))
    print("Xd shape")
    print(Xd.shape)
    for y in range(0,p+1):
        for j in range(0,x.shape[0]):
           
            Xd[j][y]=x[j]**y
    print("Xd out")
    print(Xd)
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 500}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='BFGS', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)
plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()