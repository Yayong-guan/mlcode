# -*- coding: utf-8 -*-
"""
Algorithms: LDA
Language : Python
Core idea: getthe between-class scatter matrix and the  minimum within-class scatter matrix
Created by guanyayong on 1-3-14.  
"""

import numpy as np                                         #import the necessary modules
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

#funtcion: get the data from the file 
def loadData():
    print "Please input the path of the datafile:\n"
    filepath = input()                                    #get the path of the data
    dataSet = np.fromfile(filepath,sep = ' ')             # read the file data ,get the data Matrix
    file = open(filepath)                                 #get the rows and columns
    
    m = len(file.readlines())       
    n = len(dataSet)
    dataSet.shape = m,n/m                                #set the row and column of the data Matrix
    print "The dataSet is %d*%d:\n" %(m,n/m),dataSet
    return dataSet                                       #return the data Matrix


#function: get the 
def loadStrData():
    print "Please input the path of the datafile:\n"
    filepath = input()
    dataSet = np.loadtxt(filepath,dtype = np.str)       #get the dataset fron the file
    dataSet = dataSet[1:,1:].astype(np.float)           #transform the type of data
    print "The dataSets is :\n",dataSet 
    return dataSet                                      #return the Matrix data


#function: plot the dataset in the 3-D by using the matplotlib
def showSample(dataX,dataY,redata):     
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax = plt.subplot(111,projection='3d')
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    redata = np.array(redata)
    
    Class,indices = np.unique(dataY,return_index=True)    #get the numbers and index of the class label 
    N = len(Class)                                        #get the numbers of the class
    color = ['ro','bo','go']
    for x in range(N):                                    #plot the original data and the projectional data
        if x < N-1:
            data = dataX[indices[x]:indices[x+1]]  
            p1, = ax.plot(data[:,0],data[:,1],data[:,2],color[x])
            rdata = redata[indices[x]:indices[x+1]]
            p2, = ax.plot(rdata[:,0],rdata[:,1],rdata[:,2],color[x])
        else :
            data = dataX[indices[x]:]
            ax.plot(data[:,0],data[:,1],data[:,2],color[x])
            rdata = redata[indices[x]:]
            ax.plot(rdata[:,0],rdata[:,1],rdata[:,2],color[x])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("LDA Sample Distribution")
    plt.legend([p1,p2], ['Original', 'Projection' ])
    plt.show()


#lda function: reduce the dimensions of the data with the class label
def ldaFunction():
    dataSet = loadData()                                 #get the data
    dataX = dataSet[:,:-1]                               #get the feature data
    dataY = dataSet[:,-1]                                #get the class data
    
    idy = dataY.argsort()                                #resort the class data and the feature data
    dataY = dataY[idy]            
    dataX = dataX[idy,:]
    dataX = np.matrix(dataX)   
    
    avgX = np.mean(dataX,0)                              #get all the average of  each of the feature
    
    """dataSetMean = dataX - avgX                       
    #Variance normalization  var =1
    std = np.std(dataX,0)
    dataX = dataSetMean/std"""
    
    Class,indices = np.unique(dataY,return_index=True)   #get the list of the Class and the index
    N = len(Class)                                       #get the number of the Class
        
    row = dataX.shape[0]                                 #get the dimension of the feature
    col = dataX.shape[1]         
    
    Sb = np.matrix(np.zeros((col,col),dtype=float))     #create the col*col Sb Matrix
    Sw = np.matrix(np.zeros((col,col),dtype=float))     #create the col*col Sw Matrix
    
    C = [];                                             #create a list to contain the each class of data Matrix
    for x in range(N):
        if x < N-1:
            C.append(dataX[indices[x]:indices[x+1]])    #get all the each class of data Matrix
        else :
            C.append(dataX[indices[x]:])
         
        avgC = np.average(C[x],0)                       #get the means of the class data
        X = avgC - avgX              
        
        Sb = Sb + X.T*X*len(C[x])                       #get the between class variance
        Sw = Sw + np.cov(C[x],rowvar=0)*row             #get the within class variance
    if np.linalg.det(Sw) != 0:
        
        EigenVals,EigenVects = np.linalg.eig(Sw.I*Sb)   #get the eignevalues and the eigenvectors
        
        idx = EigenVals.argsort()[::-1]                 #sorting the eigenvalues and the eigenvectors
        EigenVals = EigenVals[idx]
        EigenVects = EigenVects[:,idx]
        
        #get the kth front eigenvectors 
        print("please innput the expected value of the dimensionality reduction:\n ")
        k = int(input())
        if k >= N:
            print("the value of the dimensionality reduction have to less than %d "%N)
        else:
            W = EigenVects[:,0:k]                      #get the k-th eigenvectors
               
            lowDataSet = dataX*W                          #reduce the original data , k*m dimesions
            reMatData = lowDataSet*W.T                 #return to the original space
            
            showSample(dataX,dataY,reMatData)       #call the showSample function
            print "the worked data is %d*%d:\n"%(row,k),lowDataSet
    else:
        print "the inverse of Sw is not exist!\n"

ldaFunction()
            
        
        
    
    
        
                
                