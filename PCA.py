# -*- coding: utf-8 -*-
"""
Algorithms: PCA
Language : Python
Core idea: find the kth eigenvectors according to the k-max eigenvalues
Created by guanyayong on 1-3-14.  
"""

import numpy as np                                    #import the necessary modules
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

#function : load the dataset,include the string words
def loadStrData():   
    print "Please input the path of the datafile:\n"
    filepath = input()
    dataSet = np.loadtxt(filepath,dtype = np.str)     #get the dataset fron the file
    dataSet = dataSet[1:,1:].astype(np.float)         #transform the type of data
    print "The dataSets is :\n",dataSet 
    return dataSet                                    #return the Matrix data

#function: load the data
def loadData():
    print "Please input the path of the datafile:\n"
    filepath = input()                                #get the path of the data
    dataSet = np.fromfile(filepath,sep = ' ')         #get the data Matrix
    
    file = open(filepath)
    m = len(file.readlines())                         #get the row numbers
    n = len(dataSet)
    dataSet.shape = m,n/m                             #set the row and column of the data Matrix
    print "The dataSet is %d*%d:\n" %(m,n/m),dataSet
    return dataSet                                    #return the worked data Matrix


"""function: show the dataset in the 3-D by using the matplotlib,
   the parameter of data stands for the original data,
   the parameter of data stands for the reverted data
   """
def showSample(data,redata):
    fig = plt.figure()                                #create the object of figure
    ax = fig.gca(projection='3d')                     #set the space fo 3-D
    #ax = plt.subplot(111,projection='3d')
    data = np.array(data)
    redata = np.array(redata)
    
    ax.plot(data[:,0],data[:,1],data[:,2],'ro',label='Original')          #plot the original data in the 3-D space
    ax.plot(redata[:,0],redata[:,1],redata[:,2],'bo',label='Projection')  #plot the reverted data in the 3-D space
    ax.set_xlabel("x")                                #set the axis' label 
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("PCA Sample Distribution")              #set the plot' title
    plt.legend()                                      #show the rectangle of label
    plt.show()


#pca funtcion,reduce the dimensions of the data
def pcaFunction():
    dataSet = loadData()                              #call the loadData funtcion ,get the data Matrix
    
    avg = np.average(dataSet,0)                       #get each of the avgerage of the featuer value
    meanDataSet = dataSet - avg                       #meaning the data,make the mean of the data euqals 0
    
    std = np.std(dataSet,0)                           #Variance normalization  var =1
    normalDataSet = meanDataSet/std
    normalDataSet = np.matrix(normalDataSet)
    #CovMat = np.dot(np.transpose(dataMeans),dataMeans)/dataMeans.shape[1]  #get the Scatter Matrix
    CovMat = np.cov(normalDataSet,rowvar=0)           # get the covariance matrix
    EigenVals,EigenVects = np.linalg.eig(CovMat)      #get the characteristic value and the feature vector,
                                                      #EigenVals,EigenVects stand for the feature value and the feature vector,
                                         
    idx = EigenVals.argsort()[::-1]                   #sorting the eigenValuses and the associated eigenVectors
    EigenVals = EigenVals[idx]
    EigenVects = EigenVects[:,idx]                   
    #print(EigenVals)
    #print(EigenVects)
    
    print "please input the expected value of the dimensionality reduction:\n"
    k = int(input())                   
    EigenVects = EigenVects[:,0:k]                   #get the front k eigenVectors
    lowDimData = normalDataSet * EigenVects          #reduce the original data to k dimension
      
    reMatData = np.array((lowDimData*EigenVects.T))*std + avg       #return to the original space
    
    showSample(dataSet,reMatData)                    #call the showSample function

    return lowDimData                                #return the worked data

lowData = pcaFunction()                              #Call the PCA function,deal with the original data
print "The dataSet is %d*%d:\n"%(lowData.shape[0],lowData.shape[1]),lowData       #return the data





    