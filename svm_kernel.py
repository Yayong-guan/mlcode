# -*- coding: utf-8 -*-

"""
svm with kernel
Created by guanyayong  2014-4-16
"""

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class SVM:
    def __init__(self,dataX,labelY,kernelname,parameters,C = 1,tolerance = 0.001):
        #store the C about the slack variable of the soft margain
        self.C = C
        #store the tolerance of KKT conditions
        self.tolerance = tolerance
        #get the dataSet
        self.dataX = dataX
        #get the labelY
        self.labelY = labelY
        #get the number of sample
        self.m = len(dataX)
        #store the number of feature
        self.n = np.shape(dataX)[1]
        #store the alphas
        self.alphas = list(0 for i in range(self.m))
        #store the threshold
        self.b = float(0)
        #store the weight
        self.w = list(0 for i in range(self.n))
        #store the y error
        self.eCache = np.zeros((self.m,2),dtype = float)
        #store the kernel info  ['RBF','Linear','Polynomial']
        self.kernelName = kernelname
        #store the kerner parameters
        # 第一个参数为高斯核的参数theat,第二个参数和第三个参数分别为多项式核的参数d和p  ['1.0','1.0','1.0']
        self.kernelParameters = parameters
        # support vector machine
        self.sv = list()
        self.alphay = list()
    
    def normalData(self):
        means = np.mean(self.dataX,0)
        std = np.std(self.dataX,0)
        self.dataX = self.dataX - means
        self.dataX = self.dataX/std
        
    def matlibplotShow(self):
        fig = plt.figure()
        ax = fig.gca()
        self.dataX = np.array(self.dataX)
        self.labelY = np.array(self.labelY)

        flag = True
        # 画出原数据
        for i in range(self.m):
            if self.labelY[i] == 1 :
                p1, = ax.plot(self.dataX[i:, 0],self.dataX[i:, 1], 'ro')
            else:
                p2, = ax.plot(self.dataX[i, 0], self.dataX[i, 1], 'bo')
        # 画出支持向量
        for j in range(len(self.sv)):
            p3, = ax.plot(self.sv[j, 0], self.sv[j, 1], 'yo')
        # 画出超平面
        min_x = min(self.dataX[:, 0])
        max_x = max(self.dataX[:, 0])
        y_min_x = float(-self.b - self.w[0] * min_x) / self.w[1]
        y_max_x = float(-self.b - self.w[0] * max_x) / self.w[1]
        p4, = ax.plot([min_x, max_x], [y_min_x, y_max_x], '-g',markersize=10)
        # 设置显示的格式
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.title("SVM Sample Distribution")
        plt.legend([p1, p2, p3, p4], ["Positive samples", "Negative samples", "support vector", "Hyperplane" ])
        plt.show()
        
    def dotVector(self,i,j):    #calc the dot of i and j
        dot = float(0)
        for k in range(self.n):
            dot += self.dataX[i,k]*self.dataX[j,k]
        return dot
    
    def kernel(self,xi,yi):     #set the kernels
        """
         set the kerner function
        """
        #K(x,y) = exp(-||x-y||^2 / 2*theat*theat)
        if self.kernelName == 'RBF':                  # 高斯核
            return math.exp((2*self.dotVector(xi,yi)-self.dotVector(xi,xi)-self.dotVector(yi,yi)) \
                            /(2*float(self.kernelParameters[0])*float(self.kernelParameters[0])))
        elif self.kernelName == 'Linear':            # 线性核，和常规的一样
            return self.dotVector(xi,yi)
        elif self.kernelName == 'Polynomial':        # 多项式核  k(x,y) = (x.y+b)^p
            return (self.dotVector(xi,yi) + float(self.kernelParameters[1])) ** int(self.kernelParameters[2])
        
    def dotVector_v_u(self,v,u):
        dot = float(0.0)
        for i in range(len(v)):
            dot += v[i]*u[i]
        return dot
    
    def KernerlVector(self,v,u):

        if self.kernelName == 'RBF':                  # 高斯核
            return math.exp((2*self.dotVector_v_u(v,u)-self.dotVector_v_u(v,v)-self.dotVector_v_u(u,u)) \
                            /(2*float(self.kernelParameters[0])*float(self.kernelParameters[0])))
        elif self.kernelName == 'Linear':            # 线性核，和常规的一样
            return self.dotVector_v_u(v,u)
        elif self.kernelName == 'Polynomial':        # 多项式核  k(x,y) = (x.y+b)^p
            return (self.dotVector_v_u(v,u) + float(self.kernelParameters[1])) ** int(self.kernelParameters[2])
        
    def calcUi(self,k):
        if self.kernelName == 'Linear':
            """dot = float(0.0)
            for i in range(self.m):
                ui = float(0)
                for j in range(self.n):
                    ui = self.dataX[i,j]*self.dataX[k,j]
                dot += ui *self.alphas[i]*self.labelY[i]
            return (dot + self.b)
            """
            ui = 0.0
            for j in range(self.n):
                ui += self.w[j]*self.dataX[k,j]
            ui = ui + self.b
            return ui

        else:
            kernelUi = float(0)
            for i in range(len(self.alphas)):
                if self.alphas[i] > 0:
                    Kernel = self.kernel(k,i)
                    kernelUi += self.alphas[i]*self.labelY[i]*Kernel
            kernelUi = kernelUi + self.b
            return kernelUi

    def maxWalpahs(self,i1,j2,alphas1new,alphas2new,E1,E2,k11,k12,k22):
        alphas1 = self.alphas[i1]
        alphas2 = self.alphas[j2]
        y1 = self.labelY[i1]
        y2 = self.labelY[j2]
        s = y1*y2
        
        w = alphas1new *(y1*(self.b - E1) + alphas1*k11 + s*alphas2*k22) + \
            alphas2new *(y2*(self.b - E2) + alphas2*k22 + s*alphas1*k12) - \
            k11*alphas1new*alphas1new/2 - k22*alphas2new*alphas2new/2 -\
            s*k12*alphas1new*alphas2new
        return w
    
    def innerLoop(self,i1):
        alphas1 = self.alphas[i1]
        y1 = self.labelY[i1]
        u1 = self.calcUi(i1)
        E1 = u1 - y1
        alphas1new = float(-1)
        alphas2new = float(-1)
        #判断KKT条件
        if (y1*E1 > self.tolerance and alphas1 > 0) or (y1*E1 < -self.tolerance and alphas1 < self.C):
            #返回寻找的第二个参数lagrange
            j2 = self.findSecondVariable(i1,E1)
            if i1 == j2:
                return 0
            alphas2 = self.alphas[j2]
            y2 = self.labelY[j2]
            E2 = self.calcUi(j2) - y2
            s = y1*y2
            k11 = self.kernel(i1,i1)
            k22 = self.kernel(j2,j2)
            k12 = self.kernel(i1,j2)
            
            eta = k11 + k22 - 2*k12
            L = float(0)
            H = float(0)
            if y1*y2 == -1:
                L = max(0,alphas2 - alphas1)
                H = min(self.C,self.C + alphas2 - alphas1 )
            elif y1*y2 == 1:
                L = max(0,alphas2 + alphas1 - self.C)
                H = min(self.C,alphas2 + alphas1)
            if H == L:
                return 0
            if eta > 0:
                alphas2new = alphas2 + float(y2*(E1 - E2))/eta
                #防止越界
                if alphas2new < L:
                    alphas2new = L
                elif alphas2new > H:
                    alphas2new = H   
            else:
                alphas1new = alphas1 + s * (alphas2 - L)
                w1 = self.maxWalphas(i1,j2,alphas1new,L,E1,E2,k11,k12,k22)
                alphas1new = alphas1 + s * (alphas2 - H)
                w2 = self.maxWalphas(i1,j2,alphas1new,H,E1,E2,k11,k12,k22)
                if w1 - w2 > 0.001:
                    alphas2new = L
                elif w2 -w1 > 0.001:
                    alphas2new = H
                else:
                    alphas2new = alphas2
            if math.fabs((alphas2new - alphas2)) < 0.001:
                return 0
            
            alphas1new = alphas1 + s * (alphas2 - alphas2new)
            if alphas1new < 0:
                alphas1new = 0
            elif alphas1new > self.C:
                alphas2new += s*(alphas1new - self.C)
                alphas1new = self.C
            
            b1 = (alphas1-alphas1new) * y1 * k11 + (alphas2 - alphas2new) * y2 *k12 - E1 + self.b
            b2 = (alphas1-alphas1new) * y1 * k12 + (alphas2 - alphas2new) * y2 *k22 - E2 + self.b
            
            if alphas1new > 0 and alphas1new < self.C:
                self.b = b1
            elif alphas2new > 0 and alphas2new < self.C:
                self.b = b2
            else:
                self.b = (b1+b2)/2
            #更新新的alphas,eCache的值
            self.alphas[i1] = alphas1new
            self.alphas[j2] = alphas2new
            self.eCache[i1] = [1,E1]
            self.eCache[j2] = [1,E2]
            #更新线性的W值
            if self.kernelName == 'Linear':
                for j in range(0,self.n):
                    #self.w[j] = alphas1new * y1 * self.dataX[i1,j] + alphas2new * y2 * self.dataX[j2,j]
                    self.w[j] += (alphas1new - alphas1) * y1 * self.dataX[i1, j] \
                                 +(alphas2new - alphas2) * y2 * self.dataX[j2, j]
            return 1
        else:
            return 0

    def findSecondVariable(self, i1, E1):
        """
          find the second lagrange variable
          s.t. max|E1-E2| or random in all the sample
        """
        j2 = int(-1)
        maxdiff = -np.inf
        for j in range(len(self.alphas)):
            if self.alphas[j] > 0 and self.alphas[j] < self.C:
                y2 = self.labelY[j]
                E2 = self.calcUi(j) - y2
                tempdiff = math.fabs(E1-E2)
                if tempdiff > maxdiff:
                    maxdiff = tempdiff
                    j2 = j
        if j2 != -1:     #找到满足max|E1-E2|条件的J
            return j2
        else:            # 从non-bound中随机寻找一个
            k = random.randint(0,len(self.alphas)-1)
            for i in range(0,len(self.alphas)):
                j2 = (i+k)%len(self.alphas)
                if self.alphas[j2] > 0 and self.alphas[j2] < self.C and j2 != i1:
                    return j2;
            #如果没有在non-bound中找到，就在整个所有的lagrange multipliers中随机产生一个
            j2 = i1
            while j2 == i1:
                j2 = random.randint(0,len(self.alphas)-1)
            return j2

    def svmTrain(self,trainDataX,trainLabelY):
        """
           trainning the data 
        """
        self.dataX = trainDataX
        self.labelY = trainLabelY
        assert(len(self.dataX) == len(self.labelY))

        iter = int(0)
        entireAllSet = True
        numberchanged = int(0)
        
        while numberchanged > 0 or entireAllSet:
            numberchanged = int(0)
            if entireAllSet:
                for i in range(self.m):
                    numberchanged += self.innerLoop(i)
                iter = iter + 1
            else :
                for j in range(self.m):
                    if self.alphas[j] > 0 and self.alphas[j] < self.C:
                        numberchanged += self.innerLoop(j)
                iter = iter + 1
            if entireAllSet == True:
                entireAllSet = False
            elif numberchanged == 0:
                entireAllSet = True
        else:
            """
             其他的操作
            """
            #存储支持向量
            index = list()
            for i in range(0,len(self.alphas)):
                if self.alphas[i] > 0:
                    index.append(i)
                    self.alphay.append(self.alphas[i] * self.labelY[i])
                    
            self.sv = np.zeros((len(index),self.n),dtype = float)
                
            for i in range(0, len(index)):
                self.sv[i,:]= self.dataX[index[i],:]

    def testSample(self,testDataX,testLabelY):
        self.dataX = testDataX
        self.labelY = testLabelY
        assert(len(self.dataX) == len(self.labelY))
        testy = int(0)
        
        correctNum = int(0)
        for i in range(len(self.dataX)):
            if self.kernelName == 'Linear':
                testy =self.KernerlVector(self.w,self.dataX[i,:]) +self.b
            else:
                for j in range(0, len(self.alphay)):                  
                    testy += self.alphay[j] * self.KernerlVector(self.sv[j,:], self.dataX[i,:]) 
                testy += self.b

            if testy * self.labelY[i] >= 0:
                correctNum += 1
            #print("real label:",self.labelY[i],"test label:",testy)
        print("the number of samples is :",len(self.dataX))
        print("corretc number is :",correctNum)
        print "Accuracy is ",float(correctNum)/len(self.dataX)*100,"%"

    def calssify(self,data):
        self.dataX = data;
        # 存放类标签的列表
        self.labelY = list()
        m = len(self.dataX)
        for i in range(m):
            if self.kernelName == 'Linear':
                testy =self.KernerlVector(self.w,self.dataX[i,:]) + self.b
            else:
                for j in range(0, len(self.alphay)):
                    testy += self.alphay[j] * self.KernerlVector(self.sv[j,:], self.dataX[i,:])
                testy += self.b

            if testy >= 0:
                y = 1
                self.labelY.append(y)
            else:
                y = -1
                self.labelY.append(y)
        print "the class of samples is:\n", self.labelY

# 加载数据
def loadData(filePath):
    dataSet = np.loadtxt(filePath,dtype = np.float)
    dataX = np.copy(dataSet[:,:-1])
    labelY = np.copy(dataSet[:,-1])
    return dataX,labelY


if __name__ == "__main__":

    data ,label= loadData("F:\machine learning\PythonCode\dataset\svmdata\\train.txt")
    #---------训练数据----------
    kernelname = 'Linear'
    parameters = [0.01,2,3]
    svmObject = SVM(data,label,kernelname,parameters)
    svmObject.normalData()
    svmObject.svmTrain(data,label)
    print "the values of w vector：\n",svmObject.w,"\nthe value of b ：\n",svmObject.b
    svmObject.matlibplotShow()
    #---------测试数据-----------
    data ,label= loadData("F:\machine learning\PythonCode\dataset\svmdata\\test.txt")
    print "-----------test result:----------"
    svmObject.testSample(data,label)