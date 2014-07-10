#-*- coding:utf-8 -*-
_author__ = 'guanyayong'

from pylab import imread, imshow, figure, show, subplot
from scipy.cluster.vq import kmeans, vq
from numpy import reshape, flipud
from PIL import Image
import mpl_toolkits.mplot3d
import matplotlib
import matplotlib.pyplot as plt

#聚类的个数分别为2,10,100,256
vq_clusters = [2, 10, 100, 256]

img = imread("person.jpg")

data = reshape(img, (img.shape[0] * img.shape[1], 3))

plt.ion()
fig1 = plt.figure(1)
fig2 = plt.figure(2)


plt.figure(1)
plt.subplot(231)
imshow(img)
plt.title("original image")
plt.draw()

for i in range(len(vq_clusters)):
    k = vq_clusters[i]
    print 'Generating vq-%d...' % k
    (centroids, distor) = kmeans(data, k)
    (code, distor) = vq(data, centroids)
    print 'distor: %.6f' % distor.sum()
    centers_idx = reshape(code, (img.shape[0], img.shape[1]))
    clustered = centroids[centers_idx]
    # 显示聚类处理后的效果
    plt.figure(1)
    if i < 2:
        plt.subplot(232 + i)
    else:
        plt.subplot(233 + i)
    imshow(clustered)
    plt.title(str(k) + " vector quantization")
    plt.draw()
    # 显示3D的效果
    plt.figure(2)
    plt.gca()
    ax = plt.subplot(221 + i, projection='3d')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=centroids / 255.)
    plt.title("centers " + str(i))
    plt.draw()

plt.ioff()
plt.show()
