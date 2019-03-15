import numpy as np
import math
import tensorflow as tf
from tempfile import TemporaryFile
final_position = TemporaryFile()
final_velocity = TemporaryFile()
row,col=100,2
g=(6.667)*(10^5)
t=10^((-1)*4)
a=np.zeros([row,col])
def my_position(x,u,a,t):
    xr = x[:, 0]
    xc = x[:, 1]
    ur=u[:, 0]
    uc = u[:, 1]
    ar = a[:, 0]
    ac = a[:, 1]
    xr,xc = xr+ur*t+0.5*ar*t*t,xc+uc*t+0.5*ac*t*t

    return xr,xc
def my_velociity(u,a,t):
    ur = u[:, 0]
    uc = u[:, 1]
    ar = a[:, 0]
    ac = a[:, 1]
    v1,v2= ur+ar*t,ur+ar*t
    return v1,v2

def my_accelaration(m,x,g):
    xr=x[:,0]
    xc=x[:,1]
    acc = tf.zeros([100, 2], 'float64')
    one=tf.ones([100,100],'float64')
    xrwhole=tf.multiply(one,xr)
    xcwhole=tf.multiply(one,xc)
    xrrelative=tf.transpose(xrwhole)-xrwhole
    xcrelative=tf.transpose(xcwhole)-xcwhole
    xrrelative=tf.square(xrrelative)
    xcrelative=tf.square(xcrelative)
    x_dist=abs(tf.sqrt(tf.add(xrrelative,xcrelative)))
    tf.where(tf.less(x_dist, 1e-6), x_dist, 1. / x_dist)
    mbyr = tf.divide(tf.transpose(m), tf.pow(x_dist,3))
    axrrelative=tf.multiply(mbyr,xrrelative)
    axcrelative=tf.multiply(mbyr,xrrelative)
    acc=g*tf.concat([axrrelative,axcrelative],axis=1)
    accelaration=tf.reduce_sum(acc, 0)
    accelaration=tf.reshape(accelaration, [100, 2])
    return accelaration

m = np.load('/home/mahesh/PycharmProjects/hello/Deep Learning/Assignment1/masses.npy')
x = np.load('/home/mahesh/PycharmProjects/hello/Deep Learning/Assignment1/positions.npy')
u = np.load('/home/mahesh/PycharmProjects/hello/Deep Learning/Assignment1/velocities.npy')

count=0
distance=1.0
# acce=my_accelaration(m,x,g)
# sess=tf.Session()
# print(sess.run(tf.shape(acce)))
# print(sess.run(acce))

while distance>1.0:
    sess = tf.Session()
    a = my_accelaration(m, x, g)
    xr,xc = my_position(x, u, a, count * t)
    ur,uc = my_velociity(u, a, count * t)
    xr = tf.square(xr)
    xc = tf.square(xc)
    dist = abs(tf.sqrt(tf.add(xr, xc)))
    i,j = tf.where(dist < 0.1, x=None, y=None, name=None)
    id = tf.constant([dist])
    distance = tf.gather_nd(x, [i,j])
    tf.cond(distance<0.1,print(sess.run(distance)))












