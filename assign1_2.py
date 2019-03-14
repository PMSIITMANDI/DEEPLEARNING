import numpy as np
import math
from tempfile import TemporaryFile
final_position = TemporaryFile()
final_velocity = TemporaryFile()
row,col=100,2
g=(6.667)*(10^5)
t=10^((-1)*4)
a=np.zeros([row,col])
def my_position(x,u,a,t):
    x1 = x+u*t+0.5*a*t*t
    return x1
def my_velociity(u,a,t):
    v1= u+a*t
    return v1
def my_accelaration(m,x,g):
    sum=np.zeros([row,col])
    r = np.zeros([row, col])
    for i in range(row):
        for k in range(row):
            if k!=i:
                r[k,:]=x[i,:]-x[k,:]
                r_dist=math.sqrt(r[k][0]**2+r[k][1]**2)
                r_dist=abs(r_dist)**3
                for j in range(2):
                    sum[i][j]=sum[i][j]+(-1/r_dist)*g*m[i]*r[k][j]
    return sum
m = np.load('./dataset/masses.npy')
x = np.load('./dataset/positions.npy')
u = np.load('./dataset/velocities.npy')
count=0
threshold=1.0
while threshold>=0.1:
    a=my_accelaration(m,x,g)
    x = my_position(x, u, a, count*t)
    u = my_velociity(u, a, count*t)
    s = np.zeros([row, col])
    for i in range(row):
        for k in range(row):
            if k!=i:
                s[k,:]=x[i,:]-x[k,:]
                s_dist=math.sqrt(s[k][0]*s[k][0]+s[k][0]*s[k][0])
                threshold=s_dist
                if threshold <0.1:
                    print("threshold\t",threshold)
                    print("\nparticles\t",i,k)
                    print("\nposition vector\t",x)
                    np.save('final_position', x)
                    print("\nvelocity\t",u)
                    np.save('final_velocity', u)
                    print("\nnumber of step\t",count)
                    print("\naccelaration\t", a)

                else:
                    count=count+1

            if threshold < 0.1:
                break
        if threshold < 0.1:
            break

