import os
import shutil

from multipleClassGenerationFunction import createLineForSpecifiedClass

count=0


try:
    shutil.rmtree("dataset")
except OSError as e:
    #print ("Error: %s - %s." % (e.filename, e.strerror))
    k=1
    
    
os.mkdir("dataset")
    
    
length=[7,15]
width=[1,3]
theta=[0,15,30,45,60,75,90,105,120,135,150,165]
color =["blue","red"]

for l in length:
    for w in width:
        for t in theta:
            for c in color:
                createLineForSpecifiedClass(l,w,t,c)
                count=count+1
                
print ("NO of time function is called is: %s"%count)
                
