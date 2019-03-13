import shutil
import os
import numpy as np
import cv2
from createLineFunc import createLine
from PIL import Image

def createLineForSpecifiedClass(length,width,theta,color):
    mydir="dataset"
   
    if (length==15):
        l=1
        ls = "long"
    else:
        l=0
        ls="short"
        
    if (width==3):
        w=1
        ws="thick"
    else:
        w=0
        ws="thin"
        
    
    if (color=="blue"):
        c=1
        cs="blue"
    else:
        c=0
        cs="red"
    t=int(theta/15)
    
    
    classFolder= ls+"_"+ws+"_"+cs+"_"+str(t)
    
    classFolderPath = "dataset"+"/"+classFolder
    
    try:
        shutil.rmtree(classFolderPath)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    
    
    os.mkdir(classFolderPath)
    print("Created class folder: "+ classFolder)
    
#    filename = str(l)+"_"+str(w)+"_"+str(t)+"_"+str(c)+"_"+"%s"%d+".jpg"
#    print(filename)
#    
#    
#    
#    filePath = "./"+mydir+"/"+classFolder+"/"+filename
#    #print(path)
#    print(classFolder)
    
    no_of_images=0
    testimage = np.zeros((28,28,3), np.uint8)
    while no_of_images <1000:
        
    
        for i in range(0,28):
            for j in range(0,28):
                testimage = createLine(testimage,i,j,length,width,"red") 
                testimage=Image.fromarray(testimage)
                testimage=testimage.rotate(theta)
                testimage=np.array(testimage)
                
                sumAll=int(np.sum(testimage))
                value = (width*length*255)
                if (sumAll==value):
            
                    no_of_images=no_of_images+1
                    print ("Image will be created")
                    
                    filename = str(l)+"_"+str(w)+"_"+str(t)+"_"+str(c)+"_"+"%s"%no_of_images+".jpg"
                    filePath = "./"+mydir+"/"+classFolder+"/"+filename
                    cv2.imwrite(filePath,testimage)
                else:
                    k=1
                    
                testimage = np.zeros((28,28,3), np.uint8)
    print("No of images created: %s"%no_of_images)
        
    return
    
    

    
            
    
    