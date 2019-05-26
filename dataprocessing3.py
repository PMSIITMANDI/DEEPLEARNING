def data(path,path1):
    import cv2
    import glob
    import numpy as np
    import os
    
    file_list = sorted(glob.glob(os.path.join(os.getcwd(), path1, "*.txt")))
     
    y_train=[]
     
    for file_path in file_list:
        
        k=np.loadtxt(file_path)
        
        y_train.append(k)
    y_train=np.asarray(y_train)
    
    def datafinal(y_train):
        file_list = sorted(glob.glob(os.path.join(os.getcwd(), path, "*.jpeg")))
         
        x_train=[]
        i=0 
        j=0
        for file_path in file_list:
            
            r_padding=0
            c_padding=0
            k0=cv2.imread(file_path)
            r_padding=int((600-k0.shape[0])*0.5)
            c_padding=int((400-k0.shape[1])*0.5)
            y_train[i][j]=int(y_train[i][j])+c_padding
            y_train[i][j+1]=int(y_train[i][j+1])+r_padding
            i+=1
            j=0
            print(i)
            k=np.pad(k0,((r_padding,r_padding),(c_padding,c_padding),(0,0)),'edge')
            x_train.append(k)
        x_train=np.asarray(x_train)
        return y_train,x_train
    
    
    
    y_train,x_train=datafinal(y_train)
    
    
    
    
    return x_train,y_train
