def multi_class_seperation_processing(path):

    import os
    import cv2
    import numpy as np
    
    
    x_test=[]
    x_train=[]
    y_test_length=[]
    y_test_color=[]
    y_test_width=[]
    y_test_angle=[]
    
    y_train_length=[]
    y_train_color=[]
    y_train_width=[]
    y_train_angle=[]
    
    dataset_path = path
    allFolders = os.listdir(dataset_path)
    allFolders.sort()
    for folder in allFolders:
        className = folder.split("_")
        length = className[0]
        width = className[1]
        color = className[2]
        angle = int(className[3])
        
        if(length=="short"):
            l=0
        else:
            l=1
            
        if (width=="thin"):
            w=0
        else:
            w=1
        
        if (color =="red"):
            c=0
        else:
            c=1
                
        
        insideFolderPath = (dataset_path+folder+"/")
        allFiles=os.listdir(insideFolderPath)
        allFiles.sort()
        for i in range (0,600):
            print(insideFolderPath+allFiles[i])
            I = cv2.imread(insideFolderPath+allFiles[i])
            x_train.append(I)
            y_train_length.append(l)
            y_train_width.append(w)
            y_train_color.append(c)
            y_train_angle.append(angle)
        for i in range (600,1000):
            print(insideFolderPath+allFiles[i])
            I = cv2.imread(insideFolderPath+allFiles[i])
            x_test.append(I)
            y_test_length.append(l)
            y_test_width.append(w)
            y_test_color.append(c)
            y_test_angle.append(angle)
    return np.asarray(x_train),np.asarray(x_test),np.asarray(y_train_length),np.asarray(y_train_width),np.asarray(y_train_color),np.asarray(y_train_angle),np.asarray(y_test_length),np.asarray(y_test_width),np.asarray(y_test_color),np.asarray(y_test_angle)
#    return  (x_train), (x_test), (y_train_length), (y_train_width), (y_train_color), (y_train_angle), (y_test_length), (y_test_width), (y_test_color), (y_test_angle)