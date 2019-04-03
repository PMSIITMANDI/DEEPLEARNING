def data_preprocessing(path):

    import os
    import cv2
    import numpy as np
    
    
    x_test=[]
    x_train=[]
    y_test=[]
    y_train=[]
    class_name=[]
    
    #path = "./dataset/"
    count =0
    
    dataset_path = path
    allFolders = os.listdir(dataset_path)
    allFolders.sort()
    for folder in allFolders:
        strn= str(folder)+"$_$:"+str(count)
        class_name.append(strn)
        insideFolderPath = (dataset_path+folder+"/")
        allFiles=os.listdir(insideFolderPath)
        allFiles.sort()
        for i in range (0,600):
            # print(insideFolderPath+allFiles[i])
            I = cv2.imread(insideFolderPath+allFiles[i])
            x_train.append(I)
            y_train.append(count)
        for i in range (600,1000):
            # print(insideFolderPath+allFiles[i])
            I = cv2.imread(insideFolderPath+allFiles[i])
            x_test.append(I)
            y_test.append(count)
        count = count+1
    return np.asarray(x_train),np.asarray(x_test),np.asarray(y_train),np.asarray(y_test)