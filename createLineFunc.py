def createLine(npArray,i,j,length,width,color):
    npArray[..., 0] = 0
    npArray[..., 1] = 0
    npArray[..., 2] = 0
    
    if (color =="blue"):
        ch=2
    else:
        ch=0
        
    npArray[i:i+width,j:j+length,ch]=255
    return npArray