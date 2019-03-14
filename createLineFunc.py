def createLine(npArray,i,j,length,width,color):
    npArray[..., 0] = 0
    npArray[..., 1] = 0
    npArray[..., 2] = 0
    
    if (color =="blue"):
        ch=0
    else:
        ch=2
        
    npArray[i:i+width,j:j+length,ch]=255
    return npArray