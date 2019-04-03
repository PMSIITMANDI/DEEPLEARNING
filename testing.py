import os

currentDir = os.getcwd()

print(currentDir+"/mnist_weightFile"+"/mnist-{epoch:02d}-{loss:.4f}.hdf5")
