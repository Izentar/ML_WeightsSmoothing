import os
if None:
    print ("OK")
else:
    print ("Not ok")
'''
directory = './folder/'
w = os.walk('./folder/')
files = filter(os.path.isfile, [os.path.join(directory, i) for i in os.listdir(directory)])
for f in files:
    print(str(f))

files2_ = []
for i in os.listdir(directory):
    if(os.path.isfile(os.path.join(directory, i))) and '.txt' in i:
        files2_.append(i)

for f in files2_:
    print(str(f))

#print(os.path.isdir('./folder/innerFolder'))
print(os.listdir(directory))
#print(os.listdir(os.curdir))

#print(os.path.basename(os.path.dirname(os.path.realpath(__file__))))

for f in os.scandir(directory):
    print(str(f.name))

def getSubDirs(pathList: list, path: str, partialPath: str = ""):
    for i in os.listdir(path):
        if(os.path.isdir(os.path.join(path, i))):
            partialPathtmp = partialPath + '/' + i
            pathList.append(partialPathtmp)
            getSubDirs(pathList, path + '/' + i, partialPathtmp)

pathList = []
getSubDirs(pathList, directory)
print(pathList)'''

a = 98as
print(a)