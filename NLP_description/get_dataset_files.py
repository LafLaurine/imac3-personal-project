from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir("../dataset/FlickerImages") if isfile(join("../dataset/FlickerImages", f))]

f = open("../dataset/trainImages.txt","w+")
for i in range(int(len(onlyfiles))):
    f.write(onlyfiles[i] + "\n")

f.close() 