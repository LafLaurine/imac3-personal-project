from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(
    "dataset/Flicker8k_Dataset") if isfile(join("dataset/Flicker8k_Dataset", f))]
print(onlyfiles)

f = open("Flickr8k.trainImages.txt", "w")
for data in listdir("dataset/Flicker8k_Dataset"):
    f.write(data + '\n')
f.close()
