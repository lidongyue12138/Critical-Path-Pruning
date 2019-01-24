import os
import sys
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PATH = "./CalTech256/256_ObjectCategories"
PATH_Resized = "./CalTech256/ResizedImages"

'''
TODO:
    - Shuffle all images to target directory
    - Record Corresponding Labels 
    - Resize Images
    - Generate Train Dataset and Test Dataset
'''

def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CalTechLoader(object):
    def __init__(self, sourceDirs):
        self.images = None
        self.labels = None

        self._i = 0
        self.imgDirs = sourceDirs
    
    def load(self):
        imageList = []
        labelList = []
        for imgDir in self.imgDirs:
            tmpImage = plt.imread(os.path.join(PATH_Resized, imgDir))
            tmpLabel = imgDir.strip(".jpg")
            _, tmpLabel = tmpLabel.split("_")
            tmpLabel = int(tmpLabel)
            if tmpImage.shape == (32, 32):
                continue
            imageList.append(tmpImage)
            labelList.append(tmpLabel)

        self.images = np.array(imageList).astype(float)
        self.labels = np.array(labelList)
        self.length = len(self.labels)

        self.images = self.normalize_images(self.images)
        
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size], self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, one_hot(y, 256) 

    def next_batch_without_onehot(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size], self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    # calculate the means and stds for the whole dataset per channel
    def measure_mean_and_std(self, images):
        means = []
        stds = []
        for ch in range(images.shape[-1]):
            means.append(np.mean(images[:, :, :, ch]))
            stds.append(np.std(images[:, :, :, ch]))
        return means, stds

    # normalization for per channel
    def normalize_images(self, images):
        images = images.astype('float64')
        means, stds = self.measure_mean_and_std(images)
        for i in range(images.shape[-1]):
            images[:, :, :, i] = ((images[:, :, :, i] - means[i]) / stds[i])
        return images

class CalTechDataManager(object):
    def __init__(self):
        self.image_dirs = os.listdir(PATH_Resized)
        self.data_length = len(self.image_dirs)
        print("-------------------Begin Loading Caltech 256 Dataset-----------------------------")

        self.train = CalTechLoader(sourceDirs=self.image_dirs[:(self.data_length//5)*4]).load()
        self.test = CalTechLoader(sourceDirs=self.image_dirs[(self.data_length//5)*4+1:]).load()

        print("-------------------Loading Caltech 256 Dataset Complete--------------------------")

def resize_images(new_size, out_fold):
    listDirs = os.listdir(path=PATH)
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)
    for curClasses in listDirs:
        listImgs = os.listdir(os.path.join(PATH,curClasses))
        for imgName in listImgs:
            try:
                image = os.path.join(PATH, curClasses, imgName)

                imgName = imgName.strip(".jpg")
                classNum, imgNum = imgName.split("_")
                img = Image.open(image)
                img = img.resize((int(new_size),int(new_size)),Image.ANTIALIAS)
                
                imgNum = str(random.randint(100000, 999999))
                newName = imgNum + "_" + classNum 
                newPath = os.path.join(out_fold, newName + ".jpg")
                img.save(newPath,"JPEG",quality=90)
            except:
                pass

def display_images(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
    for i in range(size)])
    plt.imshow(im)
    plt.show()

if __name__ == "__main__":
    d = CalTechDataManager()

    images, labels = d.train.next_batch_without_onehot(100)
    print(images.shape)