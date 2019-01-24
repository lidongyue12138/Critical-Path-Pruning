import os 
import _pickle as cPickle

DATA_PATH = "cifar-100-python"
PATH = "./CalTech256/256_ObjectCategories"

'''
First Number is Index in Cifar100
Second Number is Index in Caltech256: Note this number needs to be +1
So for example, for "bear" class:
    in Cifar100: designate as 3
    in Caltech256: designate as 8+1 = 9

Mutual Class: bear 3 8
Mutual Class: butterfly 14 23
Mutual Class: camel 15 27
Mutual Class: cockroach 24 39
Mutual Class: mushroom 51 146
Mutual Class: porcupine 63 163
Mutual Class: raccoon 66 167
Mutual Class: skunk 75 185
Mutual Class: skyscraper 76 186
Mutual Class: snail 77 188
Mutual Class: snake 78 189
Mutual Class: spider 79 197
'''

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
        return dict

meta_dict = unpickle("meta")
Cifar_label_names = meta_dict["fine_label_names"]
# print(Cifar_label_names)

Caltech_label_names = os.listdir(PATH)
Caltech_label_names = [name.split(".")[1] for name in Caltech_label_names]
# print(Caltech_label_names)

for Cifar_name in Cifar_label_names:
    if Cifar_name in Caltech_label_names:
        print("Mutual Class: " + Cifar_name + 
              " " + str(Cifar_label_names.index(Cifar_name)) + 
              " " + str(Caltech_label_names.index(Cifar_name)))
