import numpy as np

from CIFAR_DataLoader import CifarDataManager, display_cifar
from vggNet import Model

'''
Configuration:
    1. learning rate 
    2. L1 loss penalty
    3. Lambda cotrol gate threshold
'''
learning_rate = 0.1
L1_loss_penalty = 0.02
threshold = 0.1
''''''


d = CifarDataManager()
print("Number of train images: {}".format(len(d.train.images)))
print("Number of train labels: {}".format(len(d.train.labels)))
print("Number of test images: {}".format(len(d.test.images)))
print("Number of test images: {}".format(len(d.test.labels)))
images = d.train.images
# display_cifar(images, 10)
print(images.shape)


model = Model(
    learning_rate = learning_rate, 
    L1_loss_penalty = L1_loss_penalty,
    threshold = threshold
)

'''
Running Test
'''
for i in range(100, 101):
    generatedGate = model.compute_encoding(d.train.images[i].reshape((1,32,32,3)))
    print(generatedGate)

