from vggTrimmedModel import TrimmedModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np
d = CifarDataManager()


model = TrimmedModel()
 
model.assign_weight()

'''
Todo List: 
    1. Modify the accuracy in trimmed network
'''
for _ in range(50):
    test_images, test_labels = d.test.next_batch(200)

    model.test_accuracy(test_images, test_labels)
