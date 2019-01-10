from vggTrimmedModel import TrimmedModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np
d = CifarDataManager()


model = TrimmedModel()

'''
Todo List: 
    1. Modify the accuracy in trimmed network (Done)
'''
for _ in range(50):
    test_images, test_labels = d.test.next_batch(200)

    model.test_accuracy_pretrim(test_images, test_labels)

    model.assign_weight()
    model.test_accuracy(test_images, test_labels)
    break




