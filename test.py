from vggTrimmedModel import TrimmedModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np

d = CifarDataManager()
model = TrimmedModel(target_class_id = [0,1,2,3], multiPruning = True)
model.assign_weight()
test_images, test_labels = d.test.next_batch(200)
model.test_accuracy(test_images, test_labels)