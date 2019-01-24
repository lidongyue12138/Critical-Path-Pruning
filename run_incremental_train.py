from CalTech_DataLoader import CalTechDataManager, one_hot
from vggFinetuneModel import FineTuneModel
import numpy as np

def modify_label(labels, test_classes = [0]):
    test_classes.sort()
    tmp_labels = []
    for i in labels:
        if i in test_classes:
            tmp_labels.append(test_classes.index(i))
        else:
            tmp_labels.append(len(test_classes))
    return one_hot(tmp_labels, vals=len(test_classes)+1)

data_loader = CalTechDataManager()

test_images, test_labels = data_loader.test.next_batch_without_onehot(500)

test_labels = modify_label(test_labels, test_classes = [9])

model = FineTuneModel(target_class_id=[3])
model.assign_weight()
model.test_accuracy(test_images, test_labels)

for i in range(10):
    train_images, train_labels = data_loader.train.next_batch_without_onehot(200)
    train_labels = modify_label(train_labels, test_classes = [9])
    model.train_model(train_images, train_labels)
    model.test_accuracy(test_images, test_labels)

