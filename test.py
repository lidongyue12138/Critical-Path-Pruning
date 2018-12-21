from vggTrimmedModel import TrimmedModel
from CIFAR_DataLoader import CifarDataManager

d = CifarDataManager()


model = TrimmedModel()

# model.assign_weight()
test_images, test_labels = d.test.next_batch(200)

model.test_accuracy(test_images, test_labels)




  