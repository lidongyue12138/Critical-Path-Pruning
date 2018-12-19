from vggNet import Model
from CIFAR10_DataLoader import CifarDataManager, display_cifar

d = CifarDataManager()
print("Number of train images: {}".format(len(d.train.images)))
print("Number of train labels: {}".format(len(d.train.labels)))
print("Number of test images: {}".format(len(d.test.images)))
print("Number of test images: {}".format(len(d.test.labels)))
images = d.train.images
# display_cifar(images, 10)
print(images.shape)


model = Model()

model.build_model(100)

model.restore_model()

for i in range(10):
    generatedGate = model.encode_input(d.train.images[0].reshape((1,32,32,3)))
    print(generatedGate)