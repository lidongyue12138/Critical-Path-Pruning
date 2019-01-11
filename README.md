# CriticalPathPruning

### How to Run Our Code

We implement our pruning algorithm based on the **TensorFlow 1.4.1** with **CUDA 8.0**. We use **CIFAR-100** dataset and **VGG-16** network for all the experiments. Codes are available at [Github Link](https://github.com/lidongyue12138/CriticalPathPruning)  

#### Prerequisite

To run our code, you have to download: 

- [**CIFAR-100 Dataset**](https://www.cs.toronto.edu/~kriz/cifar.html)： Assuming the code is put in directory ".", please download the dataset and save it in the directory "./cifar-100-python".

- [**Pretrained VGG-16 Model**](https://github.com/BoyuanFeng/vggNet-71.56-on-CIFAR100-with-Tensorflow)：Assuming the code is put in directory ".", please download the following three files and save it in the directory "./vggNet". 

  - <https://drive.google.com/open?id=1fDZDf7UpsVCn4CGGvI-Jssm3iFQgpyJw>
  - <https://drive.google.com/open?id=1ZJm8-6HIDOLBWXBt92MQjUKqWRbq_xpz>
  - <https://drive.google.com/open?id=1nXcmco9zrJIkOTQTTa4V3_VbTVajExWI>

  Credit to BoyuanFeng, Github site: https://github.com/BoyuanFeng

And you need to install following python pachages:

- pickle
- json
- keras
- numpy
- tensorflow
- sklearn

We suggest you to install [Anaconda](https://www.anaconda.com/download/) for convenience

#### Run the Code

Therre are several steps to run the code:

1. **run.py**: this file generate class encodes. You should change classes or set loops to run all classes in this file
2. **trim_and_test.py**: this file trim the model and test the accuracy with pruned model which has not been fine tuned yet. Change *target_class_id* for models of different classes
3.  **run_finetune.py**: this file fine tune the pruned model and test the accuracy with fine tuned models. You should change *target_class_id* for models of different classes

Notice: we use GPU for training, so you should designate certain GPU for training in these files.