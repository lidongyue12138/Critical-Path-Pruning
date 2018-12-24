import numpy as np
import json
from CIFAR_DataLoader import CifarDataManager, display_cifar
from vggNet import Model

'''
Configuration:
    1. learning rate 
    2. L1 loss penalty
    3. Lambda cotrol gate threshold
'''
learning_rate = 0.1
L1_loss_penalty = 0.03
threshold = 0.0
''''''

layer_names = ['FC14/gate:0', 
    'Conv7/composite_function/gate:0', 
    'Conv12/composite_function/gate:0', 
    'Conv2/composite_function/gate:0', 
    'Conv4/composite_function/gate:0', 
    'Conv1/composite_function/gate:0', 
    'Conv6/composite_function/gate:0', 
    'Conv10/composite_function/gate:0', 
    'Conv9/composite_function/gate:0', 
    'FC15/gate:0', 
    'Conv13/composite_function/gate:0', 
    'Conv5/composite_function/gate:0', 
    'Conv3/composite_function/gate:0', 
    'Conv11/composite_function/gate:0', 
    'Conv8/composite_function/gate:0'
]


def calculate_total_by_threshold(classid):
	name_shape_match = [
		{'name': 'FC14/gate:0', 'shape': 4096}, 
		{'name': 'Conv7/composite_function/gate:0', 'shape': 256}, 
		{'name': 'Conv12/composite_function/gate:0', 'shape': 512}, 
		{'name': 'Conv2/composite_function/gate:0', 'shape': 64}, 
		{'name': 'Conv4/composite_function/gate:0', 'shape': 128}, 
		{'name': 'Conv1/composite_function/gate:0', 'shape': 64}, 
		{'name': 'Conv6/composite_function/gate:0', 'shape': 256}, 
		{'name': 'Conv10/composite_function/gate:0', 'shape': 512}, 
		{'name': 'Conv9/composite_function/gate:0', 'shape': 512}, 
		{'name': 'FC15/gate:0', 'shape': 4096}, 
		{'name': 'Conv13/composite_function/gate:0', 'shape': 512}, 
		{'name': 'Conv5/composite_function/gate:0', 'shape': 256}, 
		{'name': 'Conv3/composite_function/gate:0', 'shape': 128}, 
		{'name': 'Conv11/composite_function/gate:0', 'shape': 512}, 
		{'name': 'Conv8/composite_function/gate:0', 'shape': 512}
	]
	for i in range(len(name_shape_match)):
		name_shape_match[i]['shape'] = name_shape_match[i]['shape']*[0]
	for i in range(499):
		jsonpath = "./ImageEncoding/class" + str(classid) + "-pic" + str(i) + ".json"
		with open(jsonpath,'r') as f:
			dataset = json.load(f)
			for gate in range(len(dataset)):
				for index in range(len(name_shape_match)):
					if name_shape_match[index]['name'] == dataset[gate]['layer_name']:
						tmp = dataset[gate]['layer_lambda']
						for conv in range(len(tmp)):
							if tmp[conv] > 0.1: # threshold 0.1
								name_shape_match[index]['shape'][conv] += 1
							else:
								pass
					else:
						pass
	json_write_path = "./ClassEncoding/class" + str(classid) + ".json"
	with open(json_write_path,'w') as g:
		json.dump(name_shape_match,g,sort_keys=True, indent=4, separators=(',', ':'))

def calculate_total_by_weights(classid):
	name_shape_match = [
		{'name': 'FC14/gate:0', 'shape': 4096}, 
		{'name': 'Conv7/composite_function/gate:0', 'shape': 256}, 
		{'name': 'Conv12/composite_function/gate:0', 'shape': 512}, 
		{'name': 'Conv2/composite_function/gate:0', 'shape': 64}, 
		{'name': 'Conv4/composite_function/gate:0', 'shape': 128}, 
		{'name': 'Conv1/composite_function/gate:0', 'shape': 64}, 
		{'name': 'Conv6/composite_function/gate:0', 'shape': 256}, 
		{'name': 'Conv10/composite_function/gate:0', 'shape': 512}, 
		{'name': 'Conv9/composite_function/gate:0', 'shape': 512}, 
		{'name': 'FC15/gate:0', 'shape': 4096}, 
		{'name': 'Conv13/composite_function/gate:0', 'shape': 512}, 
		{'name': 'Conv5/composite_function/gate:0', 'shape': 256}, 
		{'name': 'Conv3/composite_function/gate:0', 'shape': 128}, 
		{'name': 'Conv11/composite_function/gate:0', 'shape': 512}, 
		{'name': 'Conv8/composite_function/gate:0', 'shape': 512}
	]
	for i in range(len(name_shape_match)):
		name_shape_match[i]['shape'] = name_shape_match[i]['shape']*[0]
	for i in range(499):
		jsonpath = "./ImageEncoding/class" + str(classid) + "-pic" + str(i) + ".json"
		with open(jsonpath,'r') as f:
			dataset = json.load(f)
			for gate in range(len(dataset)):
				for index in range(len(name_shape_match)):
					if name_shape_match[index]['name'] == dataset[gate]['layer_name']:
						tmp = dataset[gate]['layer_lambda']
						for conv in range(len(tmp)):
							name_shape_match[index]['shape'][conv] += tmp[conv]
					else:
						pass
	json_write_path = "./ClassEncoding/class" + str(classid) + ".json"
	with open(json_write_path,'w') as g:
		json.dump(name_shape_match,g,sort_keys=True, indent=4, separators=(',', ':'))

d = CifarDataManager()
model = Model(
    learning_rate = learning_rate, 
    L1_loss_penalty = L1_loss_penalty,
    threshold = threshold
)
def run():
    # choose class data
    # train_images, train_labels = d.train.generateSpecializedData(class_id = 1, count = 5)
    # model.encode_class_data(1, train_images)
    calculate_total_by_weights(0)

run()

