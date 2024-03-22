from utils.music_utils import *
from model_architecture import MyModel

import sys

# Getting arguments
arguments = sys.argv
file1 = None
length1 = None
if len(arguments) < 2:
    file1 = "beeth"
    length1 = 30
elif len(argumnets)<3:
    file1 = arguments[2]
    length1 = 30
elif len(arguments)<4:
    file1 = arguments[2]
    length1 = arguments[3]


# To get mappings
file_path = f'data/{file1}'
dataset = CreateDataset(file_path, length=length1, name=file1, save_memory=True)
dataset.make_mappings()

# Creating train and seed sets
test_size = 0.01
random_state = 42
x_train, x_seed, y_train, y_seed = train_test_split(dataset.features_normalized, dataset.labels_encoded, test_size=test_size, random_state=random_state)
x_train = x_train.reshape((1, *x_train.shape))

# Creating the model
scaling = 4
model_args = {}
model_args["input_shape"] = x_train.shape[1:]
model_args["input_units"] = len(dataset.mapping)//scaling
model_args["output_units"] = y_train.shape[1]

# model = MyModel(x_train.shape[1:], input_units, y_train.shape[1])
model = MyModel(**model_args)
loss_param = "categorical_crossentropy"
optimizer_param = "adam"
model.compile(loss=loss_param, optimizer=optimizer_param)
# x_samp = np.random.uniform(-1, 1, x_train.shape)
# result_samp = model(x_samp)
model.summary()

