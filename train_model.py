from utils1.music_utils import *
from modelxarchitecture.modelxarchitecture import MyModel
import sys

# Getting arguments
arguments = sys.argv
if len(arguments) == 1:
    file1 = "beeth"
    length1 = 30
elif len(arguments)==2:
    file1 = arguments[1]
    length1 = 30
elif len(arguments)==3:
    file1 = arguments[1]
    length1 = int(arguments[2])
else:
    print("Invalid number of arguments")
    sys.exit(1)

print(file1, length1)
keyword = "_dummy"
# To get mappings
file_path = f'data/{file1}'
dataset = CreateDataset(file_path, length=length1, name=file1, save_memory=True)
dataset.make_mappings()

# Creating train and generate sets
test_size = 0.01
random_state = 42
x_train, x_gen, y_train, y_gen = train_test_split(dataset.features_normalized, dataset.labels_encoded, test_size=test_size, random_state=random_state)

np.save(f'genxsets/{file1}_{dataset.length}{keyword}.npy', x_gen) # saving generate data

num_features = 1
x_train = x_train.reshape((-1, dataset.length, num_features)) # reshaping train set
# print(x_train[:20, :, :])
print(y_train[:20])
print(length1, type(length1))
# Creating the model
scaling = 4
model_args = {}
model_args["input_shape_"] = x_train.shape[1:]
model_args["input_units"] = len(dataset.mapping)//scaling
model_args["output_units"] = y_train.shape[1]
model = MyModel(**model_args)

loss_param = "categorical_crossentropy"
optimizer_param = "adam"
model.compile(loss=loss_param, optimizer=optimizer_param)

# defining the callbacks
def create_callbacks():
    callbacks = [EarlyStopping(monitor="loss", min_delta=0.002, patience=10, verbose=1, mode="auto", baseline=None, restore_best_weights=True,),
                 ReduceLROnPlateau(monitor="loss", factor=0.1, patience=4, verbose=1, mode="auto", min_delta=0.01, cooldown=0, min_lr=1e-9)]
    return callbacks

# training the model
num_epochs = 20 # number of epochs to train
batch_size = 1
callbacks = create_callbacks()
history = model.fit(x_train, y_train, batch_size=batch_size,epochs=num_epochs, callbacks=callbacks, verbose=1)

print("completed")
# saving the model
path_to_save = './'
filename = dataset.name+f"_{dataset.length}_dummy"
save_model_as_json_h5(path_to_save, filename, model)

# Train plot
def make_train_plot():
    train_loss = history.history['loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(path_to_save+f'/{name}_{dataset.length}{keyword}.png')
make_train_plot()

