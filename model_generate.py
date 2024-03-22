from utils.music_utils import *
from tensorflow.keras.models import model_from_json
import sys

# Getting arguments
arguments = sys.argv

if len(arguments) < 2:
    num_songs = 2
    note_count = 200
elif len(argumnets)<3:
    num_songs = arguments[2]
    note_count = 200
elif len(arguments)<4:
    num_songs = arguments[2]
    note_count = arguments[3]


# Loading model archtecture
with open('models/beeth_30.json', 'r') as json_file:
    model_json = json_file.read()
model =  model_from_json(model_json)

# Loading model weights
model.load_weights('./models/beeth_30.keras')
print("successfully loaded!")

# Loading Seed data
x_seed = np.load('beeth_seed.npy')

# To get mappings
file_path = 'data/beeth'
dataset = CreateDataset(file_path, name="beeth")
dataset.make_mappings()
generator = Generator(model, x_seed, dataset)
generator.create_playlist(num_songs=num_songs, note_count=note_count)
print(f"{num_songs} Songs Created!")
