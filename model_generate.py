from utils.music_utils import *
from tensorflow.keras.models import model_from_json

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
generator.create_playlist(num_songs=2, note_count=200)
