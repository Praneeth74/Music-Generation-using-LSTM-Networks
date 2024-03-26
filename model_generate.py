from utils1.music_utils import *
from utils1.mytfutils import *
from tensorflow.keras.models import model_from_json
# from modelxarchitecture.modelxarchitecture import MyModel
import sys

# Getting arguments
arguments = sys.argv

if len(arguments) == 1:
    num_songs = 2
    note_count = 200
elif len(arguments) == 2:
    num_songs = int(arguments[1])
    note_count = 200
elif len(arguments) == 3:
    num_songs = int(arguments[1])
    note_count = int(arguments[2])
else:
    print("Invalid number of arguments")
    sys.exit(1)

# ------------------------ Change the paths as you need -------------------------- #
# Loading model archtecture and weights
model = load_model_from_json_h5('models/beeth_30.json', 'models/beeth_30.keras') 
# Loading Seed data
x_gen = np.load('beeth_gen.npy') 
file_path = 'data/beeth' # file path of the album
# -------------------------------------------------------------------------------- #

print("successfully loaded!")

# To get mappings
name = os.path.basename(file_path)
dataset = CreateDataset(file_path, name=name)
dataset.make_mappings()
generator = Generator(model, x_gen, dataset)
generator.create_playlist(num_songs=num_songs, note_count=note_count)
print(f"{num_songs} Songs Created!")
