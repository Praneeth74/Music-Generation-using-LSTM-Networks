### For training a model and more description of the process, go through the [Kaggle Notebook](https://www.kaggle.com/code/praneeth097/music-generation-using-lstm-networks)

## Generating a Playlist of Music using the LSTM model
The main file for generating music is the model_generate.py file. This file, by default, is set to generate Beethoven music using the model trained on that. It also uses a seed from the seed set (genxsets/beeth_gen.npy) as a starting point to generate a piece of music. So, if you want to generate music from a different album, you would have to train a new model on that album using the Kaggle notebook (preferably) to easily take advantage of GPU available there. After training, download the trained model (both .json and .h5 files) and seed set (.npy file) to further use them to generate music without training again. Optionally, you can use the model_generate.py file by replacing the file paths of the model and seed sets appropriately (clearly marked in the file).

Firstly, install the necessary libraries using the requirements.txt using pip -
```bash
pip install -r requirements.txt
```

To generate a playlist of music, you can use the following script:

```bash
python model_generate.py [num_songs] [note_count]
```
Replace [num_songs] with number of songs you want to generate (defaults to 2). <br>
Replace [note_count] with the number of notes or chords to generate in each song (defaults to 200). <br>

Example:
```bash
python model_generate.py 10 300
```
The above script generates 10 songs with 300 notes or chords in each.

An example of music generated by the model - 

<a href="https://github.com/Praneeth74/Music-Generation-using-LSTMs/blob/main/myMusic_0/0_beeth_generated_30_1000.midi" download>Download MIDI File 1</a>

### Data source:
Intermediate source of the data - [classical-music-midi](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi)

MIDI data original source - [Classical Piano MIDI Page](http://www.piano-midi.de/)

