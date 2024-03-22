## Generating a Playlist of Music

Firstly, we need to install the necessary libraries using the requirements.txt using pip -
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
python model_generate.py 10 100
```
The above script generates 10 songs with 300 notes or chords in each.
