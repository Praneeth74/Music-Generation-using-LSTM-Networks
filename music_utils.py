import subprocess
pip_install = ['pip', 'install']
subprocess.run(pip_install+['music21'])
subprocess.run(pip_install + ['--upgrade', 'pip'])

# Normal
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter

# os
import os
import pickle

# music21 
from music21 import *
from music21 import converter, instrument, note, chord

# tf imports
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

# music_gen_utils
def get_notes_and_chords(file):
    notes_chords_list = []
    for file_path in os.listdir(file):
        # Load MIDI file
        midi_stream = converter.parse(file+file_path)
        # Extract notes and chords from each part
        notes_and_chords = []
        for part in midi_stream.parts:
            for element in part.flat:
                if isinstance(element, note.Note):
                    notes_and_chords.append(element.nameWithOctave)
                elif isinstance(element, chord.Chord):
                    notes_and_chords.append(":".join([i.nameWithOctave for i in element.pitches]))
        notes_chords_list.append(notes_and_chords)         
    return notes_chords_list

def make_dataset(melodies_list, length=10):
    features_list = []
    labels_list = []
    for melody in melodies_list:
        melody_data = []
        pred_data = []
        for i in range(len(melody)-length-1):
            melody_data.append(melody[i:i+length])
            pred_data.append(melody[i+length])
        features_list.extend(melody_data)
        labels_list.extend(pred_data)
    return np.array(features_list), np.array(labels_list)

def ohe_labels(labels):
    label_elements = np.unique(labels)
    labels_enco = np.arange(0, len(label_elements))
    labels_mapping = dict(zip(label_elements, labels_enco))
    labels_reverse_mapping = dict(zip(labels_enco, label_elements))
    labels_ohe =  []
    for i in labels:
        label_array = np.zeros((len(label_elements), ))
        label_array[labels_mapping[i]] = 1
        labels_ohe.append(label_array)
    return np.array(labels_ohe), labels_mapping, labels_reverse_mapping 

class Generator:
    def __init__(self, model, x_seed, mapping={}, reverse_mapping={}, labels_reverse_mapping={}, name=""):
        self.model = model
        self.playlist = []
        self.counter = 0
        self.x_seed = x_seed
        self.length = x_seed.shape[-1]
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping
        self.labels_reverse_mapping = labels_reverse_mapping
        self.name = name
        self.repeat_patience = 2
    
    def convert_to_midi(self, notes_list):
        melody_stream = []
        time_offset = 0
        for item in notes_list:
            if (":" in item or item.isdigit()):
                chord_notes = item.split(":")
                note_objects = [note.Note(n) for n in chord_notes]
                chord_obj = chord.Chord(note_objects)
                chord_obj.offset = time_offset
                melody_stream.append(chord_obj)
            else: 
                single_note = note.Note(item)
                single_note.offset = time_offset
                melody_stream.append(single_note)
            time_offset += 0.5
            
        melody_stream_midi = stream.Stream(melody_stream)
        return melody_stream_midi
    
    def generate_melody(self, note_count=200):
        seed_index = np.random.randint(0, len(x_seed) - 1)
        seed = self.x_seed[seed_index]
        generated_notes = []
        music = [self.reverse_mapping[i] for i in seed]
        music_norm = list(seed.ravel())
        prev_gen = seed
        i=0
        for _ in range(note_count):
            seed_reshaped = seed.reshape((1, length, 1))
            prediction = self.model.predict(seed_reshaped, verbose=0)
            index = np.argmax(prediction.ravel())
            music.append(self.labels_reverse_mapping[index])
            music_norm.append(self.mapping[self.labels_reverse_mapping[index]])
            seed = np.array(music_norm[-length:])
            if((i+1)%(self.length*self.repeat_patience)==0 and i>1):
                if(prev_gen == seed).all():
                    print("entered")
                    seed_index = np.random.randint(0, len(x_seed) - 1)
                    seed = x_seed[seed_index]
                prev_gen = seed
            i+=1
    
        melody = self.convert_to_midi(music)
        melody_midi = stream.Stream(melody)
        self.playlist.append((self.counter, melody_midi, self.length, note_count, self.name))                
        self.counter += 1;
        return melody_midi
    
    def generate_melody_rand(self, note_count=200):
        corpus = list(mapping.values())
        np.random.shuffle(corpus)
        seed = np.array(corpus[:self.length])
        generated_notes = []
        music = [self.reverse_mapping[i] for i in seed]
        music_norm = list(seed.ravel())
        generated_music = []
        prev_gen = seed
        i=0
        for _ in range(note_count):
            seed_reshaped = seed.reshape((1, length, 1))
            prediction = self.model.predict(seed_reshaped, verbose=0)
            index = np.argmax(prediction.ravel())
            music.append(self.labels_reverse_mapping[index])
            generated_music.append(music[-1])
            music_norm.append(self.mapping[self.labels_reverse_mapping[index]])
            seed = np.array(music_norm[-length:])
            if((i+1)%(self.length*self.repeat_patience)==0 and i>1):
                if(prev_gen == seed).all():
                    print("entered")
                    np.random.shuffle(corpus)
                    seed = np.array(corpus[:self.length])
                prev_gen = seed
            i+=1
    
        melody = self.convert_to_midi(generated_music)
        melody_midi = stream.Stream(melody)
        self.playlist.append((self.counter, melody_midi, self.length, note_count, self.name))                
        self.counter += 1;
        return melody_midi
    
    def save_melodies(self, direc_name = 'myPlaylist'):
        try:
            os.mkdir(direc_name)
        except FileExistsError:
            pass
        if(self.playlist==[]):
            print("The playlist is empty!")
            return
        for counter, melody, length, note_count, name in self.playlist[:self.counter]:
            melody.write('midi',f'{direc_name}/{counter}_{name}_generated_{length}_{note_count}.midi')
    
    def create_playlist(self, num_songs=2, note_count=200, direc_name=f"myMusic", zip_file=False, rand=False):
        melodies = []
        if not rand:
            for i in range(num_songs):
                melodies.append((i, self.generate_melody(note_count)))
        else:
            for i in range(num_songs):
                melodies.append((i, self.generate_melody_rand(note_count)))
        try:
            os.mkdir(direc_name)
        except:
            pass
        for i, melody in melodies:
            melody.write('midi',f'{direc_name}/{i}_{self.name}_generated_{self.length}_{note_count}.midi')
        if zip:
            zip_dir(direc_name, f"{direc_name}Zip")
        
