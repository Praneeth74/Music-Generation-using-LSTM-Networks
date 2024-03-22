# Normal
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter

# utils
from utils.myutils import *

# os
import os
import pickle

# music21 
from music21 import *
from music21 import converter, instrument, note, chord

# tf imports
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

# music_gen_utils
class CombineMusic:
    def __init__(self, dirpath):
        """
        dirpath: Directory path to music albums
        """
        self.dirpath = dirpath
    
    @property
    def get_name_list(self):
        return os.listdir(self.dirpath)

    def combine_music(self, name_list=[]): 
        dest_dir = os.path.join(os.path.dirname(self.dirpath), "music_combo")
        try:
            os.mkdir(dest_dir)
        except FileExistsError:
            pass
        if name_list!=[]:
            pass
        else:
            name_list=self.get_name_list
        for i in name_list:
            source_dir = os.path.join(self.dirpath, f"{i}")
            for j in os.listdir(source_dir):
                source_path = os.path.join(source_dir, j)
                dest_path = os.path.join(dest_dir, j)
                print(source_path, dest_path)
                shutil.copy(source_path, dest_path)


class CreateDataset:
    """
        To preprocess midi data
    """
    def __init__(self, dirpath, length=30, name="", save_memory=False):
        """
            dirpath: Album's path
            length: Number of notes or chords to be considered in one sample
            name: Some name may be name of the album to create playlists names accordingly
            save_memory: If True deletes unecessary data for saving memory
        """
        self.dirpath = dirpath
        self.length = length
        self.name = name
        self.save_memory = save_memory
        self.notes_chords_list = []
        self.features = None
        self.labels = None
        self.features_normalized = None
        self.labels_encoded = None
        self.mapping = None 
        self.reverse_mapping = None
        self.labels_mapping = None
        self.labels_reverse_mapping = None

    def make_mappings(self):
        self.make_features_labels()
        all_notes = np.concatenate((self.features.ravel(), self.labels))
        unique_notes = np.unique(all_notes)
        sorted_notes = np.sort(unique_notes)
        mapping_numbers = np.arange(1, len(unique_notes)+1)
        norm_map_numbers = (mapping_numbers - min(mapping_numbers))/(max(mapping_numbers)-min(mapping_numbers))
        norm_map_numbers = norm_map_numbers + 0.01
        self.mapping = dict(zip(sorted_notes, norm_map_numbers))
        self.reverse_mapping = dict(zip(norm_map_numbers, sorted_notes))
        self.features_normalized = [self.mapping[note] for note in self.features.ravel()]
        self.features_normalized = np.array(self.features_normalized).reshape((-1, self.length)) 
        self.labels_encoded, self.labels_mapping, self.labels_reverse_mapping = self.make_ohe_labels()
        if self.save_memory:
            self.features = None
            self.labels = None

    def make_features_labels(self):
        file_name = f'{os.path.basename(self.dirpath)}_notes_chords.pkl'
        notes_chords_path = './notes_chords'
        if not os.path.exists(notes_chords_path):
            os.mkdir(notes_chords_path)
        if file_name in os.listdir(notes_chords_path):
            self.notes_chords_list = load_file_from_pickle(os.path.join(notes_chords_path, file_name))
        else:
            self.get_notes_and_chords()
            write_file_to_pickle(f'{notes_chords_path}/{file_name}', self.notes_chords_list)
        self.features, self.labels = self.make_dataset()

    def get_notes_and_chords(self):
        self.notes_chords_list = []
        for file_path in os.listdir(self.dirpath):
            # Load MIDI file
            midi_stream = converter.parse(os.path.join(self.dirpath, file_path))
            # Extract notes and chords from each part
            notes_and_chords = []
            for part in midi_stream.parts:
                for element in part.flat:
                    if isinstance(element, note.Note):
                        notes_and_chords.append(element.nameWithOctave)
                    elif isinstance(element, chord.Chord):
                        notes_and_chords.append(":".join([i.nameWithOctave for i in element.pitches]))
            self.notes_chords_list.append(notes_and_chords)         

    def make_dataset(self):
        features_list = []
        labels_list = []
        for melody in self.notes_chords_list:
            melody_data = []
            pred_data = []
            for i in range(len(melody)-self.length-1):
                melody_data.append(melody[i:i+self.length])
                pred_data.append(melody[i+self.length])
            features_list.extend(melody_data)
            labels_list.extend(pred_data)
        return np.array(features_list), np.array(labels_list)

    def make_ohe_labels(self):
        label_elements = np.unique(self.labels)
        labels_enco = np.arange(0, len(label_elements))
        labels_mapping = dict(zip(label_elements, labels_enco))
        labels_reverse_mapping = dict(zip(labels_enco, label_elements))
        labels_ohe =  []
        for i in self.labels:
            label_array = np.zeros((len(label_elements), ))
            label_array[labels_mapping[i]] = 1
            labels_ohe.append(label_array)
        return np.array(labels_ohe), labels_mapping, labels_reverse_mapping 

class Generator:
    def __init__(self, model, x_seed, dataset):
        self.model = model
        self.playlist = []
        self.counter = 0
        self.x_seed = x_seed
        self.dataset = dataset
        self.name = self.dataset.name
        self.repeat_patience = 2
    
    def create_playlist(self, num_songs=2, note_count=200, direc_name=f"myMusic", zip_file=False, rand=False):
        melodies = []
        if not rand:
            for i in range(num_songs):
                melodies.append((i, self.generate_melody(note_count)))
        else:
            for i in range(num_songs):
                melodies.append((i, self.generate_melody_rand(note_count)))
        dir_count = 0
        dir_exists = True
        while(dir_exists):
            try:
                os.mkdir(direc_name+f"_{dir_count}")
                dir_exists = False
            except FileExistsError:
                dir_count+=1
            
        direc_name = direc_name+f"_{dir_count}"
        for i, melody in melodies:
            melody.write('midi',f'{direc_name}/{i}_{self.dataset.name}_generated_{self.dataset.length}_{note_count}.midi')
        if zip_file:
            zip_dir(direc_name, f"{direc_name}Zip")

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
        seed_index = np.random.randint(0, len(self.x_seed) - 1)
        seed = self.x_seed[seed_index]
        generated_notes = []
        music = [self.dataset.reverse_mapping[i] for i in seed]
        music_norm = list(seed.ravel())
        prev_gen = seed
        i=0
        for _ in range(note_count):
            seed_reshaped = seed.reshape((1, self.dataset.length, 1))
            prediction = self.model.predict(seed_reshaped, verbose=0)
            index = np.argmax(prediction.ravel())
            music.append(self.dataset.labels_reverse_mapping[index])
            music_norm.append(self.dataset.mapping[self.dataset.labels_reverse_mapping[index]])
            seed = np.array(music_norm[-self.dataset.length:])
            if((i+1)%(self.dataset.length*self.repeat_patience)==0 and i>1):
                if(prev_gen == seed).all():
                    seed_index = np.random.randint(0, len(self.x_seed) - 1)
                    seed = self.x_seed[seed_index]
                prev_gen = seed
            i+=1
    
        melody = self.convert_to_midi(music)
        melody_midi = stream.Stream(melody)
        self.playlist.append((self.counter, melody_midi, self.dataset.length, note_count, self.dataset.name))                
        self.counter += 1;
        return melody_midi
    
    def generate_melody_rand(self, note_count=200):
        corpus = list(mapping.values())
        np.random.shuffle(corpus)
        seed = np.array(corpus[:self.dataset.length])
        generated_notes = []
        music = [self.dataset.reverse_mapping[i] for i in seed]
        music_norm = list(seed.ravel())
        generated_music = []
        prev_gen = seed
        i=0
        for _ in range(note_count):
            seed_reshaped = seed.reshape((1, self.dataset.length, 1))
            prediction = self.model.predict(seed_reshaped, verbose=0)
            index = np.argmax(prediction.ravel())
            music.append(self.dataset.labels_reverse_mapping[index])
            generated_music.append(music[-1])
            music_norm.append(self.dataset.mapping[self.dataset.labels_reverse_mapping[index]])
            seed = np.array(music_norm[-self.dataset.length:])
            if((i+1)%(self.dataset.length*self.repeat_patience)==0 and i>1):
                if(prev_gen == seed).all():
                    print("entered")
                    np.random.shuffle(corpus)
                    seed = np.array(corpus[:self.dataset.length])
                prev_gen = seed
            i+=1
    
        melody = self.convert_to_midi(generated_music)
        melody_midi = stream.Stream(melody)
        self.playlist.append((self.counter, melody_midi, self.dataset.length, note_count, self.dataset.name))                
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
    
