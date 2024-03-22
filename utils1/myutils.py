import os
import shutil
import pickle


# To delete all files in a directory
def del_files(dir_path):
    """
    dir_path: path of the directory whose files need to be deleted
    """
    if os.path.isdir(dir_path):   
        dir_list = os.listdir(dir_path)
    else:
        os.remove(dir_path)
    for file in dir_list:
        file_path = os.path.join(dir_path, file)
        if os.path.isdir(file_path):
            shutil.rmtree(file)
        else:
            os.remove(file)

# To zip a directory
def zip_dir(from_file_path, to_file_path):
    """
    from_file_path: directory to zip
    to_file_path: name of the zipped file *without* .zip extension
    """
    shutil.make_archive(to_file_path, 'zip', from_file_path)

# Load from a pickle file
def load_file_from_pickle(path1):
    """
    path1: path to the pickle file to load
    """
    with open(path1, 'rb') as file1:
        entity1 = pickle.load(file1)
    return entity1

# Write to a pickle file
def write_file_to_pickle(path1, entity1):
    """
    path1: path to the pickle file to write
    """
    with open(path1, 'wb') as file1:
        pickle.dump(entity1, file1)

