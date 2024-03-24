import tensorflow as tf

# tf saving and loading functions-start

def save_model_as_json(filepath, model):
    model_json = model.to_json()
    with open(filepath, 'w') as file1:
        file1.write(model_json)
        
def save_model_as_h5(filepath, model):
    model.save(filepath)
    
def save_model_as_json_h5(dirpath, filename, model):
    """
        dirpath: the path of the diretcory where the model need to be stored
        filename: name of the file without extension
        model: model to save
    """
    json_file_path = os.path.join(dirpath, filename+".json")
    h5_file_path = os.path.join(dirpath, filename+".keras")
    save_model_as_json(json_file_path, model)
    save_model_as_h5(h5_file_path, model)

def load_model_from_json(filepath):
    with open(filepath, 'r') as file1:
        model_json = file1.read()
    model = tf.keras.models.model_from_json(model_json)
    return model
        
def load_model_from_h5(filepath, model):
    model.load_weights(filepath)

def load_model_from_json_h5(json_file_path, h5_file_path):
    """
        json_file_path: model's json file path
        h5_file_path: model's h5 file path
    """
    model = load_model_from_json(json_file_path)
    load_model_from_h5(h5_file_path, model)
    return model

# tf saving and loading functions-end
