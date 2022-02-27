import tensorflow.keras.backend as k_bcknd
from tensorflow.keras.models import model_from_json, Model, load_model


def save_model(model, name, save_entire_model=False, format='tf'):
    assert format in ['tf','h5']
    if format == 'tf':
        format = ''
    else:
        format = '.' + format
    model.save_weights(filepath=name + '_weights' + format)
    with open(name + '_topology', "w") as json_file:
        json_file.write(model.to_json())
    if save_entire_model:
        model.save(name + format)


def load_model(name, custom_objects=None, load_entire_model=False, format='tf'):
    assert format in ['tf', 'h5']
    if format == 'tf':
        format = ''
    else:
        format = '.' + format
        
    if load_entire_model:
        model = load_model(name + format)
    else:
        json_file = open(name + '_topology', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, custom_objects)
        model.load_weights(filepath=name + '_weights' + format)
    
    return model


def clear_session():
    k_bcknd.clear_session()


def summary(model):
    """ Model summary.

        Parameters:
            model (Model): Model to summarize.
    """
    print("\n")
    print("________________________ Model Summary __________________________")
    print("Main model name: " + model.name)
    print(model.summary())
    print("\n")
    print("_________________ Layers/Sub-Models Summaries ___________________")
    for layer in model.layers:
        print(layer.name)
        try:
            print(layer.summary())
        except:
            pass
    print("\n")
