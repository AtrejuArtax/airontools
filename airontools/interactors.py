import tensorflow.keras.backend as k_bcknd
from tensorflow.keras.models import load_model as __load_model
from tensorflow.keras.models import model_from_json, Model

from airontools.constructors.layers import CustomDropout


def save_model(model, name, save_entire_model=False, file_format='tf'):
    assert file_format in ['tf', 'h5']
    if file_format == 'tf':
        file_format = ''
    else:
        file_format = '.' + file_format
    model.save_weights(filepath=name + '_weights' + file_format)
    with open(name + '_topology', "w") as json_file:
        json_file.write(model.to_json())
    if save_entire_model:
        model.save(name + file_format)


def load_model(name, custom_objects=None, load_entire_model=False, file_format='tf'):
    # ToDo: make the addition of custom objects more general
    if custom_objects is None:
        custom_objects = {'CustomDropout': CustomDropout}
    elif isinstance(custom_objects, dict) and 'CustomDropout' not in custom_objects.keys():
        custom_objects.update({'CustomDropout': CustomDropout})
    assert file_format in ['tf', 'h5']
    if file_format == 'tf':
        file_format = ''
    else:
        file_format = '.' + file_format
    if load_entire_model:
        model = __load_model(name + file_format)
    else:
        json_file = open(name + '_topology', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, custom_objects)
        model.load_weights(filepath=name + '_weights' + file_format)
    
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
