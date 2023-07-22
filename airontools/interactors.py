import tensorflow as tf

from airontools.constructors.layers import CustomDropout


def save_model(
    model: tf.keras.models.Model,
    filepath: str,
    save_entire_model: bool = False,
    file_format: str = "tf",
) -> None:
    """Save a Keras model.

    Parameters:
        model (tf.keras.models.Model): Keras model to save.
        filepath (str): File path to save the model.
        save_entire_model (bool): Whether to save the entire model as a whole.
        file_format (str): The format of the file, which can be either tf or h5.

    """
    # ToDo: challenge whether this function is still relevant.
    assert file_format in ["tf", "h5"]
    if file_format == "tf":
        file_format = ""
    else:
        file_format = "." + file_format
    model.save_weights(filepath=filepath + "_weights" + file_format)
    with open(filepath + "_topology", "w") as json_file:
        json_file.write(model.to_json())
    if save_entire_model:
        model.save(filepath + file_format)


def load_model(
    filepath: str,
    custom_objects: dict = None,
    load_entire_model: bool = False,
    file_format: str = "tf",
) -> tf.keras.models.Model:
    """Load a Keras model.

    Parameters:
        filepath (str): File path to save the model.
        custom_objects (dict): Custom objects, such CustomDropout, to be loaded together with the model.
        load_entire_model (bool): Whether to load the entire model as a whole.
        file_format (str): The format of the file, which can be either tf or h5.

    Returns:
        x (tf.keras.models.Model): A keras model.

    """
    # ToDo: make the addition of custom objects more general
    if custom_objects is None:
        custom_objects = {"CustomDropout": CustomDropout}
    elif (
        isinstance(custom_objects, dict)
        and "CustomDropout" not in custom_objects.keys()
    ):
        custom_objects.update({"CustomDropout": CustomDropout})
    assert file_format in ["tf", "h5"]
    if file_format == "tf":
        file_format = ""
    else:
        file_format = "." + file_format
    if load_entire_model:
        model = tf.keras.models.load_model(filepath + file_format)
    else:
        json_file = open(filepath + "_topology")
        loaded_model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(loaded_model_json, custom_objects)
        model.load_weights(filepath=filepath + "_weights" + file_format)

    return model


def summary(model) -> None:
    """Model summary.

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
