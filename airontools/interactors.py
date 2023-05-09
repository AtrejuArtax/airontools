import tensorflow as tf

from airontools.constructors.layers import CustomDropout


def load_model(
    name: str,
    custom_objects: dict = None,
    load_entire_model: bool = False,
    file_format: str = "tf",
) -> tf.keras.models.Model:
    """Load model."""
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
        model = tf.keras.models.load_model(name + file_format)
    else:
        json_file = open(name + "_topology")
        loaded_model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(loaded_model_json, custom_objects)
        model.load_weights(filepath=name + "_weights" + file_format)

    return model


def clear_session() -> None:
    """Clear session."""
    tf.keras.backend.clear_session()


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
