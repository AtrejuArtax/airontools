from tensorflow.keras.models import Model
from airontools.visualization.utils_tf import save_insights
from airontools.constructors.utils.utils_tf import get_latent_model


def get_insights(x, model, hidden_layer_names=None, **kwargs):
    """ Get insights of latent layers. Given input data and a model, this function makes use of tensorboard to get
    insights of the latent representations.

        Parameters:
            x (list, array): Data to be mapped to latent representations.
            model (Model): Model to be used to get the insights.
            hidden_layer_names (str, list): Names of the hidden layers ti get insights from.
            embeddings (list, array): Embeddings to be saved.
            embeddings_names (list, str): Embeddings names.
            metadata (list, array): Metadata.
            path (str): Path to save insights.
    """
    model_ = model if not hidden_layer_names else get_latent_model(model, hidden_layer_names)
    save_insights(model_.predict(x), **kwargs)
