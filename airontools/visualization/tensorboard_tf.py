from airontools.visualization.utils import save_insights
from airontools.net_constructors.utils_tf import get_latent_model


def get_insights(x, model, hidden_layer_names=None, **kwargs):
    model_ = model if not hidden_layer_names else get_latent_model(model, hidden_layer_names)
    save_insights(embeddings=model_.predict(x), **kwargs)


