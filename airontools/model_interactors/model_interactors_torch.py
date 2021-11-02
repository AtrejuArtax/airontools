import torch


def save_model(model, name):
    torch.save(model, name)


def load_model(name, custom_objects=None):   # ToDo: torch implementation of load_model with custom_objects.
    model = torch.load(name)
    model.eval()
    return model


def clear_session():   # ToDo: torch implementation of clear_session.
    pass


def summary():   # ToDo: torch implementation of summary.
    pass
