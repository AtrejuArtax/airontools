import abc
from typing import Dict, List, Union

import tensorflow as tf
from numpy.typing import NDArray


class Model(abc.ABC):
    @abc.abstractmethod
    def compile(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, tf.Tensor]:
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> Union[List[NDArray[float]], NDArray[float]]:
        pass

    @abc.abstractmethod
    def save_weights(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def load_weights(self, *args, **kwargs) -> None:
        pass
