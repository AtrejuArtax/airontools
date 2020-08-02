import numpy as np
from cave.utils import customized_net
from hyperopt import Trials, STATUS_OK
import hyperopt
from tensorflow.keras import callbacks
from tensorflow.keras import models
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import pickle


class DeepNet(object):

    def __init__(self):

        self.__model = None
        self.__parallel_models = None
        self.__device = None

    def create(self, specs, metrics=None, net_name='NN'):
        self.__model = customized_net(specs=specs, metrics=metrics, net_name=net_name)

    def explore(self, x_train, y_train, x_val, y_val, space, model_specs, experiment_specs, path, max_evals,
                tensor_board=False, metrics=None, trials=None, net_name='NN', verbose=0, seed=None):

        self.__parallel_models = model_specs['parallel_models']
        self.__device = model_specs['device']
        if trials is None:
            trials = Trials()

        def objective(space):

            # Create model
            specs = space.copy()
            specs.update(model_specs)
            specs.update(experiment_specs)
            model = customized_net(specs=specs, metrics=metrics, net_name=net_name)

            # Print some information
            if verbose > 0:
                print('\n')
                print('iteration : {}'.format(0 if trials.losses() is None else len(trials.losses())))
                [print('{}: {}'.format(key, value)) for key, value in specs.items()]
                print(model.summary(line_length=200))

            # Train model
            self.__train(x_train=x_train,
                         y_train=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         model=model,
                         experiment_specs=experiment_specs,
                         mode='exploration',
                         path=path,
                         use_callbacks=True,
                         verbose=verbose,
                         tensor_board=tensor_board)

            # Train and validation losses
            val_loss = model.evaluate(x=x_val, y=y_val)
            if isinstance(val_loss, list):
                val_loss = sum(val_loss)
            val_loss /= self.__parallel_models * len(self.__device)

            if verbose > 0:
                print('\n')
                print('Validation Loss:', val_loss)

            # Save trials
            with open(path + 'trials.hyperopt', 'wb') as f:
                pickle.dump(trials, f)

            return {'loss': val_loss, 'status': STATUS_OK, 'model': model}

        def optimize():

            best_param = hyperopt.fmin(
                objective,
                rstate=None if seed is None else np.random.RandomState(seed),
                space=space,
                algo=hyperopt.tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                verbose=True)
            if trials.losses() is not None:
                lowest_loss_ind = np.argmin(trials.losses())
                best_model = trials.trials[lowest_loss_ind]['result']['model']
            else:
                best_model = None

            print('best hyperparameters: ' + str(best_param))

            return best_model

        self.__model = optimize()

    def __train(self, x_train, y_train, x_val, y_val, model, experiment_specs, mode, path, use_callbacks,
                verbose, tensor_board):

        # Callbacks
        callbacks_list = []
        if use_callbacks:
            if tensor_board:
                callbacks_list += [callbacks.TensorBoard(log_dir=path + mode + '_logs')]
            callbacks_list += [callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=0,
                min_lr=0.0000001,
                verbose=verbose)]
            callbacks_list += [callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=experiment_specs['early_stopping'],
                verbose=verbose,
                mode='min')]
            callbacks_list += [callbacks.ModelCheckpoint(
                filepath=path + 'best_epoch_model_' + mode,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose)]

        # Train model
        class_weight = None if 'class_weight' not in experiment_specs.keys() \
            else {output_name: experiment_specs['class_weight'] for output_name in model.output_names}
        kargs = {'x': x_train,
                 'y': y_train,
                 'epochs': experiment_specs['epochs'],
                 'callbacks': callbacks_list,
                 'class_weight': class_weight,
                 'shuffle': True,
                 'verbose': verbose}
        if not any([val_ is None for val_ in [x_val, y_val]]):
            kargs.update({'validation_data': (x_val, y_val)})
        model.fit(**kargs)

        # Best model
        if use_callbacks:
            model.load_weights(filepath=path + 'best_epoch_model_' + mode)

    def train(self, x_train, y_train, experiment_specs, use_callbacks, x_val=None, y_val=None, path=None,
              verbose=0, tensor_board=False):

        # Train model
        self.__train(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            model=self.__model,
            experiment_specs=experiment_specs,
            mode='training',
            path=path,
            use_callbacks=use_callbacks,
            verbose=verbose,
            tensor_board=tensor_board)

    def inference(self, x_pred):
        return self.__model.predict(x_pred)

    def evaluate(self, x, y):
        return self.__model.evaluate(x=x, y=y)

    def save_weights(self, filepath):
        self.__model.save_weights(filepath=filepath)

    def load_weights(self, filepath):
        self.__model.load_weights(filepath=filepath)

    def get_weights(self):
        return self.__model.get_weights()

    def set_weights(self, weights):
        self.__model.set_weights(weights=weights)

    def save_json(self, filepath):
        model_json = self.__model.to_json()
        with open(filepath, "w") as json_file:
            json_file.write(model_json)

    def load_json(self, filepath):
        json_file = open(filepath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.__model = models.model_from_json(loaded_model_json)

    def clear_session(self):
        K.clear_session()

    def compile(self, loss, metrics=None, lr=0.001):
        self.__model.compile(optimizer=Adam(learning_rate=lr),
                             loss=loss,
                             metrics=metrics)
