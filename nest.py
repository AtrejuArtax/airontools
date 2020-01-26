import numpy as np
from utils import customized_net
from hyperopt import Trials, STATUS_OK
import hyperopt
from tensorflow.keras import callbacks
from tensorflow.keras import models


__author__ = 'claudi'


class DeepNet(object):

    def __init__(self):

        self.__model = None
        self.__parallel_models = None
        self.__device = None

    def create(self, specs, metrics=None):
        specs['units'] = [int(units) for units in
                          np.linspace(specs['n_input'], specs['n_output'], specs['n_layers'] + 1)]
        self.__model = customized_net(
            specs=specs,
            metrics=metrics,
            net_name='NN')

    def explore(self, x_train, y_train, x_val, y_val, space, model_specs, experiment_specs, path, max_evals,
                print_model=False, tensor_board=False, metrics=None):

        self.__parallel_models = model_specs['parallel_models']
        self.__device = model_specs['device']

        def objective(space):

            # Create model
            specs = space.copy()
            specs.update(model_specs)
            specs.update(experiment_specs)
            if 'n_layers' in space.keys():
                n_layers = space['n_layers']
            else:
                n_layers = specs['n_layers']
            specs['units'] = [int(units) for units in
                              np.linspace(specs['n_input'], specs['n_output'],
                                          n_layers + 1)]
            model = customized_net(
                specs=specs,
                metrics=metrics,
                net_name='NN')

            # Print model
            if print_model:
                print('\n')
                print('\n')
                for k,v in specs.items():
                    print(k + ' ' + str(v))
                print(model.summary())

            # Train model
            model = self.__train(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                model=model,
                experiment_specs=experiment_specs,
                mode='exploration',
                path=path,
                tensor_board=tensor_board)

            # Train and validation losses
            val_loss = model.evaluate(
                    x=x_val,
                    y=y_val)[0]\
                       / self.__parallel_models * len(self.__device)

            print('\n')
            print('Validation Loss:', val_loss)

            return {'loss': val_loss, 'status': STATUS_OK, 'model': model}

        def optimize():

            trials = Trials()
            best_param = hyperopt.fmin(
                objective,
                space=space,
                algo=hyperopt.tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                verbose=1)
            lowest_loss_ind = np.argmin(trials.losses())
            best_model = trials.trials[lowest_loss_ind]['result']['model']

            print('best hyperparameters: ' + str(best_param))

            return best_model

        self.__model = optimize()

    def __train(self, x_train, y_train, x_val, y_val, model, experiment_specs, mode, path, tensor_board=False):

        # Callbacks
        callbacks_list = []
        if tensor_board:
            callbacks_list += [callbacks.TensorBoard(log_dir=path + mode + '_logs')]
        callbacks_list += [callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=0,
            min_lr=0.000000001,
            verbose=0)]
        callbacks_list += [callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=experiment_specs['early_stopping'],
            verbose=0,
            mode='min')]
        callbacks_list += [callbacks.ModelCheckpoint(
            filepath=path + 'best_epoch_model_' + mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=0)]

        # Train model
        model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val,
                             y_val),
            epochs=experiment_specs['epochs'],
            verbose=0,
            callbacks=callbacks_list,
            class_weight={output_name: experiment_specs['class_weight'] for output_name in model.output_names},
            shuffle=True)

        # Best model
        model.load_weights(filepath=path + 'best_epoch_model_' + mode)

        return model

    def train(self, x_train, y_train, x_val, y_val, experiment_specs, path, tensor_board=False):

        # Train model
        self.__model = self.__train(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            model=self.__model,
            experiment_specs=experiment_specs,
            mode='training',
            path=path,
            tensor_board=tensor_board)

    def predict(self, x_pred):

        # Predict
        predictions = self.__model.predict(x_pred)

        return predictions

    def evaluate(self, x, y):

        # Predict
        eval = self.__model.evaluate(
            x=x,
            y=y)

        return eval

    def save_weights(self, filepath):

        # Save weights
        self.__model.save_weights(filepath=filepath)

    def load_weights(self, filepath):

        # Load weights
        self.__model.load_weights(filepath=filepath)

    def get_weights(self):

        # Get weights
        return self.__model.get_weights()


    def set_weights(self, weights):

        # Set weights
        self.__model.set_weights(weights=weights)

    def save_json(self, filepath):

        # Save json
        model_json = self.__model.to_json()
        with open(filepath, "w") as json_file:
            json_file.write(model_json)

    def load_json(self, filepath):

        # Load json
        json_file = open(filepath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.__model = models.model_from_json(loaded_model_json)
