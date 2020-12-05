import numpy as np
from cave.utils import customized_net
from hyperopt import Trials, STATUS_OK, STATUS_FAIL
import hyperopt
from tensorflow.keras import callbacks
from tensorflow.keras import models
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import pandas as pd
import pickle
import math
import os
import glob


class DeepNet(object):

    def __init__(self):

        self.__model = None
        self.__parallel_models = None
        self.__device = None

    def create(self, specs, metrics=None, net_name='NN'):
        self.__model = customized_net(specs=specs, metrics=metrics, net_name=net_name)

    def explore(self, x_train, y_train, x_val, y_val, space, model_specs, experiment_specs, path, max_evals,
                tensor_board=False, metric=None, trials=None, net_name='NN', verbose=0, seed=None):

        self.__parallel_models = model_specs['parallel_models']
        self.__device = model_specs['device']
        if trials is None:
            trials = Trials()

        def objective(space):

            # Create model
            specs = space.copy()
            specs.update(model_specs)
            specs.update(experiment_specs)
            model = customized_net(specs=specs, net_name=net_name, metrics=metric)

            # Print some information
            iteration = len(trials.losses())
            if verbose > 0:
                print('\n')
                print('iteration : {}'.format(0 if trials.losses() is None else iteration))
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
                         tensor_board=tensor_board,
                         batch_size=specs['batch_size'],
                         ext=iteration)

            # Exploration loss
            total_n_models = self.__parallel_models * len(self.__device)
            exp_loss = None
            if metric in [None, 'categorical_accuracy']:
                exp_loss = model.evaluate(x=x_val, y=y_val)[1:]
                if isinstance(exp_loss, list):
                    exp_loss = sum(exp_loss)
                exp_loss /= total_n_models
                exp_loss = 1 - exp_loss
            elif metric == 'i_auc':
                y_pred = model.predict(x_val)
                if not isinstance(y_pred, list):
                    y_pred = [y_pred]
                exp_loss = []
                for i in np.arange(0, total_n_models):
                    if len(np.bincount(y_val[i][:,-1])) > 1 and not math.isnan(np.sum(y_pred[i])):
                        fpr, tpr, thresholds = metrics.roc_curve(y_val[i][:, -1], y_pred[i][:, -1])
                        exp_loss += [(1 - metrics.auc(fpr, tpr))]
                exp_loss = np.mean(exp_loss) if len(exp_loss) > 0 else 1
            if verbose > 0:
                print('\n')
                print('Exploration Loss: ', exp_loss)
            status = STATUS_OK if not math.isnan(exp_loss) and exp_loss is not None else STATUS_FAIL

            # Save trials
            with open(path + 'trials.hyperopt', 'wb') as f:
                pickle.dump(trials, f)

            # Save model if it is the best so far
            best_exp_losss_name = path + 'best_' + net_name + '_exp_loss'
            best_exp_loss = None \
                if not os.path.isfile(best_exp_losss_name) else pd.read_pickle(best_exp_losss_name).values[0][0]
            if status == STATUS_OK and (best_exp_loss is None or exp_loss < best_exp_loss):
                df = pd.DataFrame(data=[exp_loss], columns=['best_exp_loss'])
                df.to_pickle(best_exp_losss_name)
                self.__save_json(filepath=path + 'best_exp_' + net_name + '_json', model=model)
                self.__save_weights(filepath=path + 'best_exp_' + net_name + '_weights', model=model)
                for dict_, name in zip([specs, space], ['_specs', '_hparams']):
                    with open(path + 'best_exp_' + net_name + name, 'wb') as f:
                        pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)

            K.clear_session()
            del model

            return {'loss': exp_loss, 'status': status}

        def optimize():

            if len(trials.trials) < max_evals:
                hyperopt.fmin(
                    objective,
                    rstate=None if seed is None else np.random.RandomState(seed),
                    space=space,
                    algo=hyperopt.tpe.suggest,
                    max_evals=max_evals,
                    trials=trials,
                    verbose=True,
                    return_argmin=False)
            with open(path + 'best_exp_' + net_name + '_hparams', 'rb') as f:
                best_hparams = pickle.load(f)
            with open(path + 'best_exp_' + net_name + '_specs', 'rb') as f:
                specs = pickle.load(f)
            best_model = self.__load_json(filepath=path + 'best_exp_' + net_name + '_json')
            best_model.load_weights(filepath=path + 'best_exp_' + net_name + '_weights')
            best_model.compile(optimizer=Adam(learning_rate=specs['lr']), loss=specs['loss'])

            print('best hyperparameters: ' + str(best_hparams))

            return best_model

        self.__model = optimize()

    def __train(self, x_train, y_train, x_val, y_val, model, experiment_specs, mode, path, use_callbacks,
                verbose, tensor_board, batch_size, ext=None):

        best_model_name = path + 'best_epoch_model_' + mode

        # Callbacks
        callbacks_list = []
        if use_callbacks:
            if tensor_board:
                board_dir = path + mode + '_logs'
                if ext is not None:
                    board_dir += '_' + str(ext)
                callbacks_list += [callbacks.TensorBoard(log_dir=board_dir)]
            callbacks_list += [callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=int(experiment_specs['early_stopping'] / 2),
                min_lr=0.0000001,
                verbose=verbose)]
            callbacks_list += [callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=experiment_specs['early_stopping'],
                verbose=verbose,
                mode='min')]
            callbacks_list += [callbacks.ModelCheckpoint(
                filepath=best_model_name,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose)]
            best_model_files = glob.glob(best_model_name + '*')
            if len(best_model_files) > 0:
                for filename in glob.glob(best_model_name + '*'):
                    os.remove(filename)

        # Train model
        class_weight = None if 'class_weight' not in experiment_specs.keys() \
            else {output_name: experiment_specs['class_weight'] for output_name in model.output_names}
        kargs = {'x': x_train,
                 'y': y_train,
                 'epochs': experiment_specs['epochs'],
                 'callbacks': callbacks_list,
                 'class_weight': class_weight,
                 'shuffle': True,
                 'verbose': verbose,
                 'batch_size': batch_size}
        if not any([val_ is None for val_ in [x_val, y_val]]):
            kargs.update({'validation_data': (x_val, y_val)})
        model.fit(**kargs)

        # Best model
        if use_callbacks:
            best_model_files = glob.glob(best_model_name + '*')
            if len(best_model_files) > 0:
                model.load_weights(filepath=best_model_name)
                for filename in glob.glob(best_model_name + '*'):
                    os.remove(filename)


    def train(self, x_train, y_train, experiment_specs, use_callbacks, batch_size=30, x_val=None, y_val=None, path=None,
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
            tensor_board=tensor_board,
            batch_size=batch_size)

    def inference(self, x_pred):
        return self.__model.predict(x_pred)

    def evaluate(self, x, y):
        return self.__model.evaluate(x=x, y=y)

    def save_weights(self, filepath):
        self.__save_weights(filepath=filepath, model=self.__model)

    def __save_weights(self, filepath, model):
        model.save_weights(filepath=filepath)

    def load_weights(self, filepath):
        self.__load_weights(filepath=filepath, model=self.__model)

    def __load_weights(self, filepath, model):
        model.load_weights(filepath=filepath)

    def get_weights(self):
        return self.__model.get_weights()

    def set_weights(self, weights):
        self.__set_weights(weights=weights, model=self.__model)

    def __set_weights(self, weights, model):
        model.set_weights(weights=weights)

    def save_json(self, filepath):
        self.__save_json(filepath=filepath, model=self.__model)

    def __save_json(self, filepath, model):
        with open(filepath, "w") as json_file:
            json_file.write(model.to_json())

    def load_json(self, filepath):
        self.__model = self.__load_json(filepath)

    def __load_json(self, filepath):
        json_file = open(filepath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        return models.model_from_json(loaded_model_json)

    def clear_session(self):
        K.clear_session()

    def compile(self, loss, metrics=None, lr=0.001):
        self.__model.compile(optimizer=Adam(learning_rate=lr),
                             loss=loss,
                             metrics=metrics)
