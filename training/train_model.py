import pandas as pd
import numpy as np
import random as rn
import tensorflow as tf
import json
import os
import argparse
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
from keras.callbacks import CSVLogger, Callback
from keras.utils import multi_gpu_model
from keras import losses, optimizers
from sklearn.metrics import mean_squared_error, matthews_corrcoef, roc_auc_score, hamming_loss, cohen_kappa_score, \
    balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler


def create_model(inp, hid, drop_in, drop_hid, activation, layers, initializer):
    """Creates a neural network to use for training, based on parameters specified

    Args:
    inp: number of input neurons
    hid: number of hidden neurons
    lr: learning rate (if required by loss)
    drop_in: droput for input layer
    drop_hid: dropout between hidden layers
    activation: activation function of Dense layers
    layers: number of hidden layers

    Returns:
    keras model: a trainable keras model
    """
    model = Sequential()
    model.add(Dense(hid, input_shape=(int(inp),), activation=activation, kernel_initializer=initializer, bias_initializer=initializer))
    model.add(Dropout(drop_in))
    for i in range(layers):
        model.add(Dense(hid, activation=activation, bias_initializer=initializer, kernel_initializer=initializer))
        model.add(Dropout(drop_hid))
    model.add(Dense(1, activation="sigmoid"))
    return model

def load_data(n,path, col = 'Fold', ):
    """loads data according to a column called Trainingset_n where it is specified
    which compound belongs to training and testset

    Args:
    n: number of Trainingsset
    path: path to data

    Returns:
    data: returns a training and test data as specified in the column
    """
    data = pd.read_csv(path)
    data = data.set_index("Reference")
    outer, inner = [int(i) for i in str(n)] #split in inner and outer loop no

    if inner > 0:
        fold_indices = [1, 2, 3, 4, 5]
        fold_indices.remove(outer)
        inner = fold_indices[inner-1]  # get fold which will be used as test set for the crossvalidation
        data = data[data[col] != outer]  # drop external test set
        data = data.dropna()
        train = data.loc[data[col] != inner].select_dtypes(include=['number'])
        test = data.loc[data[col] == inner].select_dtypes(include=['number'])

        train = train.drop(['Clusters', 'Fold'], axis=1)
        test = test.drop(['Clusters', 'Fold'], axis=1)

        return train, test
    else:
        data = data.dropna()

        train = data.loc[data[col] != outer].select_dtypes(include=['number'])
        test = data.loc[data[col] == outer].select_dtypes(include=['number'])

        train = train.drop(['Clusters', 'Fold'], axis=1)
        test = test.drop(['Clusters', 'Fold'], axis=1)

        return train, test


def build_masked_loss(loss_function, mask_value):
    """Builds a loss function that masks based on targets
    taken from: https://github.com/bioinf-jku/FCD/blob/master/fcd/FCD.py

    Args:
    loss_function: The loss function to mask
    mask_value: The value to mask in the targets

    Returns:
    function: a loss function that acts like loss_function with
    masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

        return loss_function(y_true * mask, y_pred * mask)


    return masked_loss_function


def evaluate(y, pred):
    """Evaluates  the performance of a model

    Args:
    y: true values (as pd.Series)
    y_pred: predicted values of the model (as np.array)

    Returns:
    dictionary: dictionary with all calculated values
    """
    y = np.asarray(y.to_frame())
    classes = np.greater(pred, 0.5).astype(int)
    tp = np.count_nonzero(classes * y)
    tn = np.count_nonzero((classes - 1) * (y - 1))
    fp = np.count_nonzero(classes * (y - 1))
    fn = np.count_nonzero((classes - 1) * y)

    # Calculate accuracy, precision, recall and F1 score.
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = np.NaN
    try:
        sensitivity = tp / (tp + fn)
    except ZeroDivisionError:
        sensitivity = np.NaN
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        specificity = np.NaN

    bal_acc = (sensitivity + specificity) / 2
    try:
        fpr = fp / (fp + tn)
    except ZeroDivisionError:
        fpr = np.NaN
    try:
        fnr = fn / (tp + fn)
    except ZeroDivisionError:
        fnr = np.NaN
    try:
        fmeasure = (2 * precision * sensitivity) / (precision + sensitivity)
    except ZeroDivisionError:
        fmeasure = np.nan
    mse = mean_squared_error(classes, y)
    mcc = matthews_corrcoef(y, classes)
    youden = sensitivity + specificity - 1
    try:
        AUC = roc_auc_score(y, pred)
    except ValueError:
        AUC = np.nan
    hamming = hamming_loss(y, classes)
    kappa = cohen_kappa_score(y, classes)
    gmean = sqrt(sensitivity * specificity)

    ret = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
           'acc': accuracy, 'bal_acc': bal_acc, 'sens': sensitivity, 'spec': specificity, 'fnr': fnr, 'fpr': fpr,
           'fmeas': fmeasure, 'mse': mse, 'youden': youden, 'mcc': mcc, 'auc': AUC,
           'hamming': hamming, 'cohen_kappa': kappa, 'gmean': gmean}

    return ret


class BalancedAccuracyCallback(Callback):
    def __init__(self, training_data, validation_data, prms, save, patience, savedir):
        super().__init__()
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.prms = prms
        self.patience = patience
        self.saved_since = 0
        self.val_best = 0
        self.save = save
        self.best_model = None
        self.curr_dir = savedir
        self.saveepoch = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        if self.save:
            my_store_path = os.path.abspath("".join([self.curr_dir,"/Model_weights_Epoch",str(self.saveepoch),".h5"]))
            my_json_path = os.path.abspath("".join([self.curr_dir, "/Model_architecture_Epoch", str(self.saveepoch), ".json"]))
            self.best_model.save_weights(my_store_path)
            with open(my_json_path, 'w') as outfile:
                json.dump(self.best_model.to_json(), outfile)

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs):
        def _pred_bal_acc(y, pred, group, group_by):
            '''
            function to calculate the balanced accuracy on the whole data and on grouped data
            :param y: class labels
            :param pred: predictions (binary or non binary, will be converted)
            :param group: if balanced accuracy should be calculated on grouped data
            :param group_by: the dataframe on whose index the data should be grouped
            :return: returns balanced accuracy as tuple or a tuple with bal_acc and balanced accuracy of grouped data
            '''
            cls = np.greater(pred, 0.5).astype(int)
            bal_acc = balanced_accuracy_score(y, cls)
            if group:
                pred = pd.DataFrame(pred)
                pred = pred.groupby(group_by.index).mean()
                y = y.groupby(group_by.index).mean()
                cls_grp = np.greater(pred, 0.5).astype(int)
                bal_acc_grp = balanced_accuracy_score(y, cls_grp)
                return bal_acc, bal_acc_grp
            else:
                return bal_acc, bal_acc
        # Performance
        pred = self.model.predict(self.x)
        pred_val = self.model.predict(self.x_val)
        ba_y , ba_y_tb = _pred_bal_acc(self.y, pred , True, self.y)
        ba_y_val, ba_y_val_tb = _pred_bal_acc(self.y_val, pred_val, True, self.y_val)

        print('\rbalanced_accuracy_score: %s - balanced_accuracy_score_val: %s'
              % (str(round(ba_y, 4)), str(round(ba_y_val, 4))), end=100 * ' ' + '\n')
        logs['balanced_accuracy_score'] = ba_y
        logs['balanced_accuracy_score_val'] = ba_y_val
        logs['balanced_accuracy_traceback'] = ba_y_tb
        logs['balanced_accuracy_traceback_val'] = ba_y_val_tb

        # Early stopping
        if self.val_best >= ba_y_val and self.saved_since >= self.patience:
            self.model.stop_training = True
            print('Early stopped at Epoch', epoch)
        elif self.val_best >= ba_y_val:
            self.saved_since += 1
        else:
            self.val_best = ba_y_val
            self.saved_since = 0
            self.best_model = self.model
            self.saveepoch = epoch

            print('Saved model after epoch', epoch)

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_number', default=1, type=int,
                        help='Specifies the fold which should be used to validate the models (default is 1)')
    parser.add_argument('--save_directory', default =os.getcwd(), type=str,
                        help='Specifies the directory to save files in that directory it will create a foler Run_* '
                             'where * is dependent on the hyperparameters, '
                             'defaults to the directory where the script is executed (os.getcwd())')
    parser.add_argument('--config_file', type=str,
                        default=os.path.join(os.getcwd(), "config.json"),
                        help='Specifies config JSON to train models with')
    parser.add_argument('--hidden_units', type=int,
                        default = 1024,
                        help='Specifies the number of hidden units, default is 2')
    parser.add_argument('--learning_rate', type=float,
                        default = 0.1,
                        help='Specifies the learning rate, default is 0.1')
    parser.add_argument('--dropout_input', type=float,
                        default = 0.2,
                        help='Specifies the dropout rate for the input layer, default is 0.2')
    parser.add_argument('--dropout_hidden', type=float,
                        default = 0.5,
                        help='Specifies the row of dropout rate for hidden layers, default is 0.5')
    parser.add_argument('--layers', type=int,
                        default = 2,
                        help='Specifies the number of hidden layers, default is 2')
    parser.add_argument('--gpus', type=str,
                        default="0,1",
                        help='Which gpus to use for training, only the specfied gpus will be visible for the training, '
                             'at least 2 gpus need to be specified!')
    parser.add_argument('--save-model',
                        action='store_true',
                        help='Use this flag if the best model should be saved after training')

    input_args = parser.parse_args()


    #CUDA visible devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # order devices
    os.environ["CUDA_VISIBLE_DEVICES"] = input_args.gpus


    # create paths and dirs to save
    curr_dir = input_args.save_directory

    params = "_".join(str(s) for s in
                      [input_args.hidden_units, input_args.learning_rate, input_args.dropout_input,
                       input_args.dropout_hidden, input_args.layers])
    savedir = os.path.abspath("".join([curr_dir, "/Grid_", params]))

    os.makedirs(os.path.abspath(savedir))  # make path for models to save

    # Load config from JSON
    jsn = json.load(open(input_args.config_file))
    grid_data = jsn["Grid"]
    data_params = jsn["Data"]
    nn_params = jsn["NN_params"]
    class_col = data_params["class_col"]
    data_directory = data_params["path"]
    seed = nn_params["seed"]

    # seed model

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)


    # Load data

    d_train,d_test = load_data(n=input_args.fold_number, path=data_directory)
    x = d_train.drop(class_col, axis=1)
    y = d_train[class_col]
    x_val = d_test.drop(class_col, axis=1)
    y_val = d_test[class_col]
    keys = grid_data.keys()


    # Scale data

    scaler = MinMaxScaler().fit(X=x)
    x = scaler.transform(x)
    x_val = scaler.transform(x_val)

    # save seed

    seed = list(np.random.get_state())
    with open("".join([savedir, "/seed.csv"]), 'w') as f:
        f.write('\n'.join(str(e) for e in seed))

    # Create callbacks

    csv_log = CSVLogger("".join([savedir, "/log.csv"]), separator=',', append=False)
    roc = BalancedAccuracyCallback(training_data=(x, y), validation_data=(x_val, y_val), prms=params, patience=15, save=input_args.save_model, savedir = savedir)

    # Create Model

    loss = getattr(losses, nn_params["loss"])
    masked_loss = build_masked_loss(loss, -1)

    model = create_model(inp=int(x.shape[1]), hid=input_args.hidden_units, drop_in=input_args.dropout_input,
                         drop_hid=input_args.dropout_hidden, activation=nn_params["activation"],layers=input_args.layers,
                         initializer = nn_params['initializer'])
    model = multi_gpu_model(model, gpus=2)
    model.compile(loss=masked_loss, optimizer=optimizers.SGD(lr=input_args.learning_rate, momentum=nn_params['momentum']),
                  metrics=["accuracy"])

    hist = model.fit(x=x, y=y, validation_data=(x_val, y_val), batch_size=nn_params["batch_size"],
              epochs=nn_params["epochs"], verbose=0,
                     callbacks=[roc, csv_log])
    pd.DataFrame(hist.params).to_csv("".join([savedir, "/params.csv"]))

    # Evaluate Model
    ## on all datapoints separately
    pred = roc.best_model.predict(x=x_val)
    perf = evaluate(y_val,pred)
    perf = pd.DataFrame(perf, index=[params])
    store_path = os.path.abspath("".join([curr_dir, "/../performance.csv"]))

    if os.path.exists(store_path):
        perf.to_csv(store_path, mode='a', header=False)
    else:
        perf.to_csv(store_path)

    ## on mean of predictions for one compound
    pred_tb = roc.best_model.predict(x=x_val)
    pred_tb = pd.DataFrame(pred_tb)
    pred_tb = pred_tb.groupby(y_val.index).mean()
    pred_tb = pred_tb.values
    y_val_tb = y_val.groupby(y_val.index).mean()
    perf_tb = evaluate(y_val_tb, pred_tb)
    perf_tb = pd.DataFrame(perf_tb, index=[params])
    store_path = os.path.abspath("".join([curr_dir, "/../performance_traceback.csv"]))

    if os.path.exists(store_path):
        perf.to_csv(store_path, mode='a', header=False)
    else:
        perf.to_csv(store_path)




