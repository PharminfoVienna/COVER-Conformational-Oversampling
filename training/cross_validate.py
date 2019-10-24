import argparse
import datetime
from itertools import product
import json
import os
import pandas as pd
import shutil
import sys
import subprocess


def make_dirs(*args):
    for arg in args:
        os.mkdir(os.path.abspath(arg))


def create_grid(config_file_path, csv_output_dir=None):
    '''
    TODO
    :param config_file_path: str, the config file which contains the values used for the grid generation
    :param csv_output_dir: str, if the file needs to be saved, put the save path here
    :return: pandas.DataFrame with one hyperparameter set per row
    '''
    grid_path = os.path.join(csv_output_dir, "Grid.csv")
    config_file = json.load(open(config_file_path))
    grid_data = config_file["Grid"]
    grid = list(product(*grid_data.values()))
    grid = pd.DataFrame(grid, columns=grid_data.keys())
    if csv_output_dir is not None:
        grid.to_csv(grid_path)
    return grid


def get_best_params(path):
    """Calculates the mean of all CV runs and uses the maximum to define parameters for final model
    TODO docstring settings
    """
    data = pd.read_csv(path, header=0, index_col=0)
    data = data.groupby(data.index).mean()
    params = data["bal_acc"].idxmax()
    params = params.split(sep="_")
    hidden_units, learning_rate, dropout_input, dropout_hidden, layers = params
    print(params)
    return int(hidden_units), float(learning_rate), float(dropout_input), float(dropout_hidden), int(layers)


if __name__ == "__main__":
    TRAIN_GRID_SCRIPT = "train_model.py"
    TRAIN_FINAL_SCRIPT = "train_final_model.py"

    CODE_FILES = [TRAIN_FINAL_SCRIPT, TRAIN_GRID_SCRIPT, "cross_validate.py"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-base-dir',
                        default=os.getcwd(),
                        help='Specify base directory for output directory structure, defaults to the directory where '
                             'the script is executed (os.getcwd())')
    parser.add_argument('--input-dir',
                        default=os.getcwd(),
                        help='Specify directory where source files are located, defaults to the directory where '
                             'the script is executed (os.getcwd())')
    parser.add_argument('--config-file',
                        default=os.path.join(os.getcwd(), "config.json"),
                        help='Specify config JSON to train models with')
    parser.add_argument('--gpus',
                        default="0,1",
                        help='Which gpus to use for training, only the specfied gpus will be visible for the training, '
                             'at least 2 gpus need to be specified!')
    input_args = parser.parse_args()

    # TODO maybe add verbosity /debug option ?

    if not os.path.isdir(input_args.out_base_dir):
        print(" Option --out-base-dir is not directory")
        sys.exit(1)
    if not os.path.isdir(input_args.input_dir):
        print(" Option --input-dir is not directory")
        sys.exit(1)
    for file in CODE_FILES:
        if not os.path.isfile(os.path.join(input_args.input_dir, file)):
            print("File %s does not exist in input directory" % file)
            sys.exit(1)
    if os.path.isfile(input_args.config_file):
        if os.path.isabs(input_args.config_file):
            config_file = input_args.config_file
        elif os.path.isfile(os.path.abspath(input_args.config_file)):
            config_file = os.path.abspath(input_args.config_file)
        else:
            print("Please specify the absolute path for the config file")
            sys.exit(1)
    else:
        print(" Option --config-file is not a file")
        sys.exit(1)

    output_base_dir = input_args.out_base_dir
    output_dir = os.path.join(output_base_dir, "Run_" + datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S'))
    code_dir = os.path.join(output_dir, "code")
    data_dir = os.path.join(output_dir, "data")
    final_model_dir = os.path.join(output_dir, "final_models")
    trained_model_dir = os.path.join(output_dir, "trained_models")
    make_dirs(output_dir, code_dir, data_dir, final_model_dir, trained_model_dir)

    for file in CODE_FILES:
        file_src = os.path.join(input_args.input_dir, file)
        shutil.copy2(file_src, code_dir)

    _, config_filename = os.path.split(input_args.config_file)
    shutil.copy2(input_args.config_file, data_dir)

    grid = create_grid(config_file_path=input_args.config_file, csv_output_dir=data_dir)

    for cross_val_iter in range(1, 6):
        final_cross_val_dir = os.path.join(final_model_dir, "Loop_" + str(cross_val_iter))
        final_cross_val_dir = os.path.abspath(final_cross_val_dir)
        train_cross_val_dir = os.path.join(trained_model_dir, "Loop_" + str(cross_val_iter))
        train_cross_val_dir = os.path.abspath(train_cross_val_dir)
        make_dirs(final_cross_val_dir, train_cross_val_dir)

        for grid_search_iter in range(1, 4):
            grid_search_dir = os.path.join(train_cross_val_dir, "Inner_" + str(grid_search_iter))
            grid_search_dir = os.path.abspath(grid_search_dir)
            os.mkdir(grid_search_dir)

            for grid_row in grid.itertuples():
                subprocess.call(["python", TRAIN_GRID_SCRIPT,
                                 '--fold_number', str(cross_val_iter) + str(grid_search_iter),
                                 '--save_directory', grid_search_dir,
                                 '--config_file', os.path.abspath(os.path.join(data_dir, config_filename)),
                                 '--hidden_units', str(grid_row.hidden_units),
                                 '--learning_rate', str(grid_row.learning_rate),
                                 '--dropout_input', str(grid_row.dropout_input),
                                 '--dropout_hidden', str(grid_row.dropout_hidden),
                                 '--layers', str(grid_row.layers),
                                 '--gpus', input_args.gpus],
                                cwd=code_dir)

        hidden_units, learning_rate, dropout_input, dropout_hidden, layers = get_best_params(
            train_cross_val_dir + "/performance_traceback.csv")

        subprocess.call(["python", TRAIN_GRID_SCRIPT,
                         '--fold_number', str(cross_val_iter) + str(0),
                         '--save_directory', final_cross_val_dir,
                         '--config_file', os.path.abspath(os.path.join(data_dir, config_filename)),
                         '--hidden_units', str(hidden_units),
                         '--learning_rate', str(learning_rate),
                         '--dropout_input', str(dropout_input),
                         '--dropout_hidden', str(dropout_hidden),
                         '--layers', str(layers),
                         '--gpus', input_args.gpus,
                         '--save-model'],
                        cwd=code_dir)

    print("done")
