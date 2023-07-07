import pandas as pd
import numpy as np
import os

from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from tqdm import tqdm
from architecture import (
    M1cnn96F,
    M2cnn32F,
    M3cnn24F,
    M4cnn16F,
    M5cnn24F,
    M5cnn16F3D,
    M6cnn24F1D,
    M7cnn24F1D,
)
from utils import (
    load_h5,
    report_prediction,
    init_logger,
    dump_pkl,
    write_report,
)

from dataset_loader import prepare_data_cnn


def logg_info_data(
    X_train,
    X_test,
    X_val,
    y_train,
    y_test,
    y_val,
    le,
    frac_test,
    frac_val,
    frac_false,
    n_epochs,
    batch_size,
    shuffle,
    logg,
):
    """Log information about the data

    Parameters
    ----------
    X_train : numpy array
        Input data (features) for training

    X_test : numpy array
        Input data (features) for testing

    X_val : numpy array
        Input data (features) for validation

    y_train : numpy array
        Labels of the data (str or int) for training

    y_test : numpy array
        Labels of the data (str or int) for testing

    y_val : numpy array
        Labels of the data (str or int) for validation

    le : sklearn.preprocessing.LabelEncoder
        LabelEncoder used to encode the labels

    balanced : list
        List of booleans, if True, balance the data for training and testing

    frac_val : float
        Fraction of the data to use for validation

    band_max : list
        List of the bands to use

    n_epochs : int
        Number of epochs to train the model

    batch_size : int
        Batch size to use for training

    shuffle : bool
        If True, shuffle the data

    logg : logging
        Logger

    Returns
    -------
    logging
        Logger with the information about the data
    """
    logg.info("Data loaded")
    logg.info("Preparing data")
    logg.info(f"balanced : {balanced}")
    logg.info(f"fraction validation : {frac_val}")
    logg.info(f"Bands : {band_max}")
    logg.info(f"suffle : {shuffle}")
    logg.info(f"X_train shape : {X_train.shape}")
    logg.info(f"X_test shape : {X_test.shape}")
    logg.info(f"X_val shape : {X_val.shape}")
    logg.info(f"y_train : {np.unique(y_train, return_counts=True)}")
    logg.info(f"y_test : {np.unique(y_test, return_counts=True)}")
    logg.info(f"y_val : {np.unique(y_val, return_counts=True)}")
    logg.info(f"le.classes_ : {le.classes_}")
    logg.info(f"le.transform(le.classes_) : {le.transform(le.classes_)}")
    logg.info(f"n_epochs : {n_epochs}")
    logg.info(f"batch_size : {batch_size}")
    return logg


def extract_best_lr(dic):
    """Extract the best learning rate for each model stored in a dictionary

    Parameters
    ----------
    dic : dict
        Dictionary with the f1 and accuracy for each model and learning rate

    Returns
    -------
    pandas.DataFrame
        Dataframe with the best learning rate for each model
    """

    pdf = pd.DataFrame(dic).T
    pdf.columns = ["f1", "acc"]
    pdf["learning_rate"] = pdf.index.str.split("_").str[1].astype(float)
    pdf["model"] = pdf.index.str.split("_").str[0]
    pdf.reset_index(drop=True, inplace=True)
    best_lr_per_model = pdf.groupby("model").apply(lambda x: x.loc[x["f1"].idxmax()])
    return best_lr_per_model


def info_training(m, h, logg, dic, name_model, X_test, y_test, X_train, y_train, le):
    """Log information about the training of a keras model, calculate the f1 and accuracy and store them in a dictionary.
    Log the first and last loss and val_loss

    Parameters
    ----------
    m : keras model
        Model to train

    h : keras history
        History of the training

    logg : logging
        Logger

    dic : dict
        Dictionary with the f1 and accuracy for each model and learning rate

    name_model : str
        Name of the model

    Returns
    -------
    logging
        Logger with the information about the training
    """
    logg.info(f"Model trained")
    top = {np.array(h.history["loss"])[0].round(5)}
    end = {np.array(h.history["loss"])[-1].round(5)}
    logg.info(f"Loss:  {top} -> {end}")
    top_val = {np.array(h.history["val_loss"])[0].round(5)}
    end_val = {np.array(h.history["val_loss"])[-1].round(5)}
    logg.info(f"Val loss:  {top_val} -> {end_val}")

    ypred_train = m.predict(X_train)
    acc_tfcroc, T_frcoc = FRCOC(y_train, ypred_train, 0.05)
    logg.info(f"accuracy frcoc 5% : {acc_tfcroc}")
    logg.info(f"threshold frcoc 5% : {T_frcoc}")

    acc_tbaroc, T_baroc = BAROC(y_train, ypred_train)
    logg.info(f"accuracy baroc : {acc_tbaroc}")
    logg.info(f"threshold baroc : {T_baroc}")

    ypred = m.predict(X_test)
    logg, f1, acc, _ = report_prediction(y_test, ypred, le, logg, T_frcoc, T_baroc)
    dic[name_model] = [f1, acc]
    return logg, dic


if __name__ == "__main__":
    frac_val = 0.15
    band_max = [0, 1, 6, 7]
    balanced = [True, True]
    shuffle = True

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-8,
            patience=10,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )
    ]
    n_epochs = 200
    batch_size = 1024

    name_folder = datetime.now().strftime("%d%m%y_%HH%MM%S")
    path_results = f"../results/{name_folder}/"
    path_data = "../../dataset_/dataset_A_HD16_TN0HS40_LOG/final_E4/"
    os.makedirs(path_results, exist_ok=True)
    logg, pathlog = init_logger(path_results)

    logg.info("Loading data")
    logg.info(f"Path data : {path_data}")

    X_train, Y_train = load_h5(os.path.join(path_data, "data_train.h5"))
    X_test, Y_test = load_h5(os.path.join(path_data, "data_test.h5"))

    X_train, X_test, X_val, y_train, y_test, y_val, le = prepare_data_cnn(
        [X_train, X_test],
        [Y_train, Y_test],
        frac_val=frac_val,
        band_max=band_max,
        balanced=balanced,
        shuffle=shuffle,
        categorical=False,
    )

    logg = logg_info_data(
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        le,
        balanced,
        frac_val,
        band_max,
        n_epochs,
        batch_size,
        shuffle,
        logg,
    )
    dic = {}
    lr = [8e-2, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]
    models = [
        M1cnn96F,
        M2cnn32F,
        M3cnn24F,
        M4cnn16F,
        M5cnn24F,
        M5cnn16F3D,
        M6cnn24F1D,
        M7cnn24F1D,
    ]
    for model in tqdm(models, desc="model", leave=False):
        for learning_rate in tqdm(lr, desc="learning rate", leave=False):
            m = model(learning_rate)
            name_model = f"{m.name}_{learning_rate}"
            logg.info("#" * 50)
            logg.info(f"Learning rate : {learning_rate}")
            m.summary(print_fn=lambda x: logg.info(x))
            h = m.fit(
                X_train,
                y_train,
                epochs=n_epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                workers=-1,
                use_multiprocessing=True,
                verbose=0,
            )
            logg, dic = info_training(
                m, h, logg, dic, name_model, X_test, y_test, X_train, y_train, le
            )
            del m, h
    dump_pkl(dic, os.path.join(path_results, "dic.pkl"))
    best_lr_per_model = extract_best_lr(dic)
    logg.info(f"best_lr_per_model : \n{best_lr_per_model}")

    pathreport = os.path.join(path_results, "report.txt")
    write_report(pathlog, pathreport)
