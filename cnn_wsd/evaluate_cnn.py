from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import pandas as pd
import numpy as np
import os

from dataset_loader import prepare_data_cnn
from metrics import BAROC, FRCROC
from model_selection import BFold
from architecture import M5cnn24F
from utils import (
    report_metric_from_log,
    init_logger,
    dump_pkl,
    load_h5,
    write_report,
    report_prediction,
)


def CNN_Bfold_learning(
    model, learning_rate, X, y, frac_val, batch_size, n_epochs, callbacks, logg
):
    """Evaluate the keras model with kfold cross validation, and return the f1, accuracy score and the weights learned

    Parameters
    ----------
    model : keras model
        Model to evaluate

    learning_rate : float
        Learning rate of the model

    X : np.array
        Input data (features) be careful to have the same shape as the input of the model

    y : np.array
        Labels of the data (str or int)

    frac_val : float
        Fraction of the data to use for validation

    batch_size : int
        Batch size to use for training

    n_epochs : int
        Number of epochs to train the model

    callbacks : list
        List of callbacks to use for training

    logg : logging
        Logger

    Returns
    -------
    list
        List of f1 score for each fold

    list
        List of accuracy score for each fold

    logging
        Logger

    list
        List of confusion matrix for each fold with the learned constant threshold at 5% FRCOC

    list
        List of threshold for each fold with the learned constant threshold at 5% FRCOC

    list
        List of threshold for each fold with the learned threshold for best accuracy BAROC
    """
    bkf = BFold(shuffle=False, random_state=42)

    kfold = 0
    f1 = []
    acc = []
    cmfcr = []
    Tfrcoc = []
    Tbaroc = []

    X_train = X[0]
    X_test = X[1]
    y_train = y[0]
    y_test = y[1]

    for train_index in bkf.split(X_train, y_train):
        logg.info(f"Kfold : {kfold}")
        X_train_K, y_train_k = X_train[train_index], y_train[train_index]
        logg.info(f" y_train_k : {np.unique(y_train_k, return_counts=True)}")
        m = model(learning_rate)
        logg.info("#" * 50)
        logg.info(f"Learning rate : {learning_rate}")
        m.summary(print_fn=lambda x: logg.info(x))

        h = m.fit(
            X_train_K,
            y_train_k,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_split=frac_val,
            callbacks=callbacks,
            workers=-1,
            use_multiprocessing=True,
            verbose=0,
        )
        top = {h.history["loss"][0]}
        end = {h.history["loss"][-1]}
        logg.info(f"Loss:  {top} -> {end}")
        top_val = {h.history["val_loss"][0]}
        end_val = {h.history["val_loss"][-1]}
        logg.info(f"Val loss:  {top_val} -> {end_val}")
        logg.info(f"Model trained")
        ypred = m.predict(X_test)
        ypred_train = m.predict(X_train_K)

        accF, t_frcoc = FRCROC(y_train_k, ypred_train, 0.05)
        logg.info(f"accuracy frcoc 5% : {accF}")
        logg.info(f"threshold frcoc 5% : {t_frcoc}")

        accB, t_baroc = BAROC(y_train_k, ypred_train)
        logg.info(f"accuracy baroc : {accB}")
        logg.info(f"threshold baroc : {t_baroc}")

        logg, f1_k, acc_k, cfm_k = report_prediction(
            y_test, ypred, le, logg, t_frcoc, t_baroc
        )
        f1.append(f1_k)
        acc.append(acc_k)
        cmfcr.append(cfm_k)
        Tfrcoc.append(t_frcoc)
        Tbaroc.append(t_baroc)
        kfold += 1
        del m
    return f1, acc, logg, cmfcr, Tfrcoc, Tbaroc


def RF_Bfold_learning(
    model, learning_rate, X, y, frac_val, batch_size, n_epochs, callbacks, logg
):
    """Evaluate the keras model with kfold cross validation, and return the f1, accuracy score and the weights learned

    Parameters
    ----------
    X : np.array
        Input data (features) be careful to have the same shape as the input of the model

    y : np.array
        Labels of the data (str or int)

    logg : logging
        Logger

    Returns
    -------
    list
        List of f1 score for each fold

    list
        List of accuracy score for each fold

    logging
        Logger

    list
        List of confusion matrix for each fold with the learned constant threshold at 5% FRCOC

    list
        List of threshold for each fold with the learned constant threshold at 5% FRCOC

    list
        List of threshold for each fold with the learned threshold for best accuracy BAROC
    """
    bkf = Bfold(shuffle=False, random_state=42)

    kfold = 0
    f1 = []
    acc = []
    cmfcr = []
    Tfrcoc = []
    Tbaroc = []

    X_train = X[0]
    X_test = X[1]
    y_train = y[0]
    y_test = y[1]

    for train_index in bkf.split(X_train, y_train):
        logg.info(f"Kfold : {kfold}")

        X_train_K, y_train_k = X_train[train_index], y_train[train_index]
        logg.info(f" y_train_k : {np.unique(y_train_k, return_counts=True)}")
        logg.info("#" * 50)
        rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        rf.fit(X_train_K, y_train_k)
        ypred = rf.predict(X_test)
        ypred_train = rf.predict(X_train_K)

        accF, t_frcoc = FRCROC(y_train_k, ypred_train, 0.05)
        logg.info(f"accuracy frcoc 5% : {accF}")
        logg.info(f"threshold frcoc 5% : {t_frcoc}")

        accB, t_baroc = BAROC(y_train_k, ypred_train)
        logg.info(f"accuracy baroc : {accB}")
        logg.info(f"threshold baroc : {t_baroc}")

        logg, f1_k, acc_k, cfm_k = report_prediction(
            y_test, ypred, le, logg, t_frcoc, t_baroc
        )
        f1.append(f1_k)
        acc.append(acc_k)
        cmfcr.append(cfm_k)
        Tfrcoc.append(t_frcoc)
        Tbaroc.append(t_baroc)
        kfold += 1
        del rf
    return f1, acc, logg, cmfcr, Tfrcoc, Tbaroc


if __name__ == "__main__":
    frac_val = 0.15
    band_max = [0, 1, 6, 7]
    learning_rate = 0.0010
    balanced = [False, True]
    shuffle = True
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-9,
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

    X_train, X_test, _, y_train, y_test, _, le = prepare_data_cnn(
        [X_train, X_test],
        [Y_train, Y_test],
        frac_val=-1,
        band_max=band_max,
        balanced=balanced,
        shuffle=shuffle,
        categorical=False,
    )

    logg.info(f" Y train : {np.unique(y_train, return_counts=True)}")
    logg.info(f" Y test : {np.unique(y_test, return_counts=True)}")
    logg.info(f" X train : {X_train.shape}")
    logg.info(f" X test : {X_test.shape}")
    logg.info(f" Band max : {band_max}")
    logg.info(f" Balanced : {balanced}")
    logg.info(f" Shuffle : {shuffle}")
    logg.info(f" Learning rate : {learning_rate}")

    f1a, acca, logg, cmfcr, Tfrcoc, Tbaroc = CNN_Bfold_learning(
        M5cnn24F,
        learning_rate,
        [X_train, X_test],
        [y_train, y_test],
        frac_val,
        batch_size,
        n_epochs,
        callbacks,
        logg,
    )
    logg.info(f"CNN Threshold FRCOC (5%) mean: {np.mean(Tfrcoc)}")
    logg.info(f"CNN Threshold BAROC mean: {np.mean(Tbaroc)}")

    f1b, accb, logg, cmfcr, Tfrcoc, Tbaroc = RF_Bfold_learning(
        [X_train, X_test], [y_train, y_test], logg
    )
    logg.info(f"RF Threshold FRCOC (5%) mean: {np.mean(Tfrcoc)}")
    logg.info(f"RF Threshold BAROC mean: {np.mean(Tbaroc)}")
    dic = {
        "M5cnn24F": {"f1": f1a, "acc": acca, "cfmf": cmfcra},
        "RF": {"f1": f1b, "acc": accb, "cfmf": cmfcrb},
    }
    dump_pkl(dic, os.path.join(path_results, "kfold_dic.pkl"))

    logg = report_metric_from_log(dic, logg)
    pathreport = os.path.join(path_results, "report.txt")
    write_report(pathlog, pathreport)
