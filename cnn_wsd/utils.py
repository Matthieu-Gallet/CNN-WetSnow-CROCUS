import re, os, h5py
import logging, pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from metrics import BAROC, FRCROC


def dump_pkl(obj, path):
    """Dump object in pickle file

    Parameters
    ----------
    obj : object
        Object to dump, can be a list, a dict, a numpy array, etc.

    path : str
        Path to the pickle file

    Returns
    -------
    int
        1 if the dump is successful

    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return 1


def open_pkl(path):
    """Open pickle file

    Parameters
    ----------
    path : str
        Path to the pickle file to open

    Returns
    -------
    object
        Object contained in the pickle file

    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def open_log_file(path_log):
    """Open log file (.log) to parse it

    Parameters
    ----------
    path_log : str
        Path to the log file

    Returns
    -------
    list
        List of lines in the log file

    """
    with open(path_log, "r") as f:
        log = f.readlines()
    return log


def clean_log(log):
    """Clean log file to remove useless information (line number, time, etc.) to make it more readable

    Parameters
    ----------
    log : list
        List of lines in the log file

    Returns
    -------
    list
        List of lines in the log file without useless information

    """
    result = []
    for i in range(len(log)):
        pattern = r"Line: \d+ - (.*)$"
        match = re.search(pattern, log[i])

        if match:
            result.append(match.group(1) + "\n")
        else:
            result.append(log[i])
    return result


def write_report(path_log, path_report):
    """Write a txt report from a log file (.log)

    Parameters
    ----------
    path_log : str
        Path to the log file

    path_report : str
        Path to the report file

    Returns
    -------
    list
        List of lines in the log file without useless information

    """
    op = open_log_file(path_log)
    result = clean_log(op)
    with open(path_report, "w") as f:
        f.writelines(result)
    return result


def report_metric_from_log(dic, logg):
    """Report metric from dictionary containing the f1 and accuracy score in a log file

    Parameters
    ----------
    dic : dict
        Dictionary containing the f1 and accuracy score for each model

    logg : logging
        Logger

    Returns
    -------
    logging
        Logger

    """
    logg.info(f"======== Final report ========")
    for i in list(dic.keys()):
        logg.info(f"-------- Model : {i} --------")
        f1 = dic[i]["f1"]
        acc = dic[i]["acc"]
        cm = dic[i]["cfmf"]
        logg.info(f"confusion matrix frcroc: ")
        c = pd.DataFrame(
            np.mean(cm, axis=0).round(4), columns=cm[0].columns, index=cm[0].index
        )
        logg.info(c.to_string())
        logg.info(f"f1 : {np.mean(f1)} +/- {np.std(f1)}")
        logg.info(f"acc : {np.mean(acc)} +/- {np.std(acc)}")
    logg.info(f"======== End report ========")
    return logg


def report_prediction(y_true, y_pred, le, logg, t_fcroc=None, t_baroc=None):
    """Compute the f1 and accuracy score and the confusion matrix from the true and predicted labels and report it in a log file
    The y_true and y_pred must be categorical (one hot encoded: [[0, 1, 0], [1, 0, 0], [0, 0, 1]]) or binary (0 or 1)

    Parameters
    ----------
    y_true : numpy array
        True labels

    y_pred : numpy array
        Predicted labels

    le : LabelEncoder
        LabelEncoder object

    logg : logging
        Logger

    t_fcroc : float, optional
        Threshold for the FRCROC metric, by default None

    t_baroc : float, optional
        Threshold for the BAROC metric, by default None

    Returns
    -------
    logging
        Logger

    float
        f1 score

    float
        accuracy score

    """
    logg.info("----------- REPORT -----------")
    if y_pred.shape[1] > 1:
        y_true = y_true.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)
    else:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        if t_fcroc != None:
            y_pred_frcroc = np.where(y_pred > t_fcroc, 1, 0)
            y_pred_frcroc = le.inverse_transform(y_pred_frcroc)
        if t_baroc != None:
            y_pred_baroc = np.where(y_pred > t_baroc, 1, 0)
            y_pred_baroc = le.inverse_transform(y_pred_baroc)
        y_pred = np.where(y_pred > 0.5, 1, 0)

    y_true = le.inverse_transform(y_true)
    y_pred = le.inverse_transform(y_pred)
    logg.info(f"confusion matrix : ")
    cfm = pd.DataFrame(
        100 * confusion_matrix(y_true, y_pred, normalize="true").round(4),
        columns=le.classes_,
        index=le.classes_,
    )
    logg.info(cfm.to_string())
    f1 = 100 * f1_score(y_true, y_pred, average="macro").round(5)
    acc = 100 * accuracy_score(y_true, y_pred).round(5)
    logg.info(f"f1 score : {f1}")
    logg.info(f"accuracy score : {acc}")
    if t_fcroc != None:
        cfmf = pd.DataFrame(
            100 * confusion_matrix(y_true, y_pred_frcroc, normalize="true").round(4),
            columns=le.classes_,
            index=le.classes_,
        )
        logg.info(f"confusion matrix fcroc : ")
        logg.info(cfmf.to_string())

        f1f = 100 * f1_score(y_true, y_pred_frcroc, average="macro").round(5)
        accf = 100 * accuracy_score(y_true, y_pred_frcroc).round(5)
        logg.info(f"f1 score fcroc : {f1f}")
        logg.info(f"accuracy score fcroc : {accf}")
    if t_baroc != None:
        cfmb = pd.DataFrame(
            100 * confusion_matrix(y_true, y_pred_baroc, normalize="true").round(4),
            columns=le.classes_,
            index=le.classes_,
        )
        logg.info(f"confusion matrix baroc : ")
        logg.info(cfmb.to_string())
        f1b = 100 * f1_score(y_true, y_pred_baroc, average="macro").round(5)
        accb = 100 * accuracy_score(y_true, y_pred_baroc).round(5)
        logg.info(f"f1 score baroc : {f1b}")
        logg.info(f"accuracy score baroc : {accb}")
    logg.info("----------- END REPORT -----------")
    return logg, f1, acc, cfmf


def init_logger(path_log):
    """Initialize a logger

    Parameters
    ----------
    path_log : str
        Path to the log file

    Returns
    -------
    logging
        Logger

    str
        Path to the log file

    """
    now = datetime.now()
    namlog = now.strftime("%d%m%y_%HH%MM%S")
    datestr = "%m/%d/%Y-%I:%M:%S %p "
    filename = os.path.join(path_log, f"log_{namlog}.log")
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        format="%(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s",
    )
    logging.info("Started")
    return logging, filename


def save_h5(img, label, filename):
    """Save image and label in a hdf5 file

    Parameters
    ----------
    img : numpy array
        dataset of images in float32

    label : numpy array
        dataset of labels in string

    filename : str
        Path to the hdf5 file

    Returns
    -------
    None
    """
    if ".h5" not in filename:
        filename += ".h5"
    with h5py.File(filename, "w") as hf:
        hf.create_dataset(
            "img", np.shape(img), h5py.h5t.IEEE_F32BE, compression="gzip", data=img
        )  # IEEE_F32BE is big endian float32
        hf.create_dataset(
            "label", np.shape(label), compression="gzip", data=label.astype("S")
        )


def load_h5(filename):
    """Load image and label from a hdf5 file

    Parameters
    ----------
    filename : str
        Path to the hdf5 file

    Returns
    -------

    numpy array
        dataset of images in float32

    numpy array
        dataset of labels in string

    """
    if ".h5" not in filename:
        filename += ".h5"
    with h5py.File(filename, "r") as hf:
        data = np.array(hf["img"][:]).astype(np.float32)
        meta = np.array(hf["label"][:]).astype(str)
    return data, meta


def random_shuffle(X, y, rng=-1):
    """Shuffle randomly the dataset

    Parameters
    ----------
    X : numpy array
        dataset of images

    y : numpy array
        dataset of labels

    rng : int, optional
        Random seed, by default -1, must be a np.random.default_rng() object

    Returns
    -------
    numpy array
        shuffled dataset of images

    numpy array
        shuffled dataset of labels

    """
    if rng == -1:
        rng = np.random.default_rng(42)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y
