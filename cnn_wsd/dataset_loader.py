import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import load_h5, random_shuffle
from tensorflow.keras.utils import to_categorical


def balance_dataset(X, Y, shuffle=False):
    """Balance the dataset by taking the minimum number of samples per class (under-sampling)

    Parameters
    ----------
    X : numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands)

    Y : numpy array
        dataset of labels in string, shape (n_samples,)

    shuffle : bool, optional
        Shuffle the dataset, by default False

    Returns
    -------
    numpy array
        balanced dataset of images in float32, shape (n_samples, height, width, n_bands)

    numpy array
        balanced dataset of labels in string, shape (n_samples,)
    """
    if shuffle:
        X, Y = random_shuffle(X, Y)
    cat, counts = np.unique(Y, return_counts=True)
    min_count = np.min(counts)
    X_bal = []
    Y_bal = []
    for category in cat:
        idx = np.where(Y == category)[0]
        idx = idx[:min_count]
        X_bal.append(X[idx])
        Y_bal.append(Y[idx])
    X_bal = np.concatenate(X_bal)
    Y_bal = np.concatenate(Y_bal)
    return X_bal, Y_bal


def prepare_data_cnn(
    X,
    Y,
    frac_val=0.15,
    band_max=[0, 1, 6, 7],
    balanced=[False, False],
    shuffle=True,
    categorical=False,
):
    """Prepare the data for the CNN model, it suppose that the data are stored in hdf5 files in the same folder and named "data_train.h5" and "data_test.h5"

    Parameters
    ----------
    ipath : str
        Path to the hdf5 files

    frac_val : float, optional
        Fraction of the dataset to use for validation, by default 0.15

    band_max : list, optional
        List of the bands to use, by default [0, 1, 6, 7]

    balanced : list, optional
        List of boolean to balance the dataset train and test, by default [False, False]

    shuffle : bool, optional
        Shuffle the dataset (seed=42), by default True

    categorical : bool, optional
        If True, encode the labels using to_categorical, by default False

    Returns
    -------
    numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands) of the train set

    numpy array
        dataset of labels in string, shape (n_samples,) of the train set

    numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands) of the test set

    numpy array
        dataset of labels in string, shape (n_samples,) of the test set

    numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands) of the validation set

    numpy array
        dataset of labels in string, shape (n_samples,) of the validation set

    sklearn.preprocessing.LabelEncoder
        LabelEncoder object to transform the labels into integers
    """
    if len(X) == 2:
        X_train, Y_train = X[0], Y[0]
        X_test, Y_test = X[1], Y[1]
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 0.2, shuffle=False)

    if band_max is not None:
        X_train = X_train[:, :, :, band_max]
        X_test = X_test[:, :, :, band_max]

    if shuffle:
        X_train, Y_train = random_shuffle(X_train, Y_train)
        X_test, Y_test = random_shuffle(X_test, Y_test)

    Le = LabelEncoder()
    y_train = Le.fit_transform(Y_train)
    y_test = Le.transform(Y_test)

    if balanced[0]:
        X_train, y_train = balance_dataset(X_train, y_train)
    if balanced[1]:
        X_test, y_test = balance_dataset(X_test, y_test)

    if frac_val > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=frac_val, shuffle=False
        )
    else:
        X_val, y_val = np.array([np.nan]), np.array([np.nan])

    if categorical:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_val = to_categorical(y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val, Le
