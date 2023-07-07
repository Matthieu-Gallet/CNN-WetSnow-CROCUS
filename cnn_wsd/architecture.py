import tensorflow as tf
from tensorflow.keras.models import Sequential


def M1_CNN_96F(lr, input_dim=(16, 16, 4)):
    """Model 1: 96 filters, 3x3 kernel, 2x2 maxpooling, 64 neurons in dense layer

    Parameters
    ----------
    lr : float
        Learning rate

    input_dim : tuple
        Input shape

    Returns
    -------
    model : tf.keras.Sequential
        Model 1
    """
    model = tf.keras.Sequential(name="M1_CNN_96F")
    model.add(
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_dim)
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    return model


def M2_CNN_32F(lr, input_dim=(16, 16, 4)):
    """Model 2: 32 filters, 3x3 kernel, 2x2 maxpooling, 64 neurons in dense layer

    Parameters
    ----------
    lr : float
        Learning rate

    input_dim : tuple
        Input shape

    Returns
    -------
    model : tf.keras.Sequential
        Model 2
    """
    model = Sequential(name="M2_CNN_32F")
    model.add(
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_dim)
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    return model


def M3_CNN_24F(lr, input_dim=(16, 16, 4)):
    """Model 3: 24 filters, 3x3 kernel, 64 neurons in dense layer

    Parameters
    ----------
    lr : float
        Learning rate

    input_dim : tuple
        Input shape

    Returns
    -------
    model : tf.keras.Sequential
        Model 3
    """
    model = Sequential(name="M3_CNN_24F")
    model.add(
        tf.keras.layers.Conv2D(
            8, (3, 3), padding="same", activation="relu", input_shape=input_dim
        )
    )
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    return model


def M4_CNN_16F(lr, input_dim=(16, 16, 4), num_filters=8, num_classes=1):
    """Model 4: 16 filters, 3x3 kernel, 2x2 maxpooling, 64 neurons in dense layer

    Parameters
    ----------
    lr : float
        Learning rate

    input_dim : tuple
        Input shape

    num_filters : int
        Number of filters

    num_classes : int
        Number of classes

    Returns
    -------
    model : tf.keras.Sequential
        Model 4
    """
    model = Sequential(name="M4_CNN_16F")
    model.add(
        tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=input_dim,
        )
    )
    model.add(
        tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    return model


def M5_CNN_24F(lr, input_dim=(16, 16, 4), num_filters=8, num_classes=1):
    """Model 5: 24 filters, 3x3 kernel, 2x2 maxpooling, 128 neurons in dense layer

    Parameters
    ----------
    lr : float
        Learning rate

    input_dim : tuple
        Input shape

    num_filters : int
        Number of filters

    num_classes : int
        Number of classes

    Returns
    -------
    model : tf.keras.Sequential
        Model 5
    """
    model = Sequential(name="M5_CNN_24F")
    model.add(
        tf.keras.layersConv2D(
            num_filters,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=input_dim,
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        tf.keras.layers.Conv2D(num_filters, (2, 2), padding="same", activation="relu")
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.75))
    model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    return model


def M5_CNN_16F_3D(lr, input_dim=(16, 16, 4, 1), num_filters=8, num_classes=1):
    """Model 5: 16 filters, 3x3x3 kernel, 2x2x2 maxpooling

    Parameters
    ----------
    lr : float
        Learning rate

    input_dim : tuple
        Input shape

    num_filters : int
        Number of filters

    num_classes : int
        Number of classes

    Returns
    -------
    model : tf.keras.Sequential
        Model 5
    """
    model = Sequential(name="M5_CNN_16F_3D")
    model.add(
        tf.keras.layers.Conv3D(
            num_filters,
            (3, 3, 3),
            padding="same",
            activation="relu",
            input_shape=input_dim,
        )
    )
    model.add(
        tf.keras.layers.Conv3D(
            num_filters, (3, 3, 3), padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    return model


def M6_CNN_24F1D(lr, input_dim=(16, 16, 4), num_filters=8, num_classes=1):
    """Model 6: 24 filters, 3x3 kernel, 2x2 maxpooling

    Parameters
    ----------
    lr : float
        Learning rate

    input_dim : tuple
        Input shape

    num_filters : int
        Number of filters

    num_classes : int
        Number of classes

    Returns
    -------
    model : tf.keras.Sequential
        Model 6
    """
    model = Sequential(name="M6_CNN_24F1D")
    model.add(
        tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=input_dim,
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        tf.keras.layers.Conv2D(num_filters, (2, 2), padding="same", activation="relu")
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    return model


def M7_CNN_24F1D(lr, input_dim=(16, 16, 4), num_filters=12, num_classes=1):
    """Model 7: 24 filters, 3x3 kernel, 2x2 maxpooling, 128 neurons in dense layer

    Parameters
    ----------
    lr : float
        Learning rate

    input_dim : tuple
        Input shape

    num_filters : int
        Number of filters

    num_classes : int
        Number of classes

    Returns
    -------
    model : tf.keras.Sequential
        Model 7
    """
    model = Sequential(name="M7_CNN_24F1D")
    model.add(
        tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=input_dim,
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        tf.keras.layers.Conv2D(
            2 * num_filters, (3, 3), padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        tf.keras.layers.Conv2D(num_filters, (2, 2), padding="same", activation="relu")
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    return model
