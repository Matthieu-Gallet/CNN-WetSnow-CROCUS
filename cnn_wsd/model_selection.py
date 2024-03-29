import numpy as np


class BFold:
    """Balanced Fold cross-validator, only valid for binary classification.

    Split dataset into k consecutive folds (without shuffling by default),
    ensuring that each fold has the same number of samples from each class.
    It allows to have B sub-datasets balanced, and have a complete view of the
    data.

    Parameters
    ----------
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, default=None
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Attributes
    ----------
    n_splits : int
        Returns the number of splitting iterations in the cross-validator.

    """

    def __init__(self, shuffle=False, random_state=42):
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_state)

    def get_n_splits(self, X, y, groups=None):
        argmin_minor = np.argmin(np.unique(y, return_counts=True)[1])
        minority_numb = np.unique(y, return_counts=True)[1][argmin_minor]
        majority_numb = np.max(np.unique(y, return_counts=True)[1])
        minority_class = np.unique(y, return_counts=True)[0][argmin_minor]
        self.n_splits = int(majority_numb / minority_numb)
        return self.n_splits

    def split(self, X, y, groups=None):
        """Generate indices to split data into training set.

        Parameters
        ----------
        X : numpy array
            dataset of images

        y : numpy array
            dataset of labels        test : numpy array
            The testing set indices for that split.
        ------
        train : numpy array
            The training set indices for that split.
        """
        argmin_minor = np.argmin(np.unique(y, return_counts=True)[1])
        minority_numb = np.unique(y, return_counts=True)[1][argmin_minor]
        majority_numb = np.max(np.unique(y, return_counts=True)[1])
        minority_class = np.unique(y, return_counts=True)[0][argmin_minor]
        ratio = majority_numb / minority_numb
        if majority_numb % minority_numb == 0:
            self.n_splits = int(ratio)
        else:
            self.n_splits = int(ratio) + 1

        idx_minority = np.where(y == minority_class)[0]
        idx_majority = np.where(y != minority_class)[0]

        if self.shuffle:
            self.rng.shuffle(idx_majority)

        for i in range(self.n_splits):
            start = i * minority_numb
            end = (i + 1) * minority_numb
            if i == self.n_splits - 1:
                end = len(idx_majority)
                miss = minority_numb - (end - start)
                idx_majority_balanced = np.concatenate(
                    [idx_majority[start:end], idx_majority[:miss]]
                )
            else:
                idx_majority_balanced = idx_majority[start:end]

            idx_train = np.concatenate([idx_minority, idx_majority_balanced])
            self.rng.shuffle(idx_train)
            yield idx_train
