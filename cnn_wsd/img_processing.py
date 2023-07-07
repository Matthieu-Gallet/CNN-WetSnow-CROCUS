import numpy as np


def extract_patches(image, patch_size, nbands):
    """Extract patches from an image

    Parameters
    ----------
    image : numpy array
        Image to extract patches from

    patch_size : int
        Size of the patches (square)

    nbands : int
        Number of bands to select if the image is multiband

    Returns
    -------
    numpy array
        Array of patches
    """
    win2 = int(patch_size / 2)  # ?
    patches = []
    for i in range(win2, image.shape[0] - win2):
        for j in range(win2, image.shape[1] - win2):
            if win2 % 2 == 0:
                patch = image[i - win2 : i + win2, j - win2 : j + win2, nbands]
            else:
                patch = image[i - win2 : i + win2 + 1, j - win2 : j + win2 + 1, nbands]
            patches.append(patch)
    return np.array(patches, dtype=np.float32)


def reconstruct_image(labels, image_size, patch_size):
    """Reconstruct an image from patches

    Parameters
    ----------
    labels : numpy array
        Array of labels of the patches

    image_size : tuple
        Size of the image to reconstruct

    patch_size : int
        Size of the patches (square)

    Returns
    -------
    numpy array
        Reconstructed image
    """
    win2 = int(patch_size / 2)
    reconstruct = np.zeros(image_size)
    count = 0
    for i in range(win2, image_size[0] - win2):
        for j in range(win2, image_size[1] - win2):
            reconstruct[i, j] = labels[count]
            count += 1
    return reconstruct[:, :, np.newaxis]
