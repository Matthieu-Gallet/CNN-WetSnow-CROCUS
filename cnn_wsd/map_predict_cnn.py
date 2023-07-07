from img_processing import extract_patches, reconstruct_image
from geo_tools import load_data, array2raster
from tensorflow import keras
from utils import *
import time


def validation_files(
    input_file, path_model, output_file, nbands, winsize, nagler=False
):
    img, geodata = load_data(input_file)
    t = time.time()
    patches = extract_patches(img, winsize, nbands)
    print("Shape of patches: ", patches.shape)
    print("Time to extract patches: ", time.time() - t)
    if "CNN" in path_model:
        bck = open_pkl(join(dirname(dirname(dirname(path_model))), "bckp_metric.pkl"))
        model = keras.models.load_model(path_model)
    else:
        bck = open_pkl(join(dirname(dirname(path_model)), "bckp_metric.pkl"))
        model = open_pkl(path_model)
    y_proba = model.predict(patches)

    # bck = transforme_BCK(bck)
    bckp = from_prediction_to_metric(bck)
    _, thr = best_threshold_FPRc_ROC(bckp, 0.05)
    ypred = np.where(y_proba > thr, 1, 0)
    reconstruct = reconstruct_image(ypred, img.shape[:2], winsize)
    array2raster(reconstruct, geodata, output_file[:-4] + "_BEST_FPRc__0_05.tif")

    _, thr = best_threshold_accuracy_ROC(bckp)
    ypred = np.where(y_proba > thr, 1, 0)
    reconstructACC = reconstruct_image(ypred, img.shape[:2], winsize)
    array2raster(reconstructACC, geodata, output_file[:-4] + "_BEST_acc.tif")

    ypred = np.where(y_proba > 0.5, 1, 0)
    reconstruct = reconstruct_image(ypred, img.shape[:2], winsize)
    array2raster(reconstruct, geodata, output_file[:-4] + "_05.tif")

    return 1


def validation_dataset(input_path, path_model, nbands, winsize, nagler=False):
    for file in glob.glob(input_path):
        if "CNN" in path_model:
            out_dir_r = join(dirname(dirname(dirname(path_model))), "validation")
        else:
            out_dir_r = join(dirname(dirname(path_model)), "validation")
        exist_create_folder(out_dir_r)
        model_n = basename(path_model).split(".")[0]
        output_file = join(out_dir_r, f"PRED_{model_n}" + basename(file))

        validation_files(file, path_model, output_file, nbands, winsize, nagler)
    return 1


if __name__ == "__main__":
    nbands = [0, 1, 6, 7]
    winsize = 16
    input_path = "../dataset_/select/*.tif"
    path_model = [
        "../results/CNN_KF/models/M_kf0_seed0/"
    ]  # [ "../results/RF_KF/models/M_kf1_seed0"]#.pkl"]
    for path in path_model:
        validation_dataset(input_path, path, nbands, winsize)
