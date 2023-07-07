# CNN-WetSnow-CROCUS

## 1. Description
The project explore the capacity of Convolutional Neural Network (CNN) and Random Forest (RF) algorithms to detect wet snow in SAR (Sentinel-1) images. The dataset used can be found in: [zenodo.org/record/8111485](zenodo.org/record/8111485).
This work will be presented at IGARSS 2023, the poster can be found in the folder `poster`.
## 2. Structure

```bash
.
├── README.md
├── cnn_wsd
│   ├── __init__.py
│   ├── architecture.py
│   ├── dataset_loader.py
│   ├── evaluate_cnn.py
│   ├── explore_cnn.py
│   ├── geo_tools.py
|   ├── img_processing.py
│   ├── map_predict_cnn.py
│   ├── metrics.py
│   ├── model_selection.py
│   └── utils.py
└── poster
    └── POSTER_IGARSS_23_V4.pdf
```
