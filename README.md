# 🩺 DermAssist – AI-Based Skin Lesion Analysis

**DermAssist** is an AI-powered skin lesion analysis prototype that combines classical image processing, computer vision, deep learning and explainable AI to analyse dermoscopic skin images.

The project aims to assist in preliminary skin lesion screening by performing image enhancement, lesion segmentation, ABCD feature extraction and binary classification using a transfer-learning model.

> **Disclaimer:** DermAssist is an academic and research prototype. It is not a medical diagnostic tool and must not replace professional dermatological consultation.

---

## 1. Project Overview

DermAssist follows a multi-stage image analysis pipeline:

1. **Image Preprocessing:** Removes unwanted hair structures and improves image contrast.
2. **Lesion Segmentation:** Uses Otsu thresholding and morphological operations to identify the lesion region.
3. **ABCD Feature Extraction:** Extracts asymmetry, border irregularity, colour variation and diameter-related features.
4. **Deep Learning Classification:** Uses MobileNetV2 transfer learning to classify lesions into benign and malignant project categories.
5. **Explainable AI:** Uses Grad-CAM to highlight image regions contributing to the model's prediction.
6. **Interactive Dashboard:** Provides an interface through Streamlit for uploading and analysing dermoscopic images.

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning and transfer learning |
| MobileNetV2 | Skin lesion classification |
| OpenCV | Image processing and segmentation |
| NumPy | Numerical operations |
| Pandas | Dataset management |
| Scikit-learn | Dataset splitting and evaluation |
| Matplotlib | Visualization |
| Streamlit | Interactive frontend |

## 3. Dataset

### HAM10000 – Human Against Machine with 10,000 Training Images

The project uses the HAM10000 dataset, consisting of approximately 10,015 dermoscopic images belonging to seven diagnostic categories.

The original diagnostic categories are:

| Label | Description |
|---|---|
| AKIEC | Actinic keratoses / intraepithelial carcinoma |
| BCC | Basal cell carcinoma |
| BKL | Benign keratosis-like lesions |
| DF | Dermatofibroma |
| MEL | Melanoma |
| NV | Melanocytic nevi |
| VASC | Vascular lesions |

### Binary Classification Mapping

For the current project implementation, the seven categories are grouped into two classes:

**Benign:**
- NV
- BKL
- DF
- VASC

**Malignant / Higher-risk project grouping:**
- MEL
- BCC
- AKIEC

This is an operational grouping used for the binary classification experiment. In particular, AKIEC includes both precancerous and carcinoma-in-situ conditions, so the grouping should not be interpreted as a definitive clinical diagnosis.

### Dataset Source

- [HAM10000 – Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- [Original Research Paper](https://www.nature.com/articles/sdata2018161)

**Note:** The dataset is not included in this GitHub repository because of its large storage requirements. Download it separately and keep the original images and metadata CSV on your local machine.

## 4. Project Structure

```text
DermAssist/
│
├── app.py
├── preprocess.py
├── prepare_data.py
├── train.py
├── gradcam.py
│
├── dermassist_model.keras
│
├── data/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   │
│   └── val/
│       ├── benign/
│       └── malignant/
│
├── requirements.txt
├── .gitignore
└── README.md
```

The `data/train` and `data/val` directories are generated automatically during dataset preparation.

## 5. Image Processing Pipeline

### A. Hair Removal

Hair structures can interfere with lesion analysis.

DermAssist uses:
- Blackhat morphological transformation to identify dark hair structures.
- Thresholding and dilation to generate a hair mask.
- OpenCV inpainting to reconstruct the affected regions.

### B. Contrast Enhancement

Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to the L channel of the LAB colour space to improve local contrast while limiting excessive amplification.

### C. Lesion Segmentation

Otsu thresholding is used to generate a binary lesion mask.

Morphological opening and closing help remove noise and improve the segmentation boundary. The largest suitable contour is selected as the lesion region.

### D. ABCD Feature Extraction

The system extracts four groups of features:

- **A – Asymmetry:** Estimates lesion asymmetry using principal-axis alignment and reflection comparisons.
- **B – Border:** Measures border irregularity using compactness and solidity.
- **C – Colour:** Measures colour variation using LAB colour statistics and K-means clustering.
- **D – Diameter:** Estimates lesion diameter in pixels and relative to image dimensions.

A simplified heuristic score is calculated from the four warning flags.

These measurements are experimental image-derived features, not validated clinical ABCD scores.

## 6. Deep Learning Model

### MobileNetV2 Transfer Learning

DermAssist uses MobileNetV2 as its binary skin lesion classification architecture.

The training process consists of two stages:

**Stage 1 – Feature Extraction**
- Load MobileNetV2 with ImageNet pretrained weights.
- Freeze the base network.
- Train the classification head.

**Stage 2 – Fine-Tuning**
- Unfreeze the upper layers of MobileNetV2.
- Keep Batch Normalization layers frozen.
- Fine-tune the model using a lower learning rate.

### Training Improvements

The implementation incorporates:
- Data augmentation.
- Class weighting to address class imbalance.
- Early stopping based on validation AUC.
- Accuracy, AUC and recall metrics.
- Classification report and confusion matrix.

The dataset is split by `lesion_id` to reduce data leakage between training and validation sets.

The current implementation uses a training and validation split. An independent external test dataset is still required for stronger generalization claims.

## 7. Explainable AI – Grad-CAM

DermAssist incorporates Gradient-weighted Class Activation Mapping (Grad-CAM).

Grad-CAM generates a heatmap showing image regions that contribute to the model's malignant-class prediction.

The heatmap is overlaid on the original dermoscopic image to make the model's visual attention easier to inspect.

This provides interpretability but does not establish that the prediction is medically correct.

## 8. Installation and Setup

### Prerequisites

- Python 3.11–3.13 with compatible TensorFlow installation.
- Jupyter Notebook / JupyterLab (optional).
- HAM10000 dataset.
- Internet access for downloading pretrained ImageNet weights.

### Step 1 – Clone the repository

```bash
git clone https://github.com/SparshAgrawal64/DermAssist.git
```

```bash
cd DermAssist
```

### Step 2 – Install dependencies

```bash
pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib streamlit
```

### Step 3 – Download HAM10000

Download and extract the HAM10000 dataset.

Keep the two image folders and `HAM10000_metadata.csv` together in a directory on your computer.

For example:

```text
D:/Datasets/HAM10000/
│
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_metadata.csv
```

The dataset directory should be outside the generated `DermAssist/data/` directory because dataset preparation recreates its output directory.

### Step 4 – Prepare the dataset

Run:

```bash
python prepare_data.py --raw "D:/Datasets/HAM10000"
```

The script will:

- Locate the images and metadata.
- Match image IDs with diagnostic labels.
- Convert seven diagnostic categories into two project classes.
- Split the data by lesion ID.
- Limit the number of benign training images for faster experimentation.
- Generate the training and validation folders inside `data/`.

**Important:** The preparation script deletes and recreates its output directory. Do not place your original dataset inside that output directory.

### Step 5 – Train the model

Run:

```bash
python train.py
```

The model will train in two phases:

1. Classification-head training.
2. Fine-tuning of the upper MobileNetV2 layers.

The trained model is saved as:

```text
dermassist_model.keras
```

The first run may download pretrained ImageNet weights.

### Step 6 – Launch the Streamlit application

Run:

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

Upload a dermoscopic image to explore the analysis pipeline.

## 9. Streamlit Application Features

The application provides three main analysis sections:

**1. Preprocessing and Segmentation**
- Original image.
- Hair-removed image.
- CLAHE-enhanced image.
- Otsu segmentation mask.
- Lesion boundary overlay.

**2. ABCD Analysis**
- Asymmetry measurement.
- Border compactness.
- Colour variation.
- Diameter estimation.
- Experimental ABCD warning score.

**3. AI Classification and Grad-CAM**
- Benign or malignant project-category prediction.
- Adjustable classification threshold.
- Malignant-class score.
- Grad-CAM visualization.

Image-processing and ABCD analysis can be used without a trained classification model. Classification and Grad-CAM require `dermassist_model.keras`.

## 10. Limitations and Future Scope

### Current Limitations

- Otsu segmentation may fail on images with poor contrast or complex lesion boundaries.
- ABCD thresholds are heuristic and are not clinically validated.
- Diameter is measured in pixels because HAM10000 does not provide a consistent physical image scale.
- The classifier uses a binary grouping of the original seven categories.
- Validation performance does not establish clinical reliability.
- The current implementation does not include a U-Net segmentation model or an independent test-set evaluation.

### Future Enhancements

- Implement U-Net-based lesion segmentation using ground-truth segmentation masks.
- Compare Otsu and U-Net segmentation using Dice and IoU.
- Compare MobileNetV2 with ResNet50.
- Add sensitivity, specificity, ROC-AUC and additional evaluation metrics.
- Improve explainability with additional visualization techniques.
- Validate performance on independent dermoscopic datasets.
- Improve dataset balancing and generalization.

## 11. Team Contributions

DermAssist is developed as a collaborative academic project by a team of three members.

The project work is organized into three major areas:

| Member | Responsibilities |
|---|---|
| Member 1 | Image preprocessing, hair removal, CLAHE, Otsu segmentation and ABCD feature extraction |
| Member 2 | Dataset preparation, MobileNetV2 training and model evaluation |
| Member 3 | Grad-CAM explainability, Streamlit frontend and project integration |

## 12. Ethical and Medical Disclaimer

DermAssist is developed exclusively for educational, experimental and research purposes.

Its predictions and image-derived measurements are not medical diagnoses. The model may produce incorrect predictions, including false negatives and false positives.

Users should consult qualified dermatologists for the assessment of skin lesions or any concerning changes.

---

**Developed as an academic project – DermAssist Team**

GitHub: [SparshAgrawal64/DermAssist](https://github.com/SparshAgrawal64/DermAssist)
