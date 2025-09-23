# HDNetELM: A Hybrid Face Detection Framework for Highly Occluded Faces

## Description
This repository contains the implementation of HDNetELM, a hybrid face detection model designed for highly occluded faces. The pipeline integrates:

- **DenseNet121** for deep feature extraction.

- **Histogram of Oriented Gradients (HOG)** for capturing edge/texture information.

- **Extreme Learning Machine (ELM)** as a lightweight and efficient classifier.

- **Canny** edge detection for object proposals.

The model was evaluated on the Niqab dataset (faces with heavy coverings such as niqabs/veils) and a subset of the COCO dataset (non-face images), showing robust performance in detecting occluded faces.

---

## 🔑 Key Features
- Hybrid feature extraction (Deep + Handcrafted)
- Very fast training with ELM
- Automated **Grid Search** optimization
- Detailed metrics and logs (Excel, pickle, .h5 models)
- Scalable to large datasets

---

## Dataset Information
* **Niqab Dataset**: Custom dataset containing ~10,000 images with niqab- and veil-covered faces. DOI/Link: [https://doi.org/10.5281/zenodo.17011207]

* **COCO Dataset**: Publicly available Common Objects in Context dataset. URL: https://cocodataset.org

* **Preprocessing**:

-- Images resized to **224×224 pixels**.

-- Contextual labeling used for occluded faces (bounding boxes include surrounding regions).

-- Balanced with non-face COCO subset to reduce false positives.

* 📂 **Dataset Structure**

Organize your dataset like this:

```
dataset/
├── final_dataset/       # Training data
│   ├── class1/
│   └── class2/
├── validation/          # (Optional) validation data
│   ├── class1/
│   └── class2/
└── Test/                # Testing data
    ├── faces/
    └── no face/
```

---

## 📊 Workflow Pipeline

```
Input Image
      │
      ▼
Preprocessing (resize 224×224, normalize)
      │
 ┌────┴─────┐
 ▼          ▼
DenseNet121 HOG Features
Deep Features Texture Features
 └────┬─────┘
      ▼
Feature Concatenation
      ▼
Standardization (Z-score)
      ▼
Extreme Learning Machine (ELM)
   ├── Grid Search
   └── Best Model
      ▼
Evaluation (Acc, Precision, Recall, F1, Loss, Time)
      ▼
Results saved → Excel
```

---

## ⚙️ Installation Requirements

Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

---
## 🧾 Code Information

1] ***cnn_baselines_training.py*** [*comparative training and evaluation of standard CNN architectures*]

Trains and evaluates multiple CNN baselines (VGG16, VGG19, ResNet50, ResNet101, ResNet152, InceptionV3, DenseNet121, EfficientNetB0, NASNetLarge) on the Niqab + COCO dataset. 

**Usage**
```bash
python cnn_baselines_training.py
```
**Outputs**
- Models: models/{ModelName}-{run}.h5
- Histories: models/history{ModelName}-{run}.pkl
- Metrics (accuracy, precision, recall, F1, loss, training & testing time) in Excel format:
  - excels/training_metrics_{ModelName}_5_runs_train.xlsx
  - excels/training_metrics_{ModelName}_5_runs_test.xlsx
**Notes**
- Runs 5 independent experiments per model.
- Uses Adam optimizer (lr=0.0001), batch size 32, early stopping.
- Models are initialized with ImageNet weights and fine-tuned with a custom FC head.

2] ***elm_dense_features_baseline.py*** [*Implements a DenseNet121 feature extractor + custom ELM classifier baseline*]

Trains an ELM classifier on DenseNet121 embeddings extracted from the Niqab/COCO dataset splits. Produces 5 independent runs and logs metrics in Excel. 

**Usage**
```bash
python elm_dense_features_baseline.py
```
**Outputs**
- ModelELM/testing_metrics_ELM_5_runs_train.xlsx
- ModelELM/testing_metrics_ELM_5_runs_test.xlsx

**Key Settings**
- Feature extractor: DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))
- Image size: 224×224, rescaled to 1/255
- ELM: hidden units = 100 (default), sigmoid activation
- Metrics: Accuracy, Precision, Recall, F1, Binary Cross-Entropy Loss; Confusion Matrix
- Runs: 5 independent repetitions

3] ***hdnetelm_gridsearch.py*** [*The full HDNetELM pipeline*]

Runs the HDNetELM pipeline (DenseNet121 features + HOG features + ELM classifier) with grid search over ELM hyperparameters, repeated for 5 runs. 

**Usage**
```bash
python hdnetelm_gridsearch.py
```
**What it does**
1. Extracts DenseNet121 feature maps and HOG features per image.
2. Concatenates features, standardizes them, then trains an ELM classifier.
3. Performs a manual grid search over:
```python
param_grid = {
    'n_neurons': list(range(100, 1000, 50)),  # Hidden neurons
    'activation': ['sigm'],                   # Activation function
    'rp': [0.01, 0.1, 1, 10, 100, 1000]       # Regularization parameter
}
```

- `n_neurons`: 100 → 950 (step 50)  
- `activation`: `'sigm'` (sigmoid)  
- `rp`: `[0.01, 0.1, 1, 10, 100, 1000]`  
4. Repeats for 5 runs and saves the best model metrics per run.

**Outputs**
- Excel summary: ModelELM/testing_metrics_ELM_5_runs_train_GridSearch.xlsx
Columns include train/test: accuracy, precision, recall, F1, log-loss, time, plus selected n_neurons, activation, RP.
**Notes**
- Uses hpelm for the ELM classifier.
- Requires scikit-image for HOG, OpenCV for image ops, and TensorFlow/Keras for DenseNet121.
- Keep test generator shuffle=False so predictions align with labels.

4] ***hdnetelm_region_proposal_eval.py*** [*Testing of previously trained hybrid models*]

Evaluates pretrained hybrid models (CNN backbone features + HOG features → classifier) on the Test split using Canny-based region proposals. Aggregates ROI predictions into image-level results and writes metrics to Excel.

**Usage**
```bash
python hdnetelm_region_proposal_eval.py
```

**Outputs**
- Excel: xlsxFiles/Testing_metrics_{Backbone}_runs_test.xlsx with Accuracy, Precision, Recall, F1, BCE Loss, Testing Time, Confusion Matrix.

**Notes**
- Requires the pretrained files named: CombinedModel_hog{Backbone}_cnn{1..5}_face_detector.h5.
- Uses Canny edges → contour bboxes (min area filter) → ROI extraction → CNN+HOG features → model.predict.
- Aggregation: average of ROI predictions; threshold decides final image label (default 0.65).
- 
## ▶️ Quickstart Usage

### 1. Prepare datasets in the above structure 
### 2. Train CNN baselines.
```bash
python cnn_baselines_training.py
```
### 3. Run HDNetELM with grid search:
Run the main hybrid pipeline with grid search over ELM parameters:
```bash
python hdnetelm_gridsearch.py
```
### 4. Optional: Train DenseNet+ELM baseline:
```bash
python elm_dense_features_baseline.py
```
### 5. Evaluate/Test pretrained models:
```bash
python hdnetelm_region_proposal_eval.py
```
### 6. Check Outputs:
   - Trained models in models/
   - Excel metrics in excels/, ModelELM/, or xlsxFiles/`

---
## 🧪 Methodology & Reproducibility
- **Training setup**: all CNN models initialized with ImageNet weights, frozen conv layers, custom FC head, trained with Adam (lr=0.0001), batch=32, max 25 epochs, early stopping on val_acc.
- **Evaluation**: 5 independent runs per experiment; AVG±STD reported.
- **Statistical tests**: Wilcoxon rank-sum test (α=0.05), mean-rank analysis.
- **Computing environment**: Ubuntu Linux, NVIDIA H100 GPU (80GB), 20 vCPUs, 240 GB RAM
- **Reproducibility**:
  - Datasets: Niqab dataset (Zenodo DOI), COCO subset.
  - Code: this repo + Zenodo archive.
---

## 📈 Example Results

| n_neurons | RP   | Activation | Train Acc | Test Acc | Precision | Recall | F1   | Loss | Train Time (s) | Test Time (s) |
|-----------|------|------------|-----------|----------|-----------|--------|------|------|----------------|---------------|
| 100       | 0.01 | sigm       | 0.923     | 0.915    | 0.918     | 0.910  | 0.914| 0.23 | 1.24           | 0.03          |
| 300       | 0.1  | sigm       | 0.940     | 0.931    | 0.934     | 0.928  | 0.931| 0.19 | 1.56           | 0.04          |
| 500       | 1    | sigm       | 0.951     | 0.947    | 0.950     | 0.945  | 0.947| 0.14 | 2.02           | 0.05          |
| 750       | 10   | sigm       | 0.958     | 0.952    | 0.954     | 0.951  | 0.952| 0.11 | 2.34           | 0.06          |
| 950       | 100  | sigm       | 0.960     | 0.954    | 0.956     | 0.952  | 0.954| 0.10 | 2.89           | 0.07          |

> ⚠️ Values are **illustrative only** – actual results depend on your dataset.

---

## 🏆 Example Best Model
- `n_neurons`: 750  
- `activation`: sigm  
- `rp`: 10  
- Training Accuracy: 0.958  
- Testing Accuracy: 0.952  
- Precision: 0.954  
- Recall: 0.951  
- F1: 0.952  
- Loss: 0.11  
- Training Time: ~2.3s  
- Testing Time: ~0.06s  

---

## 📊 Output

All scripts save results as Excel files in `ModelELM/`.  
Metrics include:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Loss  
- Training & Testing Time (seconds)  
- Confusion Matrix  

---

## 📌 Notes
- Update dataset paths if your dataset is stored elsewhere.
- Ensure `ModelELM/` directory exists, otherwise Excel saving will fail.

---

## ✨ Citation
If you plan to use this repository or the Niqab dataset, please include the following citations:

```bibtex
@article{ALASHBI2025101893,
  title   = {Human face localization and detection in highly occluded unconstrained environments},
  journal = {Engineering Science and Technology, an International Journal},
  volume  = {61},
  pages   = {101893},
  year    = {2025},
  issn    = {2215-0986},
  doi     = {https://doi.org/10.1016/j.jestch.2024.101893},
  url     = {https://www.sciencedirect.com/science/article/pii/S2215098624002799},
  author  = {Abdulaziz Alashbi and Abdul Hakim H.M. Mohamed and Ayman A. El-Saleh and Ibraheem Shayea and Mohd Shahrizal Sunar and Zieb Rabie Alqahtani and Faisal Saeed and Bilal Saoud}
}

@article{Alashbi2022,
  author    = {A. Alashbi, Abdulaziz and Sunar, Mohd Shahrizal and Alqahtani, Zieb},
  title     = {Deep Learning CNN for Detecting Covered Faces with Niqab},
  journal   = {Journal of Information Technology Management},
  volume    = {14},
  number    = {Special Issue: 5th International Conference of Reliable Information and Communication Technology (IRICT 2020)},
  pages     = {114--123},
  year      = {2022},
  publisher = {Univrsity Of Tehran Press},
  issn      = {2980-7972},
  eissn     = {2980-7972},
  doi       = {10.22059/jitm.2022.84888},
  url       = {https://jitm.ut.ac.ir/article_84888.html},
  eprint    = {https://jitm.ut.ac.ir/article_84888_a3b5d00476d6628dea08b1dcf27a9c27.pdf}
}

@InProceedings{10.1007/978-3-030-33582-3_20,
  author    = {Alashbi, Abdulaziz Ali Saleh and Sunar, Mohd Shahrizal},
  editor    = {Saeed, Faisal and Mohammed, Fathey and Gazem, Nadhmi},
  title     = {Occluded Face Detection, Face in Niqab Dataset},
  booktitle = {Emerging Trends in Intelligent Computing and Informatics},
  year      = {2020},
  publisher = {Springer International Publishing},
  address   = {Cham},
  pages     = {209--215},
  isbn      = {978-3-030-33582-3}
}


## 📜 License
This repository is released as open access:

- **Code**: Licensed under the [MIT License](LICENSE), allowing free use, modification, and redistribution with attribution.  
- **Dataset (Niqab Dataset)**: Licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the dataset as long as appropriate credit is given.

## 🤝 Contribution Guidelines
This repository is not open for external contributions.  
If you have questions, feedback, or issues, please open an **Issue** in the GitHub tracker.  


