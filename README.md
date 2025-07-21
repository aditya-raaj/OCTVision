# OCTVision - AI-Powered Retinal Disease Detection

## Overview

**OCTVision** is a web-based application that uses Artificial Intelligence (AI) to assist ophthalmologists and healthcare professionals in detecting retinal diseases from Optical Coherence Tomography (OCT) scans. The app automatically analyzes images to classify retinal conditions such as **Choroidal Neovascularization (CNV)**, **Diabetic Macular Edema (DME)**, **Drusen**, and **Normal Retina**.

The system utilizes a deep learning model based on **MobileNetV3Large** for efficient and accurate predictions, making it suitable for deployment in low-compute environments.

---

<div align="center">
  <img src="https://raw.githubusercontent.com/aditya-raaj/OCTVision/main/archive/home.png" alt="Project demo GIF" />
</div>

---

## Features

- **AI-Powered Disease Detection**: Automatically classify OCT images into four categories: CNV, DME, Drusen, and Normal Retina.
- **Real-Time Predictions**: Instant predictions based on uploaded images.
- **User-Friendly Interface**: Web interface for easy image uploads and result viewing.
- **Lightweight Model**: MobileNetV3 for high efficiency and fast predictions.
- **Responsive Design**: Access the app on desktop, tablet, or mobile devices.
- **Data Privacy**: Image data is processed locally with no external server uploads (unless specified).

<div align="center">
  <img src="https://raw.githubusercontent.com/aditya-raaj/OCTVision/main/archive/diagnose.gif" alt="Project demo GIF" />
</div>

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation](#training-and-evaluation)
5. [Deployment](#deployment)
6. [How to Use](#how-to-use)
7. [Retinal Diseases](#retinal-diseases)
8. [Future Work](#future-work)
9. [References](#references)

---

## Project Overview

OCTVision provides an automated solution for detecting retinal conditions using **Optical Coherence Tomography (OCT)** images. The system uses a MobileNetV3-based model to classify OCT images into one of the following categories:

- **Choroidal Neovascularization (CNV)**
- **Diabetic Macular Edema (DME)**
- **Drusen (Early AMD)**
- **Normal Retina**

The app is deployed as a **Flask-based web service** with a REST API endpoint for prediction.

---

## Dataset

- **Source**: [Mendeley Dataset](https://doi.org/10.17632/rscbjbr9sj.3)
- **OCT Images**: ~84,495 images, split into categories:
  - CNV: ~37,000 images
  - DME: ~11,500 images
  - Drusen: ~8,600 images
  - Normal: ~51,000 images
- **File Format**: JPEG
- **Resolution**: ~500x500 (resized to 224x224 for training)
- **Class Imbalance**: Handled using class weighting and balanced batch sampling.

---

## Model Architecture

<img src="https://github.com/aditya-raaj/OCTVision/blob/main/archive/MobileNetArch.png" alt="MobileNet" style="width: 100%;" />


**MobileNetV3Large** is used for efficient and accurate classification of OCT images. Key features of MobileNetV3 include:

- Depthwise separable convolutions
- Squeeze-and-excitation modules
- Hard-swish activations
- Efficient computation for real-time predictions

---

## Training and Evaluation

### Training Details

<img src="https://github.com/aditya-raaj/OCTVision/blob/main/archive/LossResult.png" alt="" style="width: auto; height: auto;" />

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (lr = 0.001)
- **Metrics**: Accuracy, F1-Score
- **Epochs**: 20
- **Batch Size**: 32
- **Hardware**: CPU, 8GB RAM laptop

<br>


### Final Metrics (Epoch 20/20)

- **Training Accuracy**: 99.30%
- **Validation Accuracy**: 97.26%
- **Training F1-Score**: 0.9980
- **Validation F1-Score**: 0.9912

  <img src="https://github.com/aditya-raaj/OCTVision/blob/main/archive/ConfustionMatrix.png" alt="" style="width: auto; height: auto;" />

---

## Deployment

The backend is built using **Flask**, and predictions are made via a REST API endpoint.

- **API Endpoint**: `/predict`
- Accepts a POST request with an image file (JPEG format).
- Returns the predicted class (CNV, DME, Drusen, Normal).

---

## How to Use

1. **Clone the Repository**:
   Clone the repository from GitHub and navigate into the project directory.

2. **Set up Dependencies**:
   Install the required Python libraries by running the following command:


```


pip install -r requirements.txt

```

3. **Start the Flask Server**:
Run the Flask app by executing:

```

python app.py

```



4. **Use the Web Interface**:
- Visit `http://localhost:5000` in your browser.
- Upload an OCT image to receive real-time predictions.

---

## Retinal Diseases

### 1. **Choroidal Neovascularization (CNV)**

- **Recommendation**: Immediate referral to a retinal specialist. CNV is often linked to age-related macular degeneration (AMD).
- **Treatment Options**: Anti-VEGF injections, Photodynamic Therapy, Laser Treatment.

<img src="https://github.com/aditya-raaj/OCTVision/blob/main/archive/CNV.jpeg" alt="CNV" style="width: 100%; height: auto;" />


### 2. **Diabetic Macular Edema (DME)**

- **Recommendation**: Coordination with an endocrinologist for diabetes management. Anti-VEGF injections are often the first-line treatment.
- **Treatment Options**: Anti-VEGF therapy, corticosteroid implants, laser therapy.


<img src="https://github.com/aditya-raaj/OCTVision/blob/main/archive/DME.jpeg" alt="DME" style="width: 100%; height: auto;" />

### 3. **Drusen (Early AMD)**

- **Recommendation**: Lifestyle modifications, including a diet rich in antioxidants. Regular OCT scans and potential use of AREDS2 supplements.
- **Treatment Options**: Dietary changes, supplements, smoking cessation.


<img src="https://github.com/aditya-raaj/OCTVision/blob/main/archive/DRUSEN.jpeg" alt="DRUSEN" style="width: 100%; height: auto;" />

### 4. **Normal Retina**

- **Recommendation**: Regular eye exams and maintenance of general health, including blood sugar and blood pressure control.


<img src="https://github.com/aditya-raaj/OCTVision/blob/main/archive/NORMAL.jpeg" alt="NORMAL" style="width: 100%; height: auto;" />


---

## Future Work

- **Model Improvements**: Integrating Grad-CAM or saliency maps for model interpretability.
- **Exploring Larger Models**: Testing Vision Transformers (ViT) on more powerful hardware.
- **Extended Features**: Adding database support, user management, and expanding to other ophthalmic conditions.

---

## References

1. [Mendeley Dataset](https://doi.org/10.17632/rscbjbr9sj.3)
2. Howard et al., "Searching for MobileNetV3", arXiv:1905.02244
3. [TensorFlow Documentation](https://www.tensorflow.org/)
4. [Keras Documentation](https://keras.io/)
5. [Flask Documentation](https://flask.palletsprojects.com/)
6. [Project Documentation](https://www.notion.so/Retinal-Image-Classification-Using-MobileNetV3Large-237c3fbad29b801fa2c2ddd5a520b058?source=copy_link/)

