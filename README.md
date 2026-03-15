# Real-Time Inference Server & Physiological Data Pipeline

This repository contains the Python-based backend infrastructure for the real-time cognitive state inference engine. It handles multi-modal data synchronization, digital signal processing (DSP), and deep learning predictions via a Flask-based REST API.

## 🖥️ Inference Server (`app.py`)

The central hub of the experiment is a Flask server that processes incoming data streams and returns a model-driven difficulty prediction to Unity.

### Core Processing Pipelines



#### 1. Eye-Tracking Analysis
The server processes raw eye-tracking data using a sophisticated pipeline:
* **Preprocessing:** Savitzky-Golay filtering for signal smoothing.
* **Blink Management:** Automated blink detection and linear interpolation with safety margins.
* **Feature Engineering:** Calculation of saccadic velocity, acceleration, and pupil constriction/dilation dynamics.
* **HMM Classification:** A **Hidden Markov Model (HMM)** is utilized to classify eye movements into fixations and saccades.

#### 2. Physiological Signal Processing
Heart rate and skin conductance data are processed to extract autonomic nervous system markers:
* **GSR (EDA):** Extraction of tonic and phasic components, mean conductance, and linear slope.
* **PPG:** Bandpass filtering ($0.5–3.0$ Hz) and peak enhancement using `HeartPy` to calculate **BPM, SDNN, RMSSD, and pNN50**.

#### 3. Deep Learning Inference
The server implements an **LSTM Classifier** with an **Attention Layer**:
* **Windowing:** Data is segmented into 4 sequential temporal windows (time steps).
* **Attention Mechanism:** Focuses on specific windows that are most predictive of cognitive load.
* **Prediction:** Returns a difficulty class and associated confidence probabilities.

---

## ⌚ Physiological Acquisition (`GSR_PPG.py`)

To feed the server, raw data must be acquired from the hardware layer.

* **Background Streaming:** This script must run in the background alongside the Unity experiment.
* **Shimmer Integration:** It connects to Shimmer3 devices via Bluetooth, converts raw ADC values into meaningful units (Ohms for GSR), and pushes samples to **Lab Streaming Layer (LSL)**.
* **Usage:** Ensure the `COM` port matches your Shimmer Bluetooth pairing.

---

## 📂 Model & Scalers

Model assets are stored in the `/model_data` folder. These files are loaded at server startup:
* `model.dill`: The serialized PyTorch LSTM model.
* `scaler.dill`: Feature normalization parameters.
* `config.json`: Hyperparameters and feature mapping.

> [!IMPORTANT]  
> The provided models and scalers are **examples**. Researchers are encouraged to use their own methods for training models and processing data specific to their experimental paradigms.

---

## 🕹️ API Reference

### Upload GSR/PPG
`POST /upload-gsr`  
Buffers raw physiological samples for synchronization.

### Process Trial & Predict
`POST /upload-eye`  
Receives the trial's eye-tracking data, triggers the physiological buffer lookup, and returns the AI prediction.

---

## 💾 Citation

If you use this server architecture or the processing scripts in your research, please cite the following publication:

> D. Szczepaniak, M. Harvey, and F. Deligianni, "Your Eyes Controlled the Game: Real-Time Cognitive Training Adaptation based on Eye-Tracking and Physiological Data in Virtual Reality," *arXiv preprint arXiv:2512.17882*, 2026. [https://arxiv.org/abs/2512.17882](https://arxiv.org/abs/2512.17882)

---

## 🛠️ Requirements
* Python 3.8+
* Flask, PyTorch, heartpy, hmmlearn, dill, scipy, pandas
