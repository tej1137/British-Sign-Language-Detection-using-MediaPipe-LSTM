Overview
This project detects British Sign Language (BSL) fingerspelling alphabets using MediaPipe Holistic hand landmarks and an LSTM neural network, enabling real-time prediction from a webcam feed.
​
The key goal was to build a solution that is accurate while remaining lightweight enough for low-spec devices, after earlier CNN-based approaches struggled with generalisation and overfitting.
​
Key Features
Real-time webcam inference using OpenCV + MediaPipe landmarks (hands only).​

Landmark-based dataset creation: saved as .npy keypoint arrays organised by alphabet folders (A–Z).​

Sequence classification with an LSTM model (input sequence shape reported as (30, 126)).​

Evaluation using accuracy curves and a confusion matrix; real-time testing achieved strong performance with some difficult letters noted (e.g., R/M/N).
​
(Work-in-progress) GUI integration attempt using Kivy for a cross-platform interface.
​
Approach / Pipeline
Data collection: capture hand landmarks using MediaPipe Holistic and store sequences as NumPy arrays (.npy) per class.
​
Preprocessing: label-encode classes, load keypoints into sequences, concatenate left + right hand landmarks into a 1D feature vector, and prepare labels for training.
​

Model: train an LSTM-based Sequential network with dropout + batch normalization + dense layers, using softmax for multi-class classification.
​

Deployment: run webcam inference and predict letters using a configurable time gap so the model predicts after analysing a full sequence window.
​

Model Notes
Train/test split used a test size of 0.05 to maximise training exposure while keeping a small evaluation set.
​

The report notes very high training/testing accuracy (100%) after long training (320 epochs) and discusses over-training concerns and future improvements such as early stopping and validation monitoring.
​

Tech Stack
Python (project code), NumPy (.npy storage), OpenCV (webcam), MediaPipe Holistic (landmarks).
​

Deep learning stack implied by the report (LSTM Sequential model, dropout, batch norm, dense + softmax).​

Optional UI: Kivy (in progress).
​
How to Run (typical)
Because project folder structures differ, adapt these steps to your repo layout:
​
Create/activate a Python environment.
Install dependencies (OpenCV, MediaPipe, NumPy, and your DL framework).

Run your data collection script to generate .npy sequences.

Train the model.

Run the real-time detection script to predict letters from webcam.

Results (from report)
Strong offline evaluation was reported with only a few misclassifications in the confusion matrix.​

In real-time webcam testing, the model achieved a high proportion of correct predictions, but struggled more with certain similar hand shapes (R/M/N).​

Future Improvements
Add early stopping and use a validation set to control overfitting and tune learning rate.​
Complete the Kivy GUI integration for a usable end-user application.​
Explore model quantization to further reduce model size for low-spec deployment (while monitoring accuracy impact).
​
