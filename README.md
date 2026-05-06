# Word-Level-American-Sign-Language
An AI-powered end-to-end sign language recognition pipeline utilizing MediaPipe Holistic and a Bidirectional LSTM (Bi-LSTM) and GRU and CNN network to classify 100 American Sign Language (ASL) words.
AI-Powered Solutions for Sign Language Recognition
Project OverviewThis project presents an end-to-end sign language recognition system designed to bridge the communication gap for individuals with hearing and speech impairments. By translating sign language manual movements, gestures, and facial expressions into text or speech, the system aims to improve social integration. The project was developed in 2026 at the Syrian Private University (SPU).
 DatasetSource: The model uses the WLASL100 (Word-Level American Sign Language) dataset sourced from Kaggle.
Scope: It includes 100 different ASL word classes.
Format: The dataset consists of video-based formats featuring diverse signers in real-world recording environments.
Methodology & PipelineData Processing:
Video frames are normalized and resized while preserving temporal sequences. Numeric class labels are created using a LabelEncoder.
Feature Extraction: The system uses MediaPipe Holistic to extract skeletal landmarks, ensuring privacy and processing efficiency. 
It extracts a compact 258-dimensional feature vector per frame, which includes 33 pose points, 21 left-hand points, and 21 right-hand points.  Data Augmentation: To improve model generalization and robustness, the pipeline incorporates rotation, scaling, flipping, time warping, and random frame dropping. 
Model ArchitectureCore Model:
A 3-layer Bidirectional LSTM (Bi-LSTM) is used to capture dynamic movement patterns and temporal dependencies in both forward and backward directions.
Pooling: The architecture utilizes Mean Pooling to average LSTM outputs across all frames, capturing richer temporal information than a last-frame approach.  Classifier Head: The sequence classifier reduces dimensions progressively using linear layers, ReLU activations, and heavy dropout (0.1) for regularization.
 ResultsThe model successfully achieved a 62% Validation Accuracy on the WLASL100 dataset.  This result validates the end-to-end pipeline and demonstrates that the MediaPipe + Bi-LSTM foundation works correctly on complex, real-world video data. 
  Future WorkIncrease dataset size and diversity to improve accuracy.
  Experiment with advanced AI research, including Attention Mechanisms and Transformer models.  Explore Graph Convolutional Networks (GCNs) to better understand the natural connections between hand joints.
  Extend the system to support continuous sign language recognition rather than just isolated words.
  Implement bidirectional Text-to-Sign translation.  
  CreditsStudents/Developers: Zain Mahfouz, Ahmad Al Tayar, Raneem Omran. 
  Supervisors: DR. Majida Albakour, Eng. Farah Fares. 
  Institution: Faculty of Artificial Intelligence, Syrian Private University (SPU).
