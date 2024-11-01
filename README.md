# Attention-Based Convolutional Neural Network for Image Classification

This project implements a Convolutional Neural Network (CNN) with attention mechanisms using Keras to classify images as either class A or B. The attention mechanism helps the model focus on important parts of the images, improving classification accuracy. It uses `ImageDataGenerator` for data augmentation and includes visualization tools for training progress, test results, and the model's attention regions.

## Features
- **Data Augmentation**: Uses Keras `ImageDataGenerator` for transformations (rescaling, shearing, zooming, and flipping).
- **Attention Mechanism**: Includes attention layers to improve focus on important image areas.
- **Callbacks**: Uses `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping` for optimized training.
- **Visualization**: Displays sample images with labels, and plots training/validation loss and accuracy.

## How It Works
1. **Data Preparation**: Loads training, validation, and test images using `ImageDataGenerator`.
2. **Model Architecture**: Builds a CNN with added attention layers to weigh important image regions.
3. **Training**: Compiles and trains the model with a sparse categorical cross-entropy loss function.
4. **Evaluation & Visualization**: After training, the model is evaluated on the test set, and predictions are visualized.

## Requirements
- Python 3.x
- TensorFlow/Keras
- Matplotlib
