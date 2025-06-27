# Crime-Detection-and-Alert-System
 markdown
# UCF-Crime Violence Detection using CNN-LSTM

This project implements a deep learning model for detecting violent or anomalous events in video sequences, leveraging the UCF-Crime dataset. The core architecture combines a Convolutional Neural Network (CNN) for spatial feature extraction with a Long Short-Term Memory (LSTM) network for capturing temporal dependencies.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Automatic violence detection in surveillance footage is a critical application of computer vision and deep learning. This project aims to build a robust model capable of identifying anomalous activities, such as various forms of violence, by analyzing sequences of video frames. The model processes individual frames using a CNN and then analyzes the temporal evolution of these features using an LSTM, classifying the sequence as either "normal" or "anomalous/violent".

## Dataset

The model is trained and evaluated on the **UCF-Crime Dataset**.
* **Source**: The dataset used here is accessed via KaggleHub, specifically `odins0n/ucf-crime-dataset`.
* **Content**: This specific version of the dataset contains video frames pre-extracted as **PNG image sequences**. These frames are organized into `Train` and `Test` splits, with various anomaly types (e.g., `Abuse`, `Arrest`, `Assault`) and `Normal` or `NormalVideos` folders containing non-anomalous sequences.
* **Structure**:
    ```
    ucf-crime-dataset/
    ├── Test/
    │   ├── Abuse/
    │   │   ├── Abuse001_x264/  (directory containing PNG frames for a video)
    │   │   └── ...
    │   ├── Arrest/
    │   │   └── ...
    │   ├── ... (other anomaly types)
    │   └── NormalVideos/    (directory containing subdirectories of PNG frames)
    │       └── Nomal_001_x264/
    │           └── ...
    └── Train/
        ├── Abuse/
        │   ├── Abuse001_x264/
        │   └── ...
        ├── ... (other anomaly types)
        └── Normal/          (directory containing subdirectories of PNG frames)
            └── Nomal_001_x264/
                └── ...
    ```
    *Note: The 'Normal' folder in the `Train` split is named `Normal`, while in the `Test` split it is named `NormalVideos`. Both contain subdirectories with image sequences.*

## Model Architecture

The deep learning model is a **CNN-LSTM** hybrid.

* **Feature Extractor (CNN)**:
    * **Model**: A pre-trained **MobileNetV2** is used as the backbone CNN.
    * **Purpose**: Extracts rich spatial features from each individual frame in the input sequence.
    * **Transfer Learning**: The MobileNetV2 is initialized with weights pre-trained on the ImageNet dataset (`weights='imagenet'`). Its convolutional base layers are **frozen** (`base_cnn.trainable = False`) to leverage powerful pre-learned features without further training, saving computational resources.
    * **`TimeDistributed`**: The `MobileNetV2` model is wrapped in a `TimeDistributed` layer to ensure it processes each frame of the input sequence independently.

* **Sequence Processor (LSTM)**:
    * **Model**: A Long Short-Term Memory (LSTM) layer.
    * **Purpose**: Analyzes the temporal dependencies and patterns across the sequence of features extracted by the CNNs from each frame. This allows the model to understand the progression of events over time, crucial for violence detection.
    * **Output**: The LSTM aggregates the temporal information and outputs a single vector summarizing the sequence (`return_sequences=False`).

* **Classification Head**:
    * Consists of `Dense` layers with `relu` activation, followed by a final `Dense` layer with `sigmoid` activation for binary classification (anomaly vs. normal).
    * `Dropout` layers are strategically placed to prevent overfitting.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/ucf-crime-violence-detection.git](https://github.com/your-username/ucf-crime-violence-detection.git)
    cd ucf-crime-violence-detection
    ```

2.  **Create a Conda environment (recommended)**:
    ```bash
    conda create -n ucf_crime python=3.9
    conda activate ucf_crime
    ```

3.  **Install dependencies**:
    ```bash
    pip install tensorflow opencv-python numpy scikit-learn matplotlib tqdm kagglehub
    ```
    *Note: Ensure you have a compatible TensorFlow version (e.g., `tensorflow-gpu` if you have an NVIDIA GPU and CUDA setup).*

## Usage

This project is designed to be run primarily within a Kaggle Notebook environment due to the dataset access via `kagglehub`.

1.  **Kaggle Notebook Setup**:
    * Create a new Kaggle Notebook.
    * Go to "Data" -> "Add Data" and search for "UCF-Crime Dataset" by `odins0n`. Add it to your notebook. This will make the dataset available at `/kaggle/input/ucf-crime-dataset`.
    * Ensure you have a GPU accelerator enabled in your Kaggle Notebook settings (Runtime -> Change runtime type -> GPU).

2.  **Copy the code**: Copy the entire Python code from `ucf_crime_violence_detector_training.py` (or directly from the provided full code) into a cell in your Kaggle Notebook.

3.  **Run the notebook**: Execute all cells in the Kaggle Notebook. The training process will start automatically.

## Training Details

* **Input Shape**: Sequences of 16 frames (`SEQUENCE_LENGTH`), each resized to 128x128 pixels (`FRAME_HEIGHT`, `FRAME_WIDTH`) with 3 color channels (RGB).
* **Batch Size**: `BATCH_SIZE = 8` sequences per batch. This may be adjusted based on GPU memory.
* **Epochs**: `EPOCHS = 30`.
* **Optimizer**: Adam optimizer with a learning rate of `0.0001`.
* **Loss Function**: `binary_crossentropy` for binary classification.
* **Metrics**: Accuracy, Precision, and Recall.
* **Callbacks**:
    * `ModelCheckpoint`: Saves the best model weights based on `val_loss`.
    * `EarlyStopping`: Stops training if `val_loss` does not improve for 7 epochs (`patience=7`).
    * `ReduceLROnPlateau`: Reduces learning rate by a factor of 0.5 if `val_loss` does not improve for 3 epochs (`patience=3`).
* **Data Generation**: Custom `ImageSequenceDataGenerator` (inheriting from `tf.keras.utils.Sequence`) efficiently loads image sequences on-the-fly, handling frame extraction, resizing, normalization, and sequence creation, including padding for short sequences. It ensures `float32` data types to prevent memory issues.

## Results

After training, the notebook will output training and validation accuracy/loss plots, and the best performing model will be saved as `violence_detector_ucf_crime_binary_image_sequence.h5`.
![image](https://github.com/user-attachments/assets/4ca06e02-09dd-481e-94e4-e404ca971a1e)
![image](https://github.com/user-attachments/assets/53ce18f9-c2a8-4a63-be57-1312ab0e8de2)


## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

