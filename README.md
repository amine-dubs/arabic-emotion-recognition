# Arabic Audio Emotion Recognition using ResNet50, SCA, and k-NN

This project implements an emotion recognition system for Arabic audio signals based on the specifications provided.

## Project Goal

The primary goal is to classify emotions from Arabic audio recordings by:
1.  Transforming audio signals into MEL spectrograms.
2.  Extracting deep features using a pre-trained ResNet50 model (average pooling layer output).
3.  Selecting the most relevant features using the Sine Cosine Algorithm (SCA) from the `mealpy` library.
4.  Classifying emotions using the k-Nearest Neighbors (k-NN) algorithm.

## Implementation Details

### 1. Dataset
-   The implementation assumes the use of the **Egyptian Arabic Emotion Dataset (EAED)** or a similarly structured dataset.
-   The dataset should be placed in a folder named `EAED` in the project root.
-   The expected structure inside `EAED` is `Show/Actor/Emotion/*.wav`.
-   Audio files are loaded using `librosa`, resampled to 16kHz, and truncated/padded to 3 seconds.

### 2. Data Preprocessing
-   Audio signals are converted into MEL spectrograms using `librosa.feature.melspectrogram`.
-   Spectrograms are converted to decibels.

### 3. Feature Extraction
-   A pre-trained ResNet50 model (`tensorflow.keras.applications.ResNet50`) with ImageNet weights is used.
-   Spectrograms are resized to (224, 224), normalized, and converted to 3 channels to match ResNet50 input requirements.
-   Features are extracted from the global average pooling layer of the ResNet50 model (output shape: 2048 features).
-   Features are scaled using `sklearn.preprocessing.StandardScaler` after splitting the data.

### 4. Feature Selection
-   Utilizes the Sine Cosine Algorithm (`mealpy.swarm_based.SCA.OriginalSCA`).
-   **Problem Type:** Binary (select or discard feature).
-   **Fitness Function:** Minimize `0.99 * (1 - validation_accuracy) + 0.01 * (num_selected_features / total_features)`.
    -   `validation_accuracy` is calculated using k-NN (k=5) on a dedicated validation set.
-   SCA parameters (e.g., `epoch`, `pop_size`) are defined in the notebook and can be tuned.

### 5. Classification
-   A final k-NN classifier (`sklearn.neighbors.KNeighborsClassifier`, k=5) is trained using the optimal feature subset identified by SCA on the training data.
-   The trained model predicts emotions on the unseen test set.

### 6. Evaluation and Visualization
-   Performance is evaluated on the test set using:
    -   Accuracy (`sklearn.metrics.accuracy_score`)
    -   Weighted F1-Score (`sklearn.metrics.f1_score`)
    -   Classification Report (`sklearn.metrics.classification_report`)
    -   Confusion Matrix (`sklearn.metrics.confusion_matrix`), visualized using `seaborn`.
-   Execution times for major steps (spectrogram generation, feature extraction, SCA, k-NN training/prediction) are measured.

## Dependencies

-   Python 3.x
-   numpy
-   pandas
-   librosa
-   matplotlib
-   seaborn
-   tensorflow (for Keras and ResNet50)
-   scikit-learn (for k-NN, metrics, scaling, splitting)
-   mealpy (for SCA)
-   scikit-image (for resizing spectrograms)
-   ipython (for notebook display elements)

(A `requirements.txt` file should be generated based on the environment used).

## How to Run

1.  **Setup:**
    -   Clone the repository (if applicable).
    -   Download the EAED dataset and place it in the `./EAED/` folder, or prepare your own dataset as described in section 1.
    -   Install dependencies: `pip install numpy pandas librosa matplotlib seaborn tensorflow scikit-learn mealpy scikit-image ipython` (adjust based on your Python environment manager).
2.  **Execution:**
    -   Open and run the `eaed-using-parallel-cnn-transformer.ipynb` notebook cell by cell.
    -   Ensure the `DATA_PATH` variable in the second code cell points to the correct dataset location.

## Future Enhancements

-   Wrap the core logic in a web application framework (e.g., Flask, Streamlit) for interactive use.
-   Improve the User Interface (UI) and User Experience (UX) of the web application.
-   Experiment with different feature extraction models (e.g., VGG, EfficientNet) or audio-specific models.
-   Experiment with other feature selection algorithms available in `mealpy` or other libraries.
-   Tune hyperparameters for k-NN (e.g., `n_neighbors`) and SCA (e.g., `epoch`, `pop_size`) using techniques like GridSearch or RandomizedSearch on the validation set.
-   Implement more sophisticated audio data augmentation techniques.
