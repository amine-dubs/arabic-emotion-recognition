#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import time
import joblib # Added for caching models/scalers
import gc # Added for garbage collection
import subprocess # Added for running the model training from app.py
import argparse # For command line arguments

# Imports for the new pipeline
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize # For resizing spectrograms

from mealpy import SCA
from mealpy.utils.space import BinaryVar # Import BinaryVar
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# --- Cache File Paths ---
FEATURES_CACHE_PATH = 'resnet_features.npy'
DATA_CACHE_PATH = 'data_df.pkl'
INDICES_CACHE_PATH = 'selected_indices.npy'
SCALER_CACHE_PATH = 'scaler.joblib'
MODEL_CACHE_PATH = 'knn_final_model.joblib'
EVALUATION_RESULTS_PATH = 'evaluation_results.joblib'
# ------------------------


# In[ ]:


EMOTIONS = { 0 : 'Angry',
             1 : 'Fearful',
             2 : 'Happy',
             3 : 'Neutral',
             4 : 'Sad',
             5 : 'Surprised'
           }  
# !!! IMPORTANT: Update this path to your local dataset location !!!
# DATA_PATH = '/kaggle/input/eaed-voice/EAED' 
# DATA_PATH = './EAED' # Example local path
DATA_PATH = './Data' # Corrected path based on workspace structure
SAMPLE_RATE = 16000
DURATION = 3 # seconds
RESNET_INPUT_SHAPE = (224, 224, 3) # ResNet50 expects 3 channels


# # 1. Data Loading and Initial Exploration

# In[ ]:


# --- Check for cached features and DataFrame ---
if os.path.exists(FEATURES_CACHE_PATH) and os.path.exists(DATA_CACHE_PATH):
    print(f"Loading cached features from {FEATURES_CACHE_PATH}...")
    resnet_features = np.load(FEATURES_CACHE_PATH)
    print(f"Loading cached DataFrame from {DATA_CACHE_PATH}...")
    Data = pd.read_pickle(DATA_CACHE_PATH)
    print("Loaded features shape:", resnet_features.shape)
    print("Loaded DataFrame head:\n", Data.head())
    # Skip data loading, spectrogram generation, and feature extraction
    SKIP_FEATURE_EXTRACTION = True
else:
    print("Cache not found. Starting data loading and feature extraction...")
    SKIP_FEATURE_EXTRACTION = False
# ---------------------------------------------

if not SKIP_FEATURE_EXTRACTION:
    file_names = []
    file_emotions = []
    file_paths = []

    # Define mapping from filename code to full emotion name
    emotion_code_map = {
        'ang': 'Angry',
        'hap': 'Happy',
        'neu': 'Neutral',
        'sad': 'Sad',
        # Add other codes if present (e.g., 'fea' for Fearful, 'sur' for Surprised)
    }

    # Iterate over each show folder (e.g., EYASE)
    for show_folder in os.listdir(DATA_PATH):
        show_path = os.path.join(DATA_PATH, show_folder)
        if not os.path.isdir(show_path):
            continue
        
        # Iterate over actor folders (e.g., Female01, Male02)
        for actor_folder in os.listdir(show_path):
            actor_path = os.path.join(show_path, actor_folder)
            if not os.path.isdir(actor_path):
                continue
            
            # Iterate over audio files within the actor folder
            for audio_file in os.listdir(actor_path):
                if audio_file.endswith(".wav"):
                    try:
                        # Parse information from the file name
                        # Example: fm01_ang (1).wav -> parts = ['fm01', 'ang (1).wav']
                        parts = audio_file.split("_", 1) 
                        if len(parts) < 2:
                            print(f"Skipping file with unexpected format: {audio_file}")
                            continue
                            
                        # Extract emotion code: 'ang (1).wav' -> 'ang'
                        emotion_code = parts[1].split(' ')[0]
                        
                        # Map code to full emotion name
                        emotion_full_name = emotion_code_map.get(emotion_code)

                        # Check if the emotion code was found in the map
                        if emotion_full_name is None:
                            print(f"Warning: Emotion code '{emotion_code}' not found in map for file {audio_file}. Skipping.")
                            continue

                        # Encode emotion using the EMOTIONS dictionary
                        try:
                            emotion_encoded = list(EMOTIONS.keys())[list(EMOTIONS.values()).index(emotion_full_name)]
                        except ValueError:
                             print(f"Warning: Emotion '{emotion_full_name}' not found in EMOTIONS dictionary for file {audio_file}. Skipping.")
                             continue

                        # Construct the full file path
                        file_path = os.path.join(actor_path, audio_file)
                        
                        # Append the information to the lists
                        file_names.append(audio_file)
                        file_emotions.append(emotion_encoded)
                        file_paths.append(file_path)
                    except Exception as e:
                        print(f"Error processing file {audio_file} in {actor_path}: {e}")


    # In[4]:


    # Create a DataFrame
    Data = pd.DataFrame({
        "Name": file_names,
        "Emotion": file_emotions,
        "Path": file_paths
    })
    # --- Save DataFrame to cache ---
    print(f"Saving DataFrame to {DATA_CACHE_PATH}...")
    Data.to_pickle(DATA_CACHE_PATH)
    # -----------------------------


# In[5]:


Data.head()


# In[6]:


print("number of files is {}".format(len(Data)))


# In[7]:


# Get the actual counts and labels present in the data
emotion_counts = Data['Emotion'].value_counts().sort_index() # Sort by index (emotion code)
emotion_labels_present = emotion_counts.index.tolist() # Get the numeric labels (0, 2, 3, 4)
emotion_names_present = [EMOTIONS[i] for i in emotion_labels_present] # Get the corresponding names

fig = plt.figure()
ax = fig.add_subplot(111)
# Use the actual number of emotions present for the x-axis positions
ax.bar(x=range(len(emotion_counts)), height=emotion_counts.values)
# Set ticks and labels based on the emotions actually present
ax.set_xticks(ticks=range(len(emotion_counts)))
ax.set_xticklabels(emotion_names_present, fontsize=10)
ax.set_xlabel('Emotions')
ax.set_ylabel('Number of examples')


# # 2. Spectrogram Generation

# In[8]::


def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length = 512,
                                              window='hamming',
                                              hop_length = 256,
                                              n_mels=128,
                                              fmax=sample_rate/2
                                             )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Function for data augmentation - add noise with controlled SNR for balancing classes
def augment_audio_with_noise(audio, snr_db=15):
    """Add noise to audio sample with specified signal-to-noise ratio.
    
    Args:
        audio: Audio signal array
        snr_db: Signal-to-noise ratio in dB (higher = less noise)
    
    Returns:
        Augmented audio signal
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    # Add noise to signal
    augmented_audio = audio + noise
    
    # Normalize to avoid clipping
    if np.max(np.abs(augmented_audio)) > 1.0:
        augmented_audio = augmented_audio / np.max(np.abs(augmented_audio))
        
    return augmented_audio

# --- Skip spectrogram generation if features are loaded ---
if not SKIP_FEATURE_EXTRACTION:
    # In[ ]:


    audio, sample_rate = librosa.load(Data.loc[0,'Path'], duration=DURATION, offset=0.5,sr=SAMPLE_RATE)
    signal = np.zeros((int(SAMPLE_RATE*DURATION,)))
    signal[:len(audio)] = audio
    mel_spectrogram = getMELspectrogram(signal, SAMPLE_RATE)
    librosa.display.specshow(mel_spectrogram, sr=sample_rate, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MEL spectrogram example')
    plt.tight_layout()
    plt.show()
    print('Original MEL spectrogram shape: ',mel_spectrogram.shape)


    # In[ ]:


    # Function to resize and prepare spectrogram for ResNet50
    def prepare_spectrogram_for_resnet(spec, output_shape):
        # Resize
        spec_resized = resize(spec, output_shape[:2], anti_aliasing=True)
        # Normalize to 0-1 range (assuming spec is in dB)
        spec_resized -= spec_resized.min()
        if spec_resized.max() > 0:
            spec_resized /= spec_resized.max()
        # Convert to 3 channels by stacking
        spec_3channel = np.stack([spec_resized]*3, axis=-1)
        return spec_3channel


    # In[ ]:


    # Test preparation function
    prepared_spec = prepare_spectrogram_for_resnet(mel_spectrogram, RESNET_INPUT_SHAPE)
    print('Prepared spectrogram shape: ', prepared_spec.shape)
    plt.imshow(prepared_spec)
    plt.title('Prepared Spectrogram (3 channels)')
    plt.show()


    # In[ ]:


    prepared_spectrograms = []
    valid_indices = [] # Keep track of successfully processed files
    start_time = time.time()
    print("Generating and preparing spectrograms...")
    # Use iterrows for safer dropping
    for i, row in Data.iterrows():
        file_path = row['Path']
        try:
            audio, sample_rate = librosa.load(file_path, duration=DURATION, offset=0.5, sr=SAMPLE_RATE)
            signal = np.zeros((int(SAMPLE_RATE*DURATION,)))
            signal[:len(audio)] = audio
            mel_spectrogram = getMELspectrogram(signal, sample_rate=SAMPLE_RATE)
            prepared_spec = prepare_spectrogram_for_resnet(mel_spectrogram, RESNET_INPUT_SHAPE)
            prepared_spectrograms.append(prepared_spec)
            valid_indices.append(i) # Add index if successful
        except Exception as e:
            print(f"\nError processing {file_path} (index {i}): {e}. Skipping this file.")
            # Don't add to prepared_spectrograms or valid_indices

        print(f"\r Processed {i+1}/{len(Data)} files", end='')

    # Filter Data to keep only successfully processed rows
    print(f"\nFiltering DataFrame to keep {len(valid_indices)} successfully processed files.")
    Data = Data.loc[valid_indices].reset_index(drop=True)
    # --- Update cached DataFrame after filtering ---
    print(f"Saving filtered DataFrame to {DATA_CACHE_PATH}...")
    Data.to_pickle(DATA_CACHE_PATH)
    # ---------------------------------------------

    prepared_spectrograms = np.array(prepared_spectrograms)
    end_time = time.time()
    print(f"\nFinished spectrograms in {end_time - start_time:.2f} seconds.")
    print("Shape of prepared spectrograms array:", prepared_spectrograms.shape)


    # # 3. Feature Extraction using ResNet50

    # In[ ]:


    # Load pre-trained ResNet50 model + higher level layers
    print("Loading ResNet50 model...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=RESNET_INPUT_SHAPE, pooling='avg')
    # base_model.summary() # Optional: Print model summary

    # Create a new model that outputs the features
    feature_model = Model(inputs=base_model.input, outputs=base_model.output)
    print("ResNet50 model loaded.")


    # In[ ]:


    # Extract features
    # Note: ResNet50 expects preprocessed input (specific normalization)
    # We'll apply the standard ResNet preprocessing
    print("Extracting features using ResNet50...")
    start_time = time.time()
    # Ensure prepared_spectrograms is not empty before predicting
    if prepared_spectrograms.shape[0] > 0:
        resnet_features = feature_model.predict(tf.keras.applications.resnet50.preprocess_input(prepared_spectrograms))
        # --- Save features to cache ---
        print(f"Saving features to {FEATURES_CACHE_PATH}...")
        np.save(FEATURES_CACHE_PATH, resnet_features)
        # ----------------------------
    else:
        print("Error: No spectrograms were successfully prepared. Cannot extract features.")
        # Handle this error appropriately, maybe exit or raise an exception
        resnet_features = np.array([]) # Assign empty array to avoid later errors

    end_time = time.time()
    print(f"Finished feature extraction in {end_time - start_time:.2f} seconds.")
    print("Shape of extracted features:", resnet_features.shape)

    # Free up memory
    del prepared_spectrograms
    gc.collect()
# --- End of SKIP_FEATURE_EXTRACTION block ---


# # 4. Data Splitting and Scaling

# In[ ]:

# Ensure resnet_features is not empty before proceeding
if resnet_features.shape[0] == 0 or resnet_features.shape[0] != len(Data):
     print("Error: Feature array is empty or does not match DataFrame length. Exiting.")
     # Exit or raise error
     exit()


X = resnet_features
y = Data['Emotion'].values

# Split data into training+validation and testing sets (80% train+val, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Split training+validation into training and validation sets (80% train, 20% val of the original 80%)
# This means 64% train, 16% val, 20% test of the total data
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val) # 0.2 * 0.8 = 0.16

print("Original data shape:", X.shape, y.shape)
print("Training data shape:", X_train.shape, y_train.shape)
print("Validation data shape:", X_val.shape, y_val.shape)
print("Test data shape:", X_test.shape, y_test.shape)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Free up memory (keep scaled data and y_test, y_train)
del X, X_train, X_val, X_test, X_train_val, y_train_val, resnet_features
gc.collect()


# # 5. Feature Selection using Sine Cosine Algorithm (SCA)

# In[ ]:

# --- Check for cached indices ---
if os.path.exists(INDICES_CACHE_PATH):
    print(f"Loading cached selected indices from {INDICES_CACHE_PATH}...")
    selected_feature_indices = np.load(INDICES_CACHE_PATH)
    num_selected_features = len(selected_feature_indices)
    num_features_total = X_train_scaled.shape[1] # Need total for print statement
    print(f"Loaded {num_selected_features} selected indices out of {num_features_total}.")
    SKIP_SCA = True
else:
    print("Cache not found. Running SCA...")
    SKIP_SCA = False
# -----------------------------

if not SKIP_SCA:
    # Define the fitness function for SCA
    num_features_total = X_train_scaled.shape[1]
    knn_eval = KNeighborsClassifier(n_neighbors=5) # k-NN for evaluation within fitness function

    def fitness_function(solution):
        selected_indices = np.where(solution == 1)[0]
        num_selected = len(selected_indices)

        if num_selected == 0:
            return 1.0 # Penalize solutions with no features

        # Select features
        X_train_subset = X_train_scaled[:, selected_indices]
        X_val_subset = X_val_scaled[:, selected_indices]

        # Train k-NN on the subset
        knn_eval.fit(X_train_subset, y_train)

        # Evaluate on validation set
        y_pred_val = knn_eval.predict(X_val_subset)
        accuracy = accuracy_score(y_val, y_pred_val)

        # Calculate fitness value (minimize this)
        fitness = 0.99 * (1 - accuracy) + 0.01 * (num_selected / num_features_total)
        return fitness


    # In[ ]:


    # SCA Parameters
    epoch = 20 # Number of iterations (reduce for quicker testing)
    pop_size = 10 # Population size (reduce for quicker testing)
    
    # Create a list of BinaryVar objects, one for each feature (each dimension)
    binary_bounds = [BinaryVar() for _ in range(num_features_total)]
    
    problem_dict = {
        "obj_func": fitness_function, # Rename fit_func to obj_func
        "bounds": binary_bounds, # List of BinaryVar objects
        "minmax": "min",
        "log_to": None, # No logging
        "verbose": True,
    }

    # Initialize and run SCA
    print("Running Sine Cosine Algorithm for feature selection...")
    start_time = time.time()
    model_sca = SCA.OriginalSCA(epoch=epoch, pop_size=pop_size)
    best_solution = model_sca.solve(problem_dict)
    end_time = time.time()
    print(f"SCA finished in {end_time - start_time:.2f} seconds.")
    
    # Extract selected indices from the best solution
    best_position = best_solution.solution  # Get the binary vector
    selected_feature_indices = np.where(best_position == 1)[0]  # Get indices of 1s
    num_selected_features = len(selected_feature_indices)
    best_fitness = best_solution.target.fitness  # Get the fitness value
    
    print(f"Selected {num_selected_features} features out of {num_features_total}.")
    print(f"Best fitness found: {best_fitness:.4f}")

    # --- Save selected indices to cache ---
    print(f"Saving selected indices to {INDICES_CACHE_PATH}...")
    np.save(INDICES_CACHE_PATH, selected_feature_indices)
    # ----------------------------------
# --- End of SKIP_SCA block ---


# # 6. Classification using k-NN with Selected Features

# In[ ]:

# --- Check for cached model and scaler ---
if os.path.exists(MODEL_CACHE_PATH) and os.path.exists(SCALER_CACHE_PATH):
    print(f"Loading cached k-NN model from {MODEL_CACHE_PATH}...")
    knn_final = joblib.load(MODEL_CACHE_PATH)
    print(f"Loading cached scaler from {SCALER_CACHE_PATH}...")
    scaler = joblib.load(SCALER_CACHE_PATH) # Overwrite scaler with the saved one
    print(f"Loading cached selected indices from {INDICES_CACHE_PATH}...")
    selected_feature_indices = np.load(INDICES_CACHE_PATH)
    print("Loaded model and scaler.")
    
    # Make predictions even when loading cached model
    X_test_selected = X_test_scaled[:, selected_feature_indices]
    print("Predicting on test set using cached model...")
    start_time = time.time()
    y_pred_test = knn_final.predict(X_test_selected)
    end_time = time.time()
    print(f"Prediction finished in {end_time - start_time:.2f} seconds.")
    
    SKIP_TRAINING = True
else:
    print("Cache not found. Training final k-NN model...")
    SKIP_TRAINING = False
# --------------------------------------

if not SKIP_TRAINING:
    # Select the features chosen by SCA
    # Ensure selected_feature_indices is loaded or calculated
    if 'selected_feature_indices' not in locals():
         print("Error: selected_feature_indices not found. Cannot train model.")
         exit() # Or handle error

    X_train_selected = X_train_scaled[:, selected_feature_indices]
    X_val_selected = X_val_scaled[:, selected_feature_indices]
    X_test_selected = X_test_scaled[:, selected_feature_indices]
    
    print("Training k-NN classifier on selected features...")

    # Add class weighting to handle imbalance
    from collections import Counter
    class_weights = Counter(y_train)
    total_samples = len(y_train)
    for label in class_weights:
        class_weights[label] = total_samples / (len(class_weights) * class_weights[label])
    print(f"Class weights to handle imbalance: {class_weights}")

    # Add KNN hyperparameter tuning
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, accuracy_score, f1_score

    # Define scoring metrics with greater emphasis on balanced performance
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro'),
    }

    # Define parameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]  # Only relevant for minkowski
    }

    # Create and train GridSearchCV
    start_time = time.time()
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit='f1_macro',  # Optimize for balanced F1 score across classes
        verbose=1,
        n_jobs=-1  # Use all available cores
    )

    knn_grid.fit(X_train_selected, y_train)
    end_time = time.time()
    print(f"Hyperparameter tuning finished in {end_time - start_time:.2f} seconds.")

    # Get best parameters and model
    best_params = knn_grid.best_params_
    print(f"Best parameters: {best_params}")
    best_score = knn_grid.best_score_
    print(f"Best cross-validation score: {best_score:.4f}")

    # Use the best model
    knn_final = knn_grid.best_estimator_
    print("Predicting on test set...")

    start_time = time.time()
    y_pred_test = knn_final.predict(X_test_selected)
    end_time = time.time()
    print(f"Prediction finished in {end_time - start_time:.2f} seconds.")

    # Save the final model
    import joblib
    joblib.dump(knn_final, 'knn_final_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Use np.save instead of joblib.dump for selected_indices to ensure proper format compatibility
    np.save('selected_indices.npy', selected_feature_indices)
    print(f"Saved selected indices using np.save")

    # Optional: Try an ensemble approach for comparison
    from sklearn.ensemble import RandomForestClassifier
    print("Training Random Forest classifier for comparison...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_selected, y_train)
    rf_pred = rf_model.predict(X_test_selected)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='macro')
    print(f"Random Forest - Accuracy: {rf_accuracy:.4f}, F1-Score: {rf_f1:.4f}")
else:
    # Make predictions even when loading cached model
    X_test_selected = X_test_scaled[:, selected_feature_indices]
    print("Predicting on test set using cached model...")
    start_time = time.time()
    y_pred_test = knn_final.predict(X_test_selected)
    end_time = time.time()
    print(f"Prediction finished in {end_time - start_time:.2f} seconds.")


# # 7. Evaluation and Visualization

# In[ ]:


# Calculate metrics
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average='weighted') # Use 'weighted
# Get the unique labels present in y_test and y_pred_test for the report and matrix
report_labels = sorted(np.unique(np.concatenate((y_test, y_pred_test))))
report_target_names = [EMOTIONS[i] for i in report_labels]
conf_matrix = confusion_matrix(y_test, y_pred_test, labels=report_labels) # Ensure matrix uses correct labels

class_report = classification_report(y_test, y_pred_test, target_names=report_target_names, labels=report_labels)

print(f"\n--- Evaluation Results ---")
print(f"Test Accuracy: {accuracy_test:.4f}")
print(f"Test F1-Score (Weighted): {f1_test:.4f}")
print("\nClassification Report:")
print(class_report)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=report_target_names, # Use names corresponding to actual labels
            yticklabels=report_target_names) # Use names corresponding to actual labels
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# # --- End of New Pipeline ---
# 
# --- (Original Kaggle Notebook Code Below - Can be removed) ---

# In[ ]:


# Original Augmentation Code (Not used in the new pipeline above)
# def addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30): 
#     ...
#     return signal + K.T * noise


# In[ ]:


# Original Augmentation Application (Not used)
# for i,signal in enumerate(signals):
#     ...
#     print("\r Processed {}/{} files".format(i,len(signals)),end='')


# In[ ]:


# Original Model Imports (Not used)
# import torch
# import torch.nn as nn


# In[ ]:


# Original ParallelModel Definition (Not used)
# class ParallelModel(nn.Module):
#     ...
#     return output_logits, output_softmax


# In[ ]:


# Original Loss Function (Not used)
# def loss_fnc(predictions, targets):
#     return nn.CrossEntropyLoss()(input=predictions,target=targets)


# In[ ]:


# Original Training Step (Not used)
# def make_train_step(model, loss_fnc, optimizer):
#     ...
#     return train_step


# In[ ]:


# Original Validation Function (Not used)
# def make_validate_fnc(model,loss_fnc):
#     ...
#     return validate


# In[ ]:


# Original Data Stacking (Handled differently now)
# X = np.stack(mel_spectrograms,axis=0)
# ...
# del signals


# In[ ]:


# Original Data Splitting (Handled differently now)
# train_ind,test_ind,val_ind = [],[],[]
# ...
# del X


# In[ ]:


# Original Scaling (Handled differently now)
# from sklearn.preprocessing import StandardScaler
# ...
# X_val = np.reshape(X_val, newshape=(b,c,h,w))


# In[ ]:


# Original Training Loop (Not used)
# EPOCHS=150
# ...
# print(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%")


# In[ ]:


# Original Test Evaluation (Handled differently now)
# X_test_tensor = torch.tensor(X_test,device=device).float()
# ...
# print(f'Test accuracy is {test_acc:.2f}%')


# In[ ]:


# Original Loss Plot (Not applicable)
# plt.plot(losses,'b')
# plt.plot(val_losses,'r')
# plt.legend(['train loss','val loss'])


# In[ ]:


# Integrated Evaluation Metrics (from extract_evaluation_metrics.py)
# ---------------------------------------------------------------------
def evaluate_and_save_results():
    """Evaluate the model and save comprehensive metrics."""
    print("\n\n" + "="*50)
    print("GENERATING COMPREHENSIVE EVALUATION METRICS")
    print("="*50)
    
    # Use the already loaded/calculated data
    # We already have X_test_scaled, y_test, y_pred_test, selected_feature_indices, etc.
    
    # Calculate advanced evaluation metrics
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    
    # Compute per-class metrics
    unique_emotions = sorted(np.unique(np.concatenate([y_test, y_pred_test])))
    emotion_names = [EMOTIONS.get(e, f"Unknown-{e}") for e in unique_emotions]
    
    # Confusion matrix with proper labels
    cm = confusion_matrix(y_test, y_pred_test, labels=unique_emotions)
    
    # Full classification report
    report = classification_report(
        y_test, 
        y_pred_test, 
        labels=unique_emotions,
        target_names=emotion_names,
        output_dict=True
    )
    
    # Calculate metrics for training and validation sets too
    X_train_selected = X_train_scaled[:, selected_feature_indices]
    X_val_selected = X_val_scaled[:, selected_feature_indices]
    
    y_train_pred = knn_final.predict(X_train_selected)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    y_val_pred = knn_final.predict(X_val_selected)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    # Compute per-class metrics
    train_per_class = {}
    val_per_class = {}
    test_per_class = {}
    
    for emotion in unique_emotions:
        # Get indices for this emotion
        train_indices = np.where(y_train == emotion)[0]
        val_indices = np.where(y_val == emotion)[0]
        test_indices = np.where(y_test == emotion)[0]
        
        # Calculate accuracies for this emotion (checking if enough samples exist)
        if len(train_indices) > 0:
            train_emotion_true = y_train[train_indices]
            train_emotion_pred = y_train_pred[train_indices]
            train_per_class[emotion] = accuracy_score(train_emotion_true, train_emotion_pred)
        else:
            train_per_class[emotion] = 0
        
        if len(val_indices) > 0:
            val_emotion_true = y_val[val_indices]
            val_emotion_pred = y_val_pred[val_indices]
            val_per_class[emotion] = accuracy_score(val_emotion_true, val_emotion_pred)
        else:
            val_per_class[emotion] = 0
        
        if len(test_indices) > 0:
            test_emotion_true = y_test[test_indices]
            test_emotion_pred = y_pred_test[test_indices]
            test_per_class[emotion] = accuracy_score(test_emotion_true, test_emotion_pred)
        else:
            test_per_class[emotion] = 0
    
    # Create confusion matrix figure
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=emotion_names,
        yticklabels=emotion_names
    )
    plt.title('Confusion Matrix for EAED (Arabic Emotion Recognition)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Create accuracy by emotion figure
    fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
    
    # Convert per-class metrics to a format suitable for plotting
    emotion_dict = {}
    for emotion in unique_emotions:
        emotion_name = EMOTIONS.get(emotion, f"Unknown-{emotion}")
        train_acc = train_per_class.get(emotion, 0)
        val_acc = val_per_class.get(emotion, 0)
        test_acc = test_per_class.get(emotion, 0)
        emotion_dict[emotion_name] = {
            'Train': train_acc,
            'Validation': val_acc,
            'Test': test_acc
        }
    
    # Create DataFrame and plot
    emotion_df = pd.DataFrame(emotion_dict).T
    emotion_df.plot(kind='bar', ax=ax_acc)
    ax_acc.set_ylim(0, 1.1)
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_xlabel('Emotion')
    ax_acc.set_title('Accuracy by Emotion for Different Datasets')
    ax_acc.legend(title='Dataset')
    plt.tight_layout()
    
    # Try to create feature importance chart
    fig_imp = None
    try:
        # For KNN, calculate feature importance via a simple distance-based metric
        importances = np.zeros(len(selected_feature_indices))
        for i, neighbor in enumerate(knn_final.kneighbors(X_test_selected)[1]):
            # Get neighbor samples
            neighbors = X_train_selected[neighbor]
            # Calculate feature-wise spread
            spread = np.std(neighbors, axis=0)
            # Features with lower spread among neighbors are more important
            importances += 1.0 / (spread + 1e-10)  # Add small constant to avoid division by zero
        
        # Normalize importances
        importances = importances / np.sum(importances)
        
        # Get top N most important features
        top_n = min(20, len(importances))
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_importances = importances[top_indices]
        
        # Create importance figure
        fig_imp, ax_imp = plt.subplots(figsize=(12, 6))
        ax_imp.bar(range(top_n), top_importances)
        ax_imp.set_xticks(range(top_n))
        ax_imp.set_xticklabels([f"Feature {selected_feature_indices[i]}" for i in top_indices], rotation=90)
        ax_imp.set_title('Top Feature Importances (Estimated from KNN)')
        ax_imp.set_ylabel('Relative Importance')
        plt.tight_layout()
    except Exception as e:
        print(f"Could not create feature importance figure: {e}")
    
    # Class distribution figure
    fig_dist = None
    try:
        # Count samples per class
        class_counts = {EMOTIONS.get(i, f"Unknown-{i}"): (y == i).sum() for i in unique_emotions}
        
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        ax_dist.bar(class_counts.keys(), class_counts.values())
        ax_dist.set_title('Distribution of Emotion Classes in Dataset')
        ax_dist.set_ylabel('Number of Samples')
        ax_dist.set_xlabel('Emotion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    except Exception as e:
        print(f"Could not create class distribution figure: {e}")
    
    # Prepare results dictionary with all metrics
    results = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'confusion_matrix': cm,
        'unique_emotions': unique_emotions,
        'emotion_names': emotion_names,
        'train_size': len(y_train),
        'val_size': len(y_val),
        'test_size': len(y_test),
        'classification_report': report,
        'train_per_class': train_per_class,
        'val_per_class': val_per_class,
        'test_per_class': test_per_class,
        'confusion_matrix_fig': fig_cm,
        'accuracy_by_emotion_fig': fig_acc,
        'feature_importance_fig': fig_imp,
        'class_distribution_fig': fig_dist,
    }
    
    # Save all results
    joblib.dump(results, EVALUATION_RESULTS_PATH)
    print(f"\nEvaluation results saved to '{EVALUATION_RESULTS_PATH}'")
    
    # Save individual figures to files
    try:
        fig_cm.savefig('confusion_matrix.png')
        fig_acc.savefig('accuracy_by_emotion.png')
        if fig_imp is not None:
            fig_imp.savefig('feature_importance.png')
        if fig_dist is not None:
            fig_dist.savefig('class_distribution.png')
        print("Evaluation plots saved as PNG files.")
    except Exception as e:
        print(f"Error saving evaluation plots: {e}")
    
    # Create a simple markdown report
    report_content = f"""
    # Arabic Emotion Recognition Model Evaluation Report

    ## Overall Performance
    - Training Accuracy: {train_accuracy:.4f}
    - Validation Accuracy: {val_accuracy:.4f}
    - Test Accuracy: {test_accuracy:.4f}
    
    - Test F1 Score (Weighted): {test_f1:.4f}
    - Test Precision (Weighted): {test_precision:.4f}
    - Test Recall (Weighted): {test_recall:.4f}
    
    ## Dataset Information
    - Training Set: {len(y_train)} samples
    - Validation Set: {len(y_val)} samples
    - Test Set: {len(y_test)} samples
    - Unique Emotions: {', '.join(emotion_names)}
    """
    
    with open('evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
        print("Saved evaluation report to 'evaluation_report.md'")
    
    # Print evaluation summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Training Set: {len(y_train)} samples")
    print(f"Validation Set: {len(y_val)} samples")
    print(f"Test Set: {len(y_test)} samples")
    print(f"Selected Features: {len(selected_feature_indices)} out of {X_train_scaled.shape[1]}")
    print("="*50)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("="*50)
    print(f"Test F1 Score (Weighted): {test_f1:.4f}")
    print(f"Test Precision (Weighted): {test_precision:.4f}")
    print(f"Test Recall (Weighted): {test_recall:.4f}")
    print("="*50)
    
    return results

# Run the evaluation and save the results
evaluate_and_save_results()

# Provide a command line interface to run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arabic Emotion Recognition Training Script')
    parser.add_argument('--clear-all', action='store_true', 
                        help='Clear all cached files before running')
    parser.add_argument('--clear-model', action='store_true', 
                        help='Clear only model files before running')
    parser.add_argument('--clear-evaluation', action='store_true', 
                        help='Clear only evaluation results before running')
    args = parser.parse_args()
    
    # Handle cache clearing if requested
    if args.clear_all:
        cache_files = [FEATURES_CACHE_PATH, DATA_CACHE_PATH, INDICES_CACHE_PATH, 
                       SCALER_CACHE_PATH, MODEL_CACHE_PATH, EVALUATION_RESULTS_PATH]
        for file in cache_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted cache file: {file}")
    
    elif args.clear_model:
        model_files = [INDICES_CACHE_PATH, SCALER_CACHE_PATH, MODEL_CACHE_PATH, EVALUATION_RESULTS_PATH]
        for file in model_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted model file: {file}")
    
    elif args.clear_evaluation:
        if os.path.exists(EVALUATION_RESULTS_PATH):
            os.remove(EVALUATION_RESULTS_PATH)
            print(f"Deleted evaluation results: {EVALUATION_RESULTS_PATH}")
    
    print("Training and evaluation complete.")




