import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Define emotion mapping globally to ensure consistency
EMOTIONS = {
    0: 'Angry',
    1: 'Fearful',
    2: 'Happy',
    3: 'Neutral',
    4: 'Sad',
    5: 'Surprised'
}

# Load the required files
def load_data_files():
    """Load the saved files from the model training process."""
    try:
        # Check if all required files exist
        required_files = [
            'resnet_features.npy',
            'data_df.pkl',
            'selected_indices.npy',
            'scaler.joblib',
            'knn_final_model.joblib'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                print(f"Missing required file: {file}")
                return None, None, None, None, None
        
        # Load the features and DataFrame - Add allow_pickle=True to fix the error
        features = np.load('resnet_features.npy', allow_pickle=True)
        data_df = pd.read_pickle('data_df.pkl')
        
        # Load selected indices and models - Add allow_pickle=True to fix the error
        selected_indices = np.load('selected_indices.npy', allow_pickle=True)
        scaler = joblib.load('scaler.joblib')
        knn_model = joblib.load('knn_final_model.joblib')
        
        print("Successfully loaded all required data files.")
        return features, data_df, selected_indices, scaler, knn_model
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None, None, None, None, None

def split_and_evaluate():
    """Split the data and evaluate the model performance."""
    features, data_df, selected_indices, scaler, knn_model = load_data_files()
    
    if features is None or len(features) == 0:
        print("Could not load the required files or files are empty.")
        return None
    
    # Get the emotion labels
    y = data_df['Emotion'].values
    
    # Split the data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    # Scale the features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Extract selected features
    X_train_selected = X_train_scaled[:, selected_indices]
    X_val_selected = X_val_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    print(f"Data shapes after preprocessing:")
    print(f"X_train_selected: {X_train_selected.shape}")
    print(f"X_val_selected: {X_val_selected.shape}")
    print(f"X_test_selected: {X_test_selected.shape}")
    
    # Evaluate on training set
    y_train_pred = knn_model.predict(X_train_selected)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    # Evaluate on validation set
    y_val_pred = knn_model.predict(X_val_selected)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    # Evaluate on test set
    y_test_pred = knn_model.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    # Get the unique emotion labels present in data
    unique_emotions = sorted(np.unique(np.concatenate([y_test, y_test_pred])))
    emotion_names = [EMOTIONS.get(e, f"Unknown-{e}") for e in unique_emotions]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=unique_emotions)
    
    # Get detailed classification report
    report = classification_report(
        y_test, 
        y_test_pred, 
        labels=unique_emotions,
        target_names=emotion_names,
        output_dict=True
    )
    
    # Prepare results dictionary
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
        'classification_report': report
    }
    
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
            test_emotion_pred = y_test_pred[test_indices]
            test_per_class[emotion] = accuracy_score(test_emotion_true, test_emotion_pred)
        else:
            test_per_class[emotion] = 0
    
    # Add per-class metrics to results
    results['train_per_class'] = train_per_class
    results['val_per_class'] = val_per_class 
    results['test_per_class'] = test_per_class
    
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
    
    # Add figures to results
    results['confusion_matrix_fig'] = fig_cm
    results['accuracy_by_emotion_fig'] = fig_acc
    
    # Create a feature importance chart
    try:
        # For KNN, calculate feature importance via a simple distance-based metric
        importances = np.zeros(len(selected_indices))
        for i, neighbor in enumerate(knn_model.kneighbors(X_test_selected)[1]):
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
        ax_imp.set_xticklabels([f"Feature {selected_indices[i]}" for i in top_indices], rotation=90)
        ax_imp.set_title('Top Feature Importances (Estimated from KNN)')
        ax_imp.set_ylabel('Relative Importance')
        plt.tight_layout()
        results['feature_importance_fig'] = fig_imp
    except Exception as e:
        print(f"Could not create feature importance figure: {e}")
    
    # Create class distribution figure
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
        results['class_distribution_fig'] = fig_dist
    except Exception as e:
        print(f"Could not create class distribution figure: {e}")
    
    # Print evaluation summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Training Set: {len(y_train)} samples")
    print(f"Validation Set: {len(y_val)} samples")
    print(f"Test Set: {len(y_test)} samples")
    print(f"Selected Features: {len(selected_indices)} out of {features.shape[1]}")
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

def save_evaluation_results():
    """Extract and save the evaluation results."""
    print("\nExtracting and evaluating model performance...")
    results = split_and_evaluate()
    
    if results:
        joblib.dump(results, 'evaluation_results.joblib')
        print("\nEvaluation results saved to 'evaluation_results.joblib'")
        
        # Also create an HTML or text report for easy viewing
        create_report(results)
        return True
    else:
        print("\nFailed to generate evaluation results")
        return False

def create_report(results):
    """Create a simple HTML report from the results."""
    try:
        # Save confusion matrix figure
        if 'confusion_matrix_fig' in results:
            results['confusion_matrix_fig'].savefig('confusion_matrix.png')
            print("Saved confusion matrix plot to 'confusion_matrix.png'")
        
        # Save accuracy by emotion figure
        if 'accuracy_by_emotion_fig' in results:
            results['accuracy_by_emotion_fig'].savefig('accuracy_by_emotion.png')
            print("Saved accuracy by emotion plot to 'accuracy_by_emotion.png'")
        
        # Save feature importance figure
        if 'feature_importance_fig' in results:
            results['feature_importance_fig'].savefig('feature_importance.png')
            print("Saved feature importance plot to 'feature_importance.png'")
        
        # Save class distribution figure
        if 'class_distribution_fig' in results:
            results['class_distribution_fig'].savefig('class_distribution.png')
            print("Saved class distribution plot to 'class_distribution.png'")
        
        # Create a simple text report
        report_content = f"""
        # Arabic Emotion Recognition Model Evaluation Report
        
        ## Overall Performance
        - Training Accuracy: {results['train_accuracy']:.4f}
        - Validation Accuracy: {results['val_accuracy']:.4f}
        - Test Accuracy: {results['test_accuracy']:.4f}
        
        - Test F1 Score (Weighted): {results['test_f1']:.4f}
        - Test Precision (Weighted): {results['test_precision']:.4f}
        - Test Recall (Weighted): {results['test_recall']:.4f}
        
        ## Dataset Information
        - Training Set: {results['train_size']} samples
        - Validation Set: {results['val_size']} samples
        - Test Set: {results['test_size']} samples
        - Unique Emotions: {', '.join(results['emotion_names'])}
        """
        
        with open('evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
            print("Saved evaluation report to 'evaluation_report.md'")
    except Exception as e:
        print(f"Error creating report: {e}")

if __name__ == "__main__":
    save_evaluation_results()