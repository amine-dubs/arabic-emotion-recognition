#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse

def main():
    """
    Script to selectively delete cached files for EAED emotion recognition model.
    This allows retraining specific components or the entire model pipeline.
    """
    parser = argparse.ArgumentParser(description='Clear cached files to enable retraining of specific components.')
    parser.add_argument('--all', action='store_true', help='Clear all cached files (features, model, indices)')
    parser.add_argument('--features', action='store_true', help='Clear only feature files (resnet_features.npy)')
    parser.add_argument('--model', action='store_true', help='Clear only model files (knn_final_model.joblib)')
    parser.add_argument('--indices', action='store_true', help='Clear only feature selection indices (selected_indices.npy)')
    parser.add_argument('--evaluation', action='store_true', help='Clear only evaluation results (evaluation_results.joblib)')
    
    args = parser.parse_args()
    
    # Define file paths
    cache_files = {
        'features': ['resnet_features.npy', 'data_df.pkl'],
        'model': ['knn_final_model.joblib', 'scaler.joblib'],
        'indices': ['selected_indices.npy'],
        'evaluation': ['evaluation_results.joblib']
    }
    
    # If no specific flags, show help
    if not (args.all or args.features or args.model or args.indices or args.evaluation):
        parser.print_help()
        return
        
    # Process deletion based on arguments
    files_to_delete = []
    
    if args.all:
        # Add all cache files to deletion list
        for file_group in cache_files.values():
            files_to_delete.extend(file_group)
    else:
        # Add specific groups
        if args.features:
            files_to_delete.extend(cache_files['features'])
        if args.model:
            files_to_delete.extend(cache_files['model'])
        if args.indices:
            files_to_delete.extend(cache_files['indices'])
        if args.evaluation:
            files_to_delete.extend(cache_files['evaluation'])
    
    # Delete the files
    deleted_count = 0
    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Successfully deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        else:
            print(f"File not found: {file}")
    
    # Summary
    print(f"\n{deleted_count} cache files deleted.")
    if deleted_count > 0:
        print("You can now retrain the specified components of your model.")

if __name__ == "__main__":
    main()