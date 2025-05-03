To keep features but retrain model with different parameters:
python clear_cache.py --model
python eaed-using-parallel-cnn-transformer.py


"""
NO NEED TO RUN EXextract_evaluation_metrics.py ANYMORE

After making model changes and wanting updated evaluation:
python clear_cache.py --evaluation
python extract_evaluation_metrics.py

To evaluate your trained model and generate reports:
python extract_evaluation_metrics.py
"""