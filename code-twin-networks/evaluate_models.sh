# Image Models
python evaluation.py --test-100-dir ../data/test-100 --test-30-dir ../data/test-30 --input-type histogram --model-dir ./models/tni-p0.0
python evaluation.py --test-100-dir ../data/test-100 --test-30-dir ../data/test-30 --input-type histogram --model-dir ./models/tni-p.02

# Histogram Models
python evaluation.py --test-100-dir ../data/test-100 --test-30-dir ../data/test-30 --input-type images --model-dir ./models/tnh
