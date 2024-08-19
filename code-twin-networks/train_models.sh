# TNI Models
python siamese_network.py --model-name tni-p0.0 --model-dir ./models --input-type images --augment 0.0 --data-dir ../data/train-100/ --val-data-dir ../data/test-100/
python siamese_network.py --model-name tni-p0.2 --model-dir ./models --input-type images --augment 0.2 --data-dir ../data/train-100/ --val-data-dir ../data/test-100/

# TNH Models
python siamese_network.py --model-name tnh --model-dir ./models --input-type histogram --data-dir ../data/train-100/ --val-data-dir ../data/test-100/
