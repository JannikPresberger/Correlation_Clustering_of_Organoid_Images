mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
make


echo "Running CC Algorithms for ../../models/tni-p0.0/"
./correlation-clustering-subsets --model-directory ../../models/tni-p0.0/

echo "Running CC Algorithms for ../../models/tni-p0.2/"
./correlation-clustering-subsets --model-directory ../../models/tni-p0.2/

echo "Running CC Algorithms for ../../models/tnh/"
./correlation-clustering-subsets --model-directory ../../models/tnh/

