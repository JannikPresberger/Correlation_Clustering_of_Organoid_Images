# Code for the paper "Correlation Clustering of Organoids"
This directory contains the code using the partial quadratic assignment problem for correlating and clustering organoid images.

Before using this application one needs to copy the datasets containing the organoid images into the folder data/images/image_sets

A file containing all keypoints for image set "organoid_dataset_100_split_2(i.e. Test-100)" is contained in the folder data/feature_vectors.
Similarly a file containing the computed matchings for "organoid_dataset_100_split_2(i.e. Test-100)" is contained in the folder data/matching_results

By default the application will read these two files and display the matchings.

If one wishes to generate (and write) the keypoints for a different image set you have to pass the argument --read_features=false, (--write_features=true) and --image_set="name of image set"
To pass the keypoints to the application pass: --read_features=false, --features="name of file"

To calculate (and write) new matchings pass: --read_matchings=false, (--write_matchings=true) 