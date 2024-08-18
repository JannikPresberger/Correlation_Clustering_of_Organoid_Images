#include <random>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>

#include <andres/graph/complete-graph.hxx>
#include <andres/graph/multicut/greedy-additive.hxx>
#include <andres/graph/multicut/kernighan-lin.hxx>
#include <andres/graph/partition-comparison.hxx>
#include <andres/graph/multicut/edge-label-mask.hxx>
#include <andres/graph/components.hxx>


int main() {
    typedef andres::graph::CompleteGraph<> Graph;
    typedef andres::RandError<double> RandError;
    typedef andres::VariationOfInformation<double> VI;

    std::string filePath = "/run/media/dstein/Elements/new-models/models-w-species-list/similarity/birdclef-song-AB-3600-2s-frame-no-overlap-no-denoise-sns-50-nfft1024-nmels128-zscore-with-augments-resnet18/analysis/siamese_outputs_unseen_only.csv";

    //    "/run/media/dstein/Elements/new-models/models-w-species-list/similarity/birdclef-song-AB-600-2s-frame-no-overlap-no-denoise-sns-50-nfft1024-nmels128-zscore-no-augments-resnet18",
//            "/run/media/dstein/Elements/new-models/models-w-species-list/similarity/birdclef-song-AB-600-2s-frame-no-overlap-no-denoise-sns-50-nfft1024-nmels128-zscore-with-augments-resnet18",
//            "/run/media/dstein/Elements/new-models/models-w-species-list/similarity/birdclef-song-A-600-2s-frame-no-overlap-no-denoise-sns-50-nfft1024-nmels128-zscore-with-augments-resnet18",
//            "/run/media/dstein/Elements/new-models/models-w-species-list/similarity/birdclef-song-AB-3600-2s-frame-no-overlap-no-denoise-sns-50-nfft1024-nmels128-zscore-with-augments-resnet18",
//            "/run/media/dstein/Elements/new-models/models-w-species-list/similarity/birdclef-song-A-3600-2s-frame-no-overlap-no-denoise-sns-50-nfft1024-nmels128-zscore-with-augments-resnet18",
//    ]

    std::ifstream fin;
    fin.open(filePath);
    // pre read header

    std::string line;
    fin >> line;

    size_t tp= 0;
    size_t count = 0;

    std::cout << "Reading in file..." << std::endl;
    while (!fin.eof()){
        fin >> line;

        std::stringstream  ss(line);
        std::vector<std::string> values;

        std::string w;
        for (auto x : line){
            if ((x == ',') || (x == '\n')){
                values.push_back(w);
                w = "";
            } else {
                w = w + x;
            }
        }
        values.push_back(w);

        int fromIndex = std::stoi(values[0]);
        int toIndex = std::stoi(values[1]);
        if (fromIndex == toIndex) continue;

        count += 1;
        float pred = std::stof(values[2]);

        size_t pred_label;
        if (pred < 0.5){
            pred_label = 0;
        } else {
            pred_label = 1;
        }

        size_t groundTruth = std::stoi(values[3]);

        if (pred_label == groundTruth){
            tp += 1;
        }
    }

    std::cout << (double)tp / (double)count;





    return 0;
}
