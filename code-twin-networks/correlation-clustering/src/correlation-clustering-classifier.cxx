#include <random>
#include <vector>
#include <fstream>
#include <iostream>

#include <andres/graph/complete-graph.hxx>
#include <andres/graph/partition-comparison.hxx>
#include "helpers/reader.hxx"

void analyseClassifierProtocol(){
    typedef andres::graph::CompleteGraph<> Graph;
    typedef andres::RandError<double> RandError;
    typedef andres::VariationOfInformation<double> VI;

    std::string filePath = "/run/media/dstein/Elements/models-3s-with-background/similarity/3s-AB-600-with-augments-380k/analysis/classifier_protocol.h5";

    std::vector<long > groundTruths;
    std::vector<long > predictions;
    std::vector<std::string > labels;
    std::vector<std::string > fileIds;
    size_t numberOfElements;

    helpers::readHDF5ClassifierProtocol(
            filePath,
            numberOfElements,
            groundTruths,
            predictions,
            labels,
            fileIds
    );

    size_t count = 0;
    size_t tp = 0;

    for (size_t index = 0; index < predictions.size(); ++index){
        if (predictions[index] == groundTruths[index]){
            tp += 1;
        }
        count += 1;
    }

    std::cout << "Classification Accuracy: " << (double)tp / (double)count << std::endl;


    // get TP, TN, FP, FN
    std::cout << "Using Partition Comparison Classes..." << std::endl;

    RandError randError(groundTruths.begin(), groundTruths.end(), predictions.begin(), false);
    VI vi(groundTruths.begin(), groundTruths.end(), predictions.begin(), false);

    std::cout << "Rand Metrics" << std::endl;
    std::cout << std::string (20, '=') << std::endl;
    std::cout << "Precision Cuts: " << randError.precisionOfCuts() << std::endl;
    std::cout << "Recall Cuts: " << randError.recallOfCuts() << std::endl;
    std::cout << "Precision Joins: " << randError.precisionOfJoins() << std::endl;
    std::cout << "Recall Joins: " << randError.recallOfJoins() << std::endl;
    std::cout << "Rand Index: " << randError.index() << std::endl;
    std::cout << "Rand Error: " << randError.error() << std::endl;

    std::cout << "False Joins (Rand class): " << randError.falseJoins() << std::endl;
    std::cout << "False Cuts (Rand class): " << randError.falseCuts() << std::endl;

    std::cout << std::endl;
    std::cout << "VariationOfInformation Metrics" << std::endl;
    std::cout << std::string (20, '=') << std::endl;
    std::cout << "VariationOfInformation: " << vi.value() << std::endl;
    std::cout << "VariationOfInformation False Cuts: " << vi.valueFalseCut() << std::endl;
    std::cout << "VariationOfInformation False Joins: " << vi.valueFalseJoin() << std::endl;
}

void analyseClassifierEvaluation(){
    typedef andres::graph::CompleteGraph<> Graph;
    typedef andres::RandError<double> RandError;
    typedef andres::VariationOfInformation<double> VI;

    std::string filePath = "/home/dstein/GitRepos/PhD/bird-song-recognition/src/birdnet-analyzer/split-600-A-test-min-sns-50/classification.h5";

    std::vector<long > groundTruths;
    std::vector<long > predictions;
    size_t numberOfElements;

    helpers::readHDF5ClassifierEvaluation(
            filePath,
            numberOfElements,
            groundTruths,
            predictions
    );

    size_t count = 0;
    size_t tp = 0;

    for (size_t index = 0; index < predictions.size(); ++index){
        if (predictions[index] == groundTruths[index]){
            tp += 1;
        }
        count += 1;
    }

    std::cout << "Classification Accuracy: " << (double)tp / (double)count << std::endl;


    // get TP, TN, FP, FN
    std::cout << "Using Partition Comparison Classes..." << std::endl;

    RandError randError(groundTruths.begin(), groundTruths.end(), predictions.begin(), false);
    VI vi(groundTruths.begin(), groundTruths.end(), predictions.begin(), false);

    std::cout << "Rand Metrics" << std::endl;
    std::cout << std::string (20, '=') << std::endl;
    std::cout << "Precision Cuts: " << randError.precisionOfCuts() << std::endl;
    std::cout << "Recall Cuts: " << randError.recallOfCuts() << std::endl;
    std::cout << "Precision Joins: " << randError.precisionOfJoins() << std::endl;
    std::cout << "Recall Joins: " << randError.recallOfJoins() << std::endl;
    std::cout << "Rand Index: " << randError.index() << std::endl;
    std::cout << "Rand Error: " << randError.error() << std::endl;

    std::cout << "False Joins (Rand class): " << randError.falseJoins() << std::endl;
    std::cout << "False Cuts (Rand class): " << randError.falseCuts() << std::endl;

    std::cout << std::endl;
    std::cout << "VariationOfInformation Metrics" << std::endl;
    std::cout << std::string (20, '=') << std::endl;
    std::cout << "VariationOfInformation: " << vi.value() << std::endl;
    std::cout << "VariationOfInformation False Cuts: " << vi.valueFalseCut() << std::endl;
    std::cout << "VariationOfInformation False Joins: " << vi.valueFalseJoin() << std::endl;
}


int main() {
//    analyseClassifierProtocol();
    analyseClassifierEvaluation();

    return 0;
}
