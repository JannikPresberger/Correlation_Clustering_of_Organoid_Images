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
#include <filesystem>
#include "helpers/reader.hxx"
#include "andres/graph/multicut/ilp.hxx"
#include "andres/ilp/gurobi.hxx"
#include "helpers/argparse.hxx"

typedef andres::graph::CompleteGraph<> Graph;
typedef andres::RandError<double> RandError;
typedef andres::VariationOfInformation<double> VariationOfInformation;


std::vector<size_t> edgeLabelsToNodeLabels(
        Graph const & graph,
        std::vector<size_t> const & edgeLabels
        ){
    // ground truth partition
    CutSolutionMask<size_t> edgeLabelMask(edgeLabels);
    andres::graph::ComponentsBySearch<Graph> componentsBySearch;
    componentsBySearch.build(graph, edgeLabelMask);

    return componentsBySearch.labels_;

}

template<class T>
void findLocalSearchEdgeLabels(
        Graph const & graph,
        std::vector<T> const & edgeCosts,
        std::vector<size_t> & edgeLabels
        ){
    std::cout << "Applying Additive Edge Contraction..." << std::endl;
    andres::graph::multicut::greedyAdditiveEdgeContractionCompleteGraph(
            graph,
            edgeCosts,
            edgeLabels
    );

    std::cout << "Applying Kernighan Lin..." << std::endl;
    andres::graph::multicut::kernighanLin(
            graph,
            edgeCosts,
            edgeLabels,
            edgeLabels
    );
    std::cout << "Applying Kernighan Lin done..." << std::endl;
}

std::vector<std::string> findUniqueSubsets(
        std::vector<std::string> const & subsetData
        ){
    // find unique subsets
    std::vector<std::string> uniqueSubsets;
    for (auto & subset: subsetData){
        if (std::find(uniqueSubsets.begin(), uniqueSubsets.end(), subset) == uniqueSubsets.end()){
            uniqueSubsets.push_back(subset);
        }
    }

    return uniqueSubsets;
}

void analyzeSubsetMetrics(
        Graph const & graph,
        std::vector<std::string> const & subsetData,
        std::vector<std::string> const & uniqueSubsets,
        std::vector<size_t> const & edgeLabels,
        std::vector<size_t> const & truthPartitionEdgeLabels,
        std::ostream & out
        ){
    size_t numberOfElements = subsetData.size();
    size_t numberOfUniqueSubsets = uniqueSubsets.size();
    size_t numberOfUniqueSubsetCombinations = numberOfUniqueSubsets*(numberOfUniqueSubsets + 1) / 2;
    // indexing
    // 0 1 2
    // - 3 4
    // - - 5

    std::vector<size_t> trueJoins(numberOfUniqueSubsetCombinations, 0);
    std::vector<size_t> trueCuts(numberOfUniqueSubsetCombinations, 0);
    std::vector<size_t> falseJoins(numberOfUniqueSubsetCombinations, 0);
    std::vector<size_t> falseCuts(numberOfUniqueSubsetCombinations, 0);
    std::vector<size_t> counts(numberOfUniqueSubsetCombinations, 0);

    size_t index = 0;
    for (size_t i = 0; i < uniqueSubsets.size(); ++i){
        for (size_t j = 0; j <= i; ++j){
            const std::string& subset1 = uniqueSubsets[i];
            const std::string& subset2 = uniqueSubsets[j];

            std::vector<size_t > nodeLabelsSubset;
            std::vector<size_t > truthNodeLabelsSubset;

            // iterate over all pairs of elements
            for (size_t l = 0; l < numberOfElements; ++l){
                for (size_t k = 0; k < l; ++k) {
                    if (subsetData[l] == subset1 && subsetData[k] == subset2) {
                        counts[index] += 1;

                        auto edge = graph.findEdge(l, k);
                        if (!edge.first) {
                            throw std::runtime_error("Edge not found.");
                        }

                        size_t predictedEdgeLabel = edgeLabels[edge.second];
                        size_t groundTruthEdgeLabel = truthPartitionEdgeLabels[edge.second];

                        if (predictedEdgeLabel == 1 && groundTruthEdgeLabel == 1) {
                            // true cut
                            trueCuts[index] += 1;
                        } else if (predictedEdgeLabel == 1 && groundTruthEdgeLabel == 0) {
                            // false cut
                            falseCuts[index] += 1;
                        } else if (predictedEdgeLabel == 0 && groundTruthEdgeLabel == 1) {
                            // false join
                            falseJoins[index] += 1;
                        } else if (predictedEdgeLabel == 0 && groundTruthEdgeLabel == 0) {
                            // true join
                            trueJoins[index] += 1;
                        } else {
                            throw std::runtime_error("Incorrect ground truth or edge labels.");
                        }
                    }
                }
            }
            index++;
        }
    }

    index = 0;
    for (size_t i = 0; i < uniqueSubsets.size(); ++i){
        for (size_t j = 0; j <= i; ++j){
            std::string subset1 = uniqueSubsets[i];
            std::string subset2 = uniqueSubsets[j];


            double precisionCuts = static_cast<double>(trueCuts[index]) / static_cast<double>(trueCuts[index] + falseCuts[index]);
            double precisionJoins = static_cast<double>(trueJoins[index]) / static_cast<double>(trueJoins[index] + falseJoins[index]);
            double recallCuts = static_cast<double>(trueCuts[index]) / static_cast<double>(trueCuts[index] + falseJoins[index]);
            double recallJoins = static_cast<double>(trueJoins[index]) / static_cast<double>(trueJoins[index] + falseCuts[index]);
            double randIndex = static_cast<double>(trueJoins[index] + trueCuts[index]) / static_cast<double>(trueJoins[index] + trueCuts[index] + falseJoins[index] + falseCuts[index]);
            double f1Cuts = 2 / (std::pow(precisionCuts, -1) + std::pow(recallCuts, -1));
            double f1Joins = 2/ (std::pow(precisionJoins, -1) + std::pow(recallJoins, -1));


            out << "Metrics for " << subset1 << "-" << subset2 << std::endl;
            out << "(TC, TJ, FC, FJ)" << trueCuts[index] << " " << trueJoins[index] << " " << falseCuts[index] << " " << falseJoins[index] << std::endl;
            out << "RI: " << randIndex << std::endl;
            out << "PC: " << precisionCuts << std::endl;
            out << "RC: " << recallCuts << std::endl;
            out << "PJ: " << precisionJoins << std::endl;
            out << "RJ: " << recallJoins << std::endl;
            out << "F1C: " << f1Cuts << std::endl;
            out << "F1J: " << f1Joins << std::endl;

            index++;
        }
    }

    size_t trueJoinsTotal = std::accumulate(trueJoins.begin(), trueJoins.end(), decltype(trueJoins)::value_type (0));
    size_t trueCutsTotal = std::accumulate(trueCuts.begin(), trueCuts.end(), decltype(trueCuts)::value_type (0));
    size_t falseJoinsTotal = std::accumulate(falseJoins.begin(), falseJoins.end(), decltype(falseJoins)::value_type (0));
    size_t falseCutsTotal = std::accumulate(falseCuts.begin(), falseCuts.end(), decltype(falseCuts)::value_type (0));

    double precisionCuts = static_cast<double>(trueCutsTotal) / static_cast<double>(trueCutsTotal+ falseCutsTotal);
    double precisionJoins = static_cast<double>(trueJoinsTotal) / static_cast<double>(trueJoinsTotal + falseJoinsTotal);
    double recallCuts = static_cast<double>(trueCutsTotal) / static_cast<double>(trueCutsTotal + falseJoinsTotal);
    double recallJoins = static_cast<double>(trueJoinsTotal) / static_cast<double>(trueJoinsTotal + falseCutsTotal);
    double randIndex = static_cast<double>(trueJoinsTotal + trueCutsTotal) / static_cast<double>(trueJoinsTotal + trueCutsTotal + falseJoinsTotal + falseCutsTotal);
    double f1Cuts = 2 / (std::pow(precisionCuts, -1) + std::pow(recallCuts, -1));
    double f1Joins = 2/ (std::pow(precisionJoins, -1) + std::pow(recallJoins, -1));


    out << "Metrics for all SUBSETS" << std::endl;
    out << "RI: " << randIndex << std::endl;
    out << "PC: " << precisionCuts << std::endl;
    out << "RC: " << recallCuts << std::endl;
    out << "PJ: " << precisionJoins << std::endl;
    out << "RJ: " << recallJoins << std::endl;
    out << "F1C: " << f1Cuts << std::endl;
    out << "F1J: " << f1Joins << std::endl;
}


void analyzeSubsetVariationOfInformation(
        std::vector<std::string> const & subsetData,
        std::vector<std::string> const & uniqueSubsets,
        std::vector<size_t > const & nodeLabels,
        std::vector<size_t > const & truthNodeLabels,
        std::ostream & out
){
    size_t numberOfElements = subsetData.size();
    std::vector<double > variationOfInformationFalseCuts(uniqueSubsets.size(), 0);
    std::vector<double > variationOfInformationFalseJoins(uniqueSubsets.size(), 0);

    std::vector<size_t > nodeLabelsSubset;
    std::vector<size_t > truthNodeLabelsSubset;

    for (size_t i = 0; i < uniqueSubsets.size(); ++i){

        for (size_t k = 0; k < numberOfElements; ++k){
            if (subsetData[k] == uniqueSubsets[i]){
                nodeLabelsSubset.push_back(nodeLabels[k]);
                truthNodeLabelsSubset.push_back(truthNodeLabels[k]);
            }
        }

        VariationOfInformation vi(truthNodeLabelsSubset.begin(), truthNodeLabelsSubset.end(), nodeLabelsSubset.begin(), false);

        variationOfInformationFalseCuts[i] = vi.valueFalseCut();
        variationOfInformationFalseJoins[i] = vi.valueFalseJoin();

        nodeLabelsSubset.clear();
        truthNodeLabelsSubset.clear();
    }


    for (size_t i = 0; i < uniqueSubsets.size(); ++i){
            std::string const & subset = uniqueSubsets[i];

            out << "VI Metrics for " << subset << std::endl;

            out << "VI: " << variationOfInformationFalseJoins[i] + variationOfInformationFalseCuts[i] << std::endl;
            out << "VI_FC: " << variationOfInformationFalseCuts[i] << std::endl;
            out << "VI_FJ: " << variationOfInformationFalseJoins[i] << std::endl;
        }

    VariationOfInformation  vi(truthNodeLabels.begin(), truthNodeLabels.end(), nodeLabels.begin(), false);

    out << "VI Metrics for all SUBSETS " << std::endl;

    out << "VI: " << vi.value() << std::endl;
    out << "VI_FC: " << vi.valueFalseCut() << std::endl;
    out << "VI_FJ: " << vi.valueFalseJoin() << std::endl;
}


std::map<size_t, std::map<size_t , size_t >> calculateOverlaps(
        std::vector<size_t> const & truthNodeLabels,
        std::vector<size_t> const & predictedNodeLabels
        ){

    std::map<size_t , std::map<size_t , size_t >> overlaps;

    for (size_t index = 0; index < predictedNodeLabels.size(); ++index){
        size_t predictedClusterId = predictedNodeLabels[index];
        size_t truthClusterId = truthNodeLabels[index];

        if (overlaps.find(predictedClusterId) == overlaps.end()){
            overlaps[predictedClusterId] = std::map<size_t , size_t >();
        }

        if (overlaps[predictedClusterId].find(truthClusterId) == overlaps[predictedClusterId].end()){
            overlaps[predictedClusterId][truthClusterId] = 0;
        }

        overlaps[predictedClusterId][truthClusterId] += 1;
    }

    return overlaps;
}

void writeSolutionFile(
        std::string const & outFilePath,
        std::vector<size_t> const & localSearchNodeLabels,
        std::vector<size_t> const & groundTruthNodeLabels,
        std::vector<std::string> const & subsetData,
        std::vector<std::string> const & labelData,
        std::vector<std::string> const & fileIds
) {
    auto overlaps = calculateOverlaps(
            groundTruthNodeLabels,
            localSearchNodeLabels
            );

    std::ofstream predictedClusterFile(outFilePath);
    predictedClusterFile << "index,predClusterId,truthClusterId,subset,overlap,truthLabel,fileId" << std::endl;

    for (size_t index = 0; index < localSearchNodeLabels.size(); ++index){
        size_t predId = localSearchNodeLabels[index];
        size_t truthId = groundTruthNodeLabels[index];
        std::string const &subset = subsetData[index];
        std::string const &truthLabel = labelData[index];
        std::string const &fileId = fileIds[index];
        size_t overlap = overlaps[predId][truthId];

        predictedClusterFile << index << "," << predId << "," << truthId << "," << subset << "," << overlap << "," << truthLabel << "," << fileId << std::endl;
    }
    predictedClusterFile.close();
}


template<class T = float>
inline
void
computeROCCurveCuts(
        std::vector<T> const &edgeProbabilities,
        std::vector<size_t> const &groundTruthEdgeLabels,
        size_t thresholds,
        std::string const &fileName
) {
    std::ofstream rocFile(fileName);
    rocFile << "threshold,precision,recall" << std::endl;

    for (size_t i = 0; i < thresholds; ++i) {
        // TJ, TC, FJ, FC
        size_t tj = 0;
        size_t tc = 0;
        size_t fj = 0;
        size_t fc = 0;

        T threshold = static_cast<T>(i) / static_cast<T>(thresholds - 1);

        for (size_t edge = 0; edge < edgeProbabilities.size(); ++edge) {

            size_t gtLabel = groundTruthEdgeLabels[edge];

            if (gtLabel == 2)
                continue;

            size_t pred = edgeProbabilities[edge] >= threshold ? 1 : 0;

            if (gtLabel == 1 && pred == 1) {
                tc += 1;
            } else if (gtLabel == 1 && pred == 0) {
                fj += 1;
            } else if (gtLabel == 0 && pred == 1) {
                fc += 1;
            } else if (gtLabel == 0 && pred == 0) {
                tj += 1;
            }
        }

        T precision;
        if (tc + fc == 0){
            precision = 1;
        } else {
            precision = static_cast<T>(tc) / static_cast<T>(tc + fc);
        }
        T recall = static_cast<T>(tc) / static_cast<T>(tc + fj);

        rocFile << threshold << "," << precision << "," << recall << std::endl;
    }

    rocFile.close();
}



template<class T = float>
inline
void
computeROCCurveJoins(
        std::vector<T> const &edgeProbabilities,
        std::vector<size_t> const &groundTruthEdgeLabels,
        size_t thresholds,
        std::string const &fileName
) {

    std::ofstream rocFile(fileName);
    rocFile << "threshold,precision,recall" << std::endl;

    for (size_t i = 0; i < thresholds; ++i) {
        // TJ, TC, FJ, FC
        size_t tj = 0;
        size_t tc = 0;
        size_t fj = 0;
        size_t fc = 0;

        T threshold = static_cast<T>(i) / static_cast<T>(thresholds - 1);

        for (size_t edge = 0; edge < edgeProbabilities.size(); ++edge) {

            size_t gtLabel = groundTruthEdgeLabels[edge];

            if (gtLabel == 2)
                continue;

            size_t pred = edgeProbabilities[edge] > threshold ? 1 : 0;

            if (gtLabel == 1 && pred == 1) {
                tc += 1;
            } else if (gtLabel == 1 && pred == 0) {
                fj += 1;
            } else if (gtLabel == 0 && pred == 1) {
                fc += 1;
            } else if (gtLabel == 0 && pred == 0) {
                tj += 1;
            }
        }

        T precision;
        if (tj + fj == 0){
            precision = 1;
        } else {
            precision = static_cast<T>(tj) / static_cast<T>(tj + fj);
        }
        T recall = static_cast<T>(tj) / static_cast<T>(tj + fc);

        rocFile << threshold << "," << precision << "," << recall << std::endl;
    }

    rocFile.close();
}

template<class T = float>
inline
void
computeROCCurveClustering(
        Graph const & graph,
        std::vector<T> const &edgeProbabilities,
        std::vector<size_t> const &groundTruthEdgeLabels,
        size_t thresholds,
        std::string const &fileName,
        T const epsilon
) {
    std::ofstream rocFile(fileName);
    rocFile << "logThreshold,pThreshold,precisionCuts,recallCuts,precisionJoins,recallJoins,VI,VI_FC,VI_FJ,durationNanoseconds" << std::endl;

    std::vector<T> edgeCosts(graph.numberOfEdges());

    std::cout << "Computing PR curves for cuts " << std::endl;

    T minimum = std::numeric_limits<T>::max();
    T maximum = std::numeric_limits<T>::min();

    // find min and max in edgeCosts
    for (size_t i = 0; i < edgeProbabilities.size(); ++i){
        T edgeCost = std::clamp(edgeProbabilities[i], epsilon, 1 - epsilon);
        edgeCost = std::log((1-edgeCost) / edgeCost);
        minimum = std::min(minimum, edgeCost);
        maximum = std::max(maximum, edgeCost);
    }

    // correct by epsilon
    minimum -= epsilon;
    maximum += epsilon;


    for (size_t i = 0; i < thresholds; ++i) {

        T interpolation = static_cast<T>(i) / static_cast<T>(thresholds - 1);
//        pThreshold = std::clamp(pThreshold, epsilon, 1 - epsilon);

        // interpolation = 0 -> add - minimum so all costs are positive
        // interpolation = 1 -> add -maximum so all costs are negative
        T threshold = - minimum + interpolation*(minimum - maximum);
//        T threshold = - std::log((1-pThreshold) / pThreshold);

        // TJ, TC, FJ, FC
        size_t tj = 0;
        size_t tc = 0;
        size_t fj = 0;
        size_t fc = 0;

        // probabilities resemble cut probabilities
        // p = epsilon -> join -> positive cost, but -log((1-epsilon)/epsilon) is negative

//        T threshold = -std::log((1-epsilon)/epsilon) + 2*std::log((1-epsilon)/ epsilon)*static_cast<T>(i) / static_cast<T>(thresholds - 1);

        for (size_t edgeIndex = 0; edgeIndex < edgeCosts.size(); ++edgeIndex){
            T prob = std::clamp(edgeProbabilities[edgeIndex], epsilon, 1 - epsilon);

            edgeCosts[edgeIndex] = std::log((1 - prob) / prob) + threshold;

        }
        std::vector<size_t > optimalEdgeLabels(graph.numberOfEdges(), 1);
        std::vector<double> doubleEdgeCosts(edgeCosts.begin(), edgeCosts.end());

        auto t1 = std::chrono::high_resolution_clock::now();
        andres::graph::multicut::ilp<andres::ilp::Gurobi>(graph, doubleEdgeCosts, optimalEdgeLabels, optimalEdgeLabels);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        std::vector<size_t > optimalNodeLabels = edgeLabelsToNodeLabels(graph, optimalEdgeLabels);
        std::vector<size_t > groundTruthNodeLabels = edgeLabelsToNodeLabels(graph, groundTruthEdgeLabels);

        VariationOfInformation vi(groundTruthNodeLabels.begin(), groundTruthNodeLabels.end(), optimalNodeLabels.begin(), false);

        for (size_t edge = 0; edge < edgeProbabilities.size(); ++edge) {

            size_t gtLabel = groundTruthEdgeLabels[edge];

            if (gtLabel == 2)
                continue;

            size_t pred = optimalEdgeLabels[edge];

            if (gtLabel == 1 && pred == 1) {
                tc += 1;
            } else if (gtLabel == 1 && pred == 0) {
                fj += 1;
            } else if (gtLabel == 0 && pred == 1) {
                fc += 1;
            } else if (gtLabel == 0 && pred == 0) {
                tj += 1;
            }
        }

        T precisionCuts;
        if (tc + fc == 0){
            precisionCuts = 1;
        } else {
            precisionCuts = static_cast<T>(tc) / static_cast<T>(tc + fc);
        }
        T recallCuts = static_cast<T>(tc) / static_cast<T>(tc + fj);

        T precisionJoins;
        if (tj + fj == 0){
            precisionJoins = 1;
        } else {
            precisionJoins = static_cast<T>(tj) / static_cast<T>(tj + fj);
        }
        T recallJoins = static_cast<T>(tj) / static_cast<T>(tj + fc);

        rocFile
            << threshold << "," << interpolation << ","
            << precisionCuts << "," << recallCuts << ","
            << precisionJoins << "," << recallJoins << ","
            << vi.value() << "," << vi.valueFalseCut() << "," << vi.valueFalseJoin() << ","
            << duration << std::endl;
    }

    rocFile.close();
}

template<class T>
void printObjectiveCosts(
        std::vector<T> const & edgeCosts,
        std::vector<size_t> const & edgeLabels
        ){
    T cutCost = 0;
    T joinCost = 0;
    T totalCost = 0;

    for (size_t i = 0; i < edgeCosts.size(); ++i){
        cutCost += edgeCosts[i]*edgeLabels[i];
        joinCost += edgeCosts[i]*(1-edgeLabels[i]);
        totalCost += edgeCosts[i];
    }

    std::cout << "Total Cost: " << totalCost << std::endl;
    std::cout << "Cut Cost: " << cutCost << std::endl;
    std::cout << "Join Cost: " << joinCost << std::endl;


}

template<class T>
void printCostStatistics(
        std::vector<T> const & edgeCosts
        ){
    T positiveCost = 0;
    T negativeCost = 0;

    for (size_t i = 0; i < edgeCosts.size(); ++i){

        if (edgeCosts[i] < -15 || edgeCosts[i] > 15){
            std::cout << "WARNING: " << edgeCosts[i] << std::endl;
        }

        if (edgeCosts[i] < 0){
            negativeCost += edgeCosts[i];
        } else {
            positiveCost += edgeCosts[i];
        }
    }
    std::cout << "Positive Cost: " << positiveCost << std::endl;
    std::cout << "Negative Cost: " << negativeCost << std::endl;
}


template<class T = float>
void fromHDF5(
        std::string const & hdfPath,
        bool const & writeToLogFile = false,
        T const epsilon = 1e-6
        ) {

    std::ofstream realOutFile;

    if (writeToLogFile){
        std::string localSearchClusterFilePath = hdfPath;
        localSearchClusterFilePath.replace(hdfPath.find(".h5"), std::string (".h5").length(), "_logs.csv");

        realOutFile.open(localSearchClusterFilePath, std::ios::out);
    }

    std::ostream & out = (writeToLogFile ? realOutFile : std::cout);


    size_t numberOfElements;
    std::vector<T> affinityData;
    std::vector<std::string> labelData;
    std::vector<std::string> subsetData;
    std::vector<std::string> fileIds;

    out << "1. Reading HDF5 File..." << std::endl;
    helpers::readHDF5CorrelationClustering(hdfPath, numberOfElements, affinityData, labelData, subsetData, fileIds);

    Graph graph(numberOfElements);

    std::vector<T> edgeCosts(graph.numberOfEdges(), 0);
    std::vector<T> edgeProbabilities(graph.numberOfEdges(), 0);
    std::vector<size_t> truthPartitionEdgeLabels(graph.numberOfEdges());

    std::vector<size_t > roundingEdgeLabels(graph.numberOfEdges(), 1);

    out << "2. Assembling edge costs..." << std::endl;
    // assemble edgeCosts

    // to save memory, we can overwrite the entries in affinities
    // affinities contain in [...numberOfElements] the costs for 0 against 0...numberOfElements
    //                       [numberOfElements...2numberOfElements] the costs for 1 against 0...numberOfElements
    for (size_t i = 0; i < numberOfElements; ++i){
        for (size_t j = 0; j < i ; ++j){
            // index1 - cost entry for i - j -> i*numberOfElements + j
            // index2 - cost entry of j - i -> j*numberOfElements + i
            size_t index1 = i*numberOfElements + j;
            size_t index2 = j*numberOfElements + i;
            // index1 = (n - 1)*n + n-2 = n**2 -n + n - 2 = n**2 - 2
            // index2 = (n-2)*n + n-1 = n**2 - n -1 = n(n-1) - 1

            // (numberOfVertices() - 1) * vertex0 - vertex0 * (vertex0 + 1) / 2 + vertex1 -

            std::pair<bool , size_t > graphEdgeIndex = graph.findEdge(j, i);
            if (!graphEdgeIndex.first){
                std::cout << j << " - " << i << std::endl;
                throw std::runtime_error("Graph Edge does not exist.");
            }
            edgeCosts[graphEdgeIndex.second] = std::clamp((affinityData[index1] + affinityData[index2]) / 2, epsilon, 1 - epsilon);
            // cost > 0.5 -> < 1 -> negative cost -> want to join -> should be positive cost (not cut)
            edgeCosts[graphEdgeIndex.second] = -std::log((1 - edgeCosts[graphEdgeIndex.second]) / edgeCosts[graphEdgeIndex.second]);

            // costs go from -log((1-epsilon)/epsilon) to + (log((1-epsilon)/epsilon)

            edgeProbabilities[graphEdgeIndex.second] = 1 - (affinityData[index1] + affinityData[index2]) / 2;

            if (edgeCosts[graphEdgeIndex.second] < 0){
                // want to cut
                roundingEdgeLabels[graphEdgeIndex.second] = 1;
            } else {
                roundingEdgeLabels[graphEdgeIndex.second] = 0;
            }

            truthPartitionEdgeLabels[graphEdgeIndex.second] = labelData[i] != labelData[j];

            if (graphEdgeIndex.second != (numberOfElements - 1) * j - j * (j + 1) / 2 + i - 1){
                std::cout << graphEdgeIndex.second << " - " << index1 << std::endl;
                throw std::runtime_error("Edge Indices do not match.");
            }
        }
    }

//
//    for (size_t c : edgeCosts){
//        std::cout << c << std::endl;
//    }
//    exit(1);

    affinityData.clear();
    affinityData.shrink_to_fit();

    if (numberOfElements <= 200){

        // compute ROC curve
        std::string rocFileName = hdfPath;
        rocFileName.replace(hdfPath.find(".h5"), std::string(".h5").length(),
                            "_cuts_roc.csv");
        computeROCCurveCuts(
                edgeProbabilities,
                truthPartitionEdgeLabels,
                1001,
                rocFileName
        );

        // compute ROC curve
        std::string rocFileNameJoins = hdfPath;
        rocFileNameJoins.replace(hdfPath.find(".h5"), std::string(".h5").length(),
                                 "_joins_roc.csv");
        computeROCCurveJoins(
                edgeProbabilities,
                truthPartitionEdgeLabels,
                1001,
                rocFileNameJoins
        );

        // compute ROC curve
        std::string rocFileNameClustering = hdfPath;
        rocFileNameClustering.replace(hdfPath.find(".h5"), std::string(".h5").length(),
                                      "_roc_clustering.csv");
        computeROCCurveClustering(
                graph,
                edgeProbabilities,
                truthPartitionEdgeLabels,
                101,
                rocFileNameClustering,
                epsilon
        );
    }

    printCostStatistics(edgeCosts);

    std::vector<std::size_t> localSearchEdgeLabels(graph.numberOfEdges(), 1);

    findLocalSearchEdgeLabels(graph, edgeCosts, localSearchEdgeLabels);

    printObjectiveCosts(edgeCosts, localSearchEdgeLabels);

    out << "3. Converting edge labels to node labels..." << std::endl;
    std::vector<size_t > groundTruthNodeLabels = edgeLabelsToNodeLabels(graph, truthPartitionEdgeLabels);
    std::vector<size_t > localSearchNodeLabels = edgeLabelsToNodeLabels(graph, localSearchEdgeLabels);


    VariationOfInformation vi(groundTruthNodeLabels.begin(), groundTruthNodeLabels.end(), localSearchNodeLabels.begin(), false);
    RandError randError(groundTruthNodeLabels.begin(), groundTruthNodeLabels.end(), localSearchNodeLabels.begin(), false);

    out << "4. Calculating Metrics..." << std::endl;

    out << std::string(100, '=') << std::endl;
    out << "4.1. Total Metrics" << std::endl;
    out << "RI: " << randError.index() << std::endl;
    out << "VI: " << vi.value() << std::endl;
    out << "VI_FC: " << vi.valueFalseCut() << std::endl;
    out << "VI_FJ: " << vi.valueFalseJoin() << std::endl;
    out << "PC: " << randError.precisionOfCuts() << std::endl;
    out << "RC: " << randError.recallOfCuts() << std::endl;
    out << "PJ: " << randError.precisionOfJoins() << std::endl;
    out << "RJ: " << randError.recallOfJoins() << std::endl;
    out << std::endl;

    std::vector<std::string> uniqueSubsets = findUniqueSubsets(subsetData);

    out << std::string(100, '=') << std::endl;
    out << "4.2. Before Correlation Clustering" << std::endl;

    analyzeSubsetMetrics(
            graph,
            subsetData,
            uniqueSubsets,
            roundingEdgeLabels,
            truthPartitionEdgeLabels,
            out
            );
    out << std::endl;


    out << std::string(100, '=') << std::endl;
    out << "4.3. After Correlation Clustering" << std::endl;
    analyzeSubsetMetrics(
            graph,
            subsetData,
            uniqueSubsets,
            localSearchEdgeLabels,
            truthPartitionEdgeLabels,
            out
    );
    analyzeSubsetVariationOfInformation(
            subsetData,
            uniqueSubsets,
            localSearchNodeLabels,
            groundTruthNodeLabels,
            out
            );
    out << std::endl;

    std::string localSearchClusterFilePath = hdfPath;
    localSearchClusterFilePath.replace(hdfPath.find(".h5"), std::string (".h5").length(), "_localSearchSolution.csv");
    writeSolutionFile(
            localSearchClusterFilePath,
            localSearchNodeLabels,
            groundTruthNodeLabels,
            subsetData,
            labelData,
            fileIds
    );

    if (numberOfElements <= 300){
        std::vector<size_t > optimalEdgeLabels(graph.numberOfEdges(), 1);
        // todo: really necessary to cast here?? ilp expects double iterator
        std::vector<double> doubleEdgeCosts(edgeCosts.begin(), edgeCosts.end());

        andres::graph::multicut::ilp<andres::ilp::Gurobi>(graph, doubleEdgeCosts, optimalEdgeLabels, optimalEdgeLabels);


        std::vector<size_t > optimalSolutionNodeLabels = edgeLabelsToNodeLabels(graph, optimalEdgeLabels);

        out << "4.4. Metrics for optimal solution" << std::endl;
        analyzeSubsetMetrics(
                graph,
                subsetData,
                uniqueSubsets,
                optimalEdgeLabels,
                truthPartitionEdgeLabels,
                out
        );
        analyzeSubsetVariationOfInformation(
                subsetData,
                uniqueSubsets,
                optimalSolutionNodeLabels,
                groundTruthNodeLabels,
                out
        );
        out << std::endl;

        std::string optimalClusterFilePath = hdfPath;
        optimalClusterFilePath.replace(hdfPath.find(".h5"), std::string (".h5").length(), "_optimalSolution.csv");
        writeSolutionFile(
                optimalClusterFilePath,
                optimalSolutionNodeLabels,
                groundTruthNodeLabels,
                subsetData,
                labelData,
                fileIds
        );

        if (writeToLogFile){
            realOutFile.close();
        }
    } else {
        std::cout << "Skipping computation of the optimal solution due to instance size." << std::endl;
    }

}


void runCorrelationClusteringAnalysis(std::filesystem::path const & modelDirectory){
    fromHDF5<float>(modelDirectory / "analysis/test.h5", true);
    fromHDF5<float>(modelDirectory / "analysis/test-unseen.h5", true);
    fromHDF5<float>(modelDirectory / "analysis/test-and-unseen.h5", true);
}


int main(int argc, char* argv[]) {
    argparse::ArgumentParser parser("correlation-clustering");
    parser.add_argument("--model-directory").default_value("../../models/tni-p0.0");
    parser.parse_args(argc, argv);

    std::cout << parser.get<std::string >("model-directory") << std::endl;

    runCorrelationClusteringAnalysis(parser.get<std::string >("model-directory"));
}