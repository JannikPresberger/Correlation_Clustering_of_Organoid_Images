#include <random>
#include <vector>
#include <fstream>
#include <iostream>

#include <andres/graph/complete-graph.hxx>
#include <andres/graph/multicut/greedy-additive.hxx>
#include <andres/graph/multicut/kernighan-lin.hxx>
#include <andres/graph/partition-comparison.hxx>
#include <andres/graph/multicut/edge-label-mask.hxx>
#include <andres/graph/components.hxx>
#include "helpers/reader.hxx"
#include "andres/graph/multicut/ilp.hxx"
#include "andres/ilp/gurobi.hxx"

typedef andres::graph::CompleteGraph<> Graph;
typedef andres::RandError<double> RandError;
typedef andres::VariationOfInformation<double> VariationOfInformation;


std::vector<size_t> edgeLabelsToNodeLabels(
        Graph const &graph,
        std::vector<size_t> &edgeLabels
) {
    // ground truth partition
    CutSolutionMask<size_t> edgeLabelMask(edgeLabels);
    andres::graph::ComponentsBySearch<Graph> componentsBySearch;
    componentsBySearch.build(graph, edgeLabelMask);

    return componentsBySearch.labels_;

}

template<class T>
void findLocalSearchEdgeLabels(
        Graph const &graph,
        std::vector<T> const &edgeCosts,
        std::vector<size_t> &edgeLabels
) {
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
        std::vector<std::string> const &subsetData
) {
    // find unique subsets
    std::vector<std::string> uniqueSubsets;
    for (auto &subset: subsetData) {
        if (std::find(uniqueSubsets.begin(), uniqueSubsets.end(), subset) == uniqueSubsets.end()) {
            uniqueSubsets.push_back(subset);
        }
    }

    return uniqueSubsets;
}

void analyzeSubsetMetrics(
        Graph const &graph,
        std::vector<std::string> const &subsetData,
        std::vector<std::string> const &uniqueSubsets,
        std::vector<size_t> const &edgeLabels,
        std::vector<size_t> const &truthPartitionEdgeLabels,
        std::ostream &out
) {
    size_t numberOfElements = subsetData.size();
    size_t numberOfUniqueSubsets = uniqueSubsets.size();
    size_t numberOfUniqueSubsetCombinations = numberOfUniqueSubsets * (numberOfUniqueSubsets + 1) / 2;
    // indexing
    // 0 1 2
    // - 3 4
    // - - 5
    std::vector<size_t> trueJoins(numberOfUniqueSubsetCombinations, 0);
    std::vector<size_t> trueCuts(numberOfUniqueSubsetCombinations, 0);
    std::vector<size_t> falseJoins(numberOfUniqueSubsetCombinations, 0);
    std::vector<size_t> falseCuts(numberOfUniqueSubsetCombinations, 0);

    size_t index = 0;
    for (size_t i = 0; i < uniqueSubsets.size(); ++i) {
        for (size_t j = 0; j <= i; ++j) {
            const std::string &subset1 = uniqueSubsets[i];
            const std::string &subset2 = uniqueSubsets[j];

            std::vector<size_t> nodeLabelsSubset;
            std::vector<size_t> truthNodeLabelsSubset;

            // iterate over all pairs of elements
            for (size_t l = 0; l < numberOfElements; ++l) {
                for (size_t k = 0; k < l; ++k) {
                    if (subsetData[l] == subset1 && subsetData[k] == subset2) {

                        auto edge = graph.findEdge(l, k);
                        if (!edge.first) {
                            throw std::runtime_error("Edge not found.");
                        }

                        size_t predictedEdgeLabel = edgeLabels[edge.second];
                        size_t groundTruthEdgeLabel = truthPartitionEdgeLabels[edge.second];

                        // skip counting
                        if (groundTruthEdgeLabel == 2)
                            continue;

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
    for (size_t i = 0; i < uniqueSubsets.size(); ++i) {
        for (size_t j = 0; j <= i; ++j) {
            std::string const &subset1 = uniqueSubsets[i];
            std::string const &subset2 = uniqueSubsets[j];

            double precisionCuts =
                    static_cast<double>(trueCuts[index]) / static_cast<double>(trueCuts[index] + falseCuts[index]);
            double precisionJoins =
                    static_cast<double>(trueJoins[index]) / static_cast<double>(trueJoins[index] + falseJoins[index]);
            double recallCuts =
                    static_cast<double>(trueCuts[index]) / static_cast<double>(trueCuts[index] + falseJoins[index]);
            double recallJoins =
                    static_cast<double>(trueJoins[index]) / static_cast<double>(trueJoins[index] + falseCuts[index]);
            double randIndex = static_cast<double>(trueJoins[index] + trueCuts[index]) /
                               static_cast<double>(trueJoins[index] + trueCuts[index] + falseJoins[index] +
                                                   falseCuts[index]);
            double f1Cuts = 2 / (std::pow(precisionCuts, -1) + std::pow(recallCuts, -1));
            double f1Joins = 2 / (std::pow(precisionJoins, -1) + std::pow(recallJoins, -1));


            out << "Metrics for " << subset1 << "-" << subset2 << std::endl;
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
}


void analyzeSubsetVariationOfInformation(
        std::vector<std::string> const &subsetData,
        std::vector<std::string> const &uniqueSubsets,
        std::vector<size_t> const &nodeLabels,
        std::vector<size_t> const &truthNodeLabels,
        std::ostream &out
) {
    size_t numberOfElements = subsetData.size();
    std::vector<double> variationOfInformationFalseCuts(uniqueSubsets.size(), 0);
    std::vector<double> variationOfInformationFalseJoins(uniqueSubsets.size(), 0);

    std::vector<size_t> nodeLabelsSubset;
    std::vector<size_t> truthNodeLabelsSubset;

    for (size_t i = 0; i < uniqueSubsets.size(); ++i) {

        for (size_t k = 0; k < numberOfElements; ++k) {
            if (subsetData[k] == uniqueSubsets[i]) {
                nodeLabelsSubset.push_back(nodeLabels[k]);
                truthNodeLabelsSubset.push_back(truthNodeLabels[k]);
            }
        }

        VariationOfInformation vi(truthNodeLabelsSubset.begin(), truthNodeLabelsSubset.end(), nodeLabelsSubset.begin(),
                                  false);

        variationOfInformationFalseCuts[i] = vi.valueFalseCut();
        variationOfInformationFalseJoins[i] = vi.valueFalseJoin();

        nodeLabelsSubset.clear();
        truthNodeLabelsSubset.clear();
    }


    for (size_t i = 0; i < uniqueSubsets.size(); ++i) {
        std::string const &subset = uniqueSubsets[i];

        out << "VI Metrics for " << subset << std::endl;

        out << "VI: " << variationOfInformationFalseJoins[i] + variationOfInformationFalseCuts[i] << std::endl;
        out << "VI_FC: " << variationOfInformationFalseCuts[i] << std::endl;
        out << "VI_FJ: " << variationOfInformationFalseJoins[i] << std::endl;
    }
}


std::map<size_t, std::map<size_t, size_t >> calculateOverlaps(
        std::vector<size_t> const &truthNodeLabels,
        std::vector<size_t> const &predictedNodeLabels
) {

    std::map<size_t, std::map<size_t, size_t >> overlaps;

    for (size_t index = 0; index < predictedNodeLabels.size(); ++index) {
        size_t predictedClusterId = predictedNodeLabels[index];
        size_t truthClusterId = truthNodeLabels[index];

        if (overlaps.find(predictedClusterId) == overlaps.end()) {
            overlaps[predictedClusterId] = std::map<size_t, size_t>();
        }

        if (overlaps[predictedClusterId].find(truthClusterId) == overlaps[predictedClusterId].end()) {
            overlaps[predictedClusterId][truthClusterId] = 0;
        }

        overlaps[predictedClusterId][truthClusterId] += 1;
    }

    return overlaps;
}

void writeSolutionFile(
        std::string const &outFilePath,
        std::vector<size_t> const &localSearchNodeLabels,
        std::vector<std::string> const &subsetData,
        std::vector<std::string> const &labelData,
        std::vector<std::string> const &fileIds
) {

    std::ofstream predictedClusterFile(outFilePath);
    predictedClusterFile << "index,predClusterId,subset,truthLabel,fileId" << std::endl;

    for (size_t index = 0; index < localSearchNodeLabels.size(); ++index) {
        size_t predId = localSearchNodeLabels[index];
        std::string const &subset = subsetData[index];
        std::string const &truthLabel = labelData[index];
        std::string const &fileId = fileIds[index];

        predictedClusterFile << index << "," << predId << "," << subset << "," << truthLabel << "," << fileId
                             << std::endl;
    }
    predictedClusterFile.close();
}

inline
std::vector<std::string>
split(std::string const &s, std::string const &delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}


inline
std::pair<bool, int>
indexOf(std::vector<std::string> const &s, std::string const &element) {
    auto it = std::find(s.begin(), s.end(), element);

    if (it != s.end()) {
        return std::make_pair(true, std::distance(s.begin(), it));
    } else {
        return std::make_pair(false, -1);
    }
}

inline
std::vector<size_t>
readIndependentDecisions(
        Graph const &graph,
        std::string const &decisionsPath,
        std::vector<std::string> const &numericIds
) {
    std::vector<size_t> truthPartitionEdgeLabels(graph.numberOfEdges(), 3);

    std::string currentDecisionLine;

    std::ifstream decisionFile(decisionsPath);

    while (std::getline(decisionFile, currentDecisionLine)) {
        std::vector<std::string> tokens = split(currentDecisionLine, ",");
        if (tokens.size() == 4) {
            std::string id1 = tokens[0];
            std::string id2 = tokens[1];

            std::string decision = tokens[3];

            auto node1 = indexOf(numericIds, id1);
            auto node2 = indexOf(numericIds, id2);

            if (node1.first && node2.first) {
                auto edge = graph.findEdge(node1.second, node2.second);

                if (decision == "1") {
                    // same -> truth edge label is 0
                    truthPartitionEdgeLabels[edge.second] = 0;

                } else if (decision == "2") {
                    // unsure
                    truthPartitionEdgeLabels[edge.second] = 2;
                } else if (decision == "3") {
                    // different
                    truthPartitionEdgeLabels[edge.second] = 1;
                } else {
                    throw std::runtime_error("Unknown decision in decision file.");
                }
            }
        }
    }

    // double check truthPartitionEdgeLabels
    for (auto label: truthPartitionEdgeLabels) {
        if (label == 3) {
            throw std::runtime_error("Truth edge labels should not be 3 anymore.");
        }
    }

    return truthPartitionEdgeLabels;
}

inline
void
resolveConflicts(
        Graph const &graph,
        std::vector<size_t> &edgeLabels,
        std::ostream &out
) {
    std::set<size_t> edgeIndicesConflicted;

    for (size_t edge = 0; edge < graph.numberOfEdges(); ++edge) {

        if (edgeLabels[edge] == 1) {

            size_t v0 = graph.vertexOfEdge(edge, 0);
            size_t v1 = graph.vertexOfEdge(edge, 1);

            for (size_t i = 0; i < graph.numberOfVertices(); ++i) {
                if (v0 == i || v1 == i)
                    continue;

                size_t edge0 = graph.findEdge(v0, i).second;
                size_t edge1 = graph.findEdge(v1, i).second;

                if (edgeLabels[edge0] == 0 && edgeLabels[edge1] == 0) {
                    edgeIndicesConflicted.insert(edge0);
                    edgeIndicesConflicted.insert(edge1);
                    edgeIndicesConflicted.insert(edge);
                }
            }
        }
    }

    out << "Edges set to unsure (conflicts): " << edgeIndicesConflicted.size() << std::endl;

    for (auto edgeIndex: edgeIndicesConflicted) {
        edgeLabels[edgeIndex] = 2;
    }
}

inline
void
computeUndersegmentation(
        Graph const &graph,
        std::vector<size_t> const &decisions,
        std::vector<size_t> &out
) {
    std::deque<size_t> edgeQueue;

    // todo: check this again

    // 1. initialize with current edge values
    for (size_t edge = 0; edge < graph.numberOfEdges(); ++edge) {
        out[edge] = decisions[edge];
    }

    for (size_t edge = 0; edge < graph.numberOfEdges(); ++edge) {
        if (decisions[edge] != 2)
            continue;

        size_t v0 = graph.vertexOfEdge(edge, 0);
        size_t v1 = graph.vertexOfEdge(edge, 1);

        for (size_t k = 0; k < graph.numberOfVertices(); ++k) {
            if (v0 == k || v1 == k)
                continue;

            size_t edge0 = graph.findEdge(v0, k).second;
            size_t edge1 = graph.findEdge(v1, k).second;

            if (out[edge0] == 0 && out[edge1] == 0) {
                edgeQueue.push_back(edge);
            }
        }
    }

    // transitive closure
    while (!edgeQueue.empty()) {
        size_t edge = edgeQueue.front();
        edgeQueue.pop_front();

        size_t v0 = graph.vertexOfEdge(edge, 0);
        size_t v1 = graph.vertexOfEdge(edge, 1);

        for (size_t k = 0; k < graph.numberOfVertices(); ++k) {
            if (v0 == k || v1 == k)
                continue;

            size_t edge1 = graph.findEdge(v0, k).second;
            size_t edge2 = graph.findEdge(v1, k).second;

            if (out[edge1] == 2 && out[edge2] == 0) {
                if (std::find(edgeQueue.begin(), edgeQueue.end(), edge1) == edgeQueue.end())
                    edgeQueue.push_back(edge1);

            }
            if (out[edge2] == 2 && out[edge1] == 0) {
                if (std::find(edgeQueue.begin(), edgeQueue.end(), edge2) == edgeQueue.end())
                    edgeQueue.push_back(edge2);
            }
        }
    }


    // now there should not be any conflicts anymore, set all other remaining 2-edges to 0
    for (size_t edge = 0; edge < graph.numberOfEdges(); ++edge) {
        if (out[edge] != 2)
            continue;

        out[edge] = 0;
    }
}

template<class T = float>
inline
void
computeROCCurve(
        std::vector<T> const &edgeProbabilities,
        std::vector<size_t> const &groundTruthEdgeLabels,
        size_t thresholds,
        std::string const &fileName
) {

    std::ofstream rocFile(fileName);
    rocFile << "threshold,tj,tc,fj,fc" << std::endl;

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

        rocFile << threshold << "," << tj << "," << tc << "," << fj << "," << fc << std::endl;
    }

    rocFile.close();
}


template<class T = float>
void fromHDF5(
        std::string const &hdfPath,
        std::string const &decisionsPath,
        bool const &writeToLogFile = false
) {

    std::ofstream realOutFile;

    if (writeToLogFile) {
        std::string localSearchClusterFilePath = hdfPath;
        localSearchClusterFilePath.replace(hdfPath.find(".h5"), std::string(".h5").length(),
                                           "_independentDecisions_logs.txt");

        realOutFile.open(localSearchClusterFilePath, std::ios::out);
    }

    std::ostream &out = (writeToLogFile ? realOutFile : std::cout);


    size_t numberOfElements;
    std::vector<T> affinityData;
    std::vector<std::string> labelData;
    std::vector<std::string> subsetData;
    std::vector<std::string> fileIds;

    out << "1. Reading HDF5 File..." << std::endl;
    helpers::readHDF5CorrelationClustering(hdfPath, numberOfElements, affinityData, labelData, subsetData, fileIds);

    std::vector<std::string> numericIds(numberOfElements);
    for (size_t i = 0; i < fileIds.size(); ++i) {
        numericIds[i] = split(split(fileIds[i], "_").back(), ".")[0];
    }

    Graph graph(numberOfElements);

    std::vector<T> edgeCosts(graph.numberOfEdges(), 0);
    std::vector<T> edgeProbabilities(graph.numberOfEdges(), 0);
    std::vector<size_t> truthPartitionEdgeLabels = readIndependentDecisions(graph, decisionsPath, numericIds);
//    std::vector<size_t> undersegmentationTruth(truthPartitionEdgeLabels.size(), 0);
//    std::vector<size_t> oversegmentationTruth(truthPartitionEdgeLabels.size(), 0);

//    computeUndersegmentation(graph, truthPartitionEdgeLabels, undersegmentationTruth);

    resolveConflicts(graph, truthPartitionEdgeLabels, out);


    // compute over and under segmentation of truthPartitionEdgeLabels
    // check conflicts

    // up to here finished
    std::vector<size_t> roundingEdgeLabels(graph.numberOfEdges(), 1);

    out << "2. Assembling edge costs..." << std::endl;
    // assemble edgeCosts

    // to save memory, we can overwrite the entries in affinities
    // affinities contain in [...numberOfElements] the costs for 0 against 0...numberOfElements
    //                       [numberOfElements...2numberOfElements] the costs for 1 against 0...numberOfElements
    for (size_t i = 0; i < numberOfElements; ++i) {
        for (size_t j = 0; j < i; ++j) {
            // index1 - cost entry for i - j -> i*numberOfElements + j
            // index2 - cost entry of j - i -> j*numberOfElements + i
            size_t index1 = i * numberOfElements + j;
            size_t index2 = j * numberOfElements + i;
            // index1 = (n - 1)*n + n-2 = n**2 -n + n - 2 = n**2 - 2
            // index2 = (n-2)*n + n-1 = n**2 - n -1 = n(n-1) - 1

            // (numberOfVertices() - 1) * vertex0 - vertex0 * (vertex0 + 1) / 2 + vertex1 -

            std::pair<bool, size_t> graphEdgeIndex = graph.findEdge(j, i);
            if (!graphEdgeIndex.first) {
                std::cout << j << " - " << i << std::endl;
                throw std::runtime_error("Graph Edge does not exist.");
            }
            edgeCosts[graphEdgeIndex.second] = std::clamp((affinityData[index1] + affinityData[index2]) / 2,
                                                          static_cast<T>(1e-6), static_cast<T>(1 - 1e-6));
            // cost > 0.5 -> < 1 -> negative cost -> want to join -> should be positive cost (not cut)
            edgeCosts[graphEdgeIndex.second] = -std::log(
                    (1 - edgeCosts[graphEdgeIndex.second]) / edgeCosts[graphEdgeIndex.second]);

            edgeProbabilities[graphEdgeIndex.second] = 1 - (affinityData[index1] + affinityData[index2]) / 2;

            if (edgeCosts[graphEdgeIndex.second] < 0) {
                // want to cut
                roundingEdgeLabels[graphEdgeIndex.second] = 1;
            } else {
                roundingEdgeLabels[graphEdgeIndex.second] = 0;
            }

            if (graphEdgeIndex.second != (numberOfElements - 1) * j - j * (j + 1) / 2 + i - 1) {
                std::cout << graphEdgeIndex.second << " - " << index1 << std::endl;
                throw std::runtime_error("Edge Indices do not match.");
            }
        }
    }

    affinityData.clear();
    affinityData.shrink_to_fit();


    // compute ROC curve
    std::string rocFileName = hdfPath;
    rocFileName.replace(hdfPath.find(".h5"), std::string(".h5").length(),
                        "_roc.csv");
    computeROCCurve(
            edgeProbabilities,
            truthPartitionEdgeLabels,
            1001,
            rocFileName
    );

    std::vector<std::size_t> localSearchEdgeLabels(graph.numberOfEdges(), 1);

    findLocalSearchEdgeLabels(graph, edgeCosts, localSearchEdgeLabels);

    out << "3. Converting edge labels to node labels..." << std::endl;
    std::vector<size_t> localSearchNodeLabels = edgeLabelsToNodeLabels(graph, localSearchEdgeLabels);


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
//    analyzeSubsetVariationOfInformation(
//            subsetData,
//            uniqueSubsets,
//            localSearchNodeLabels,
//            groundTruthNodeLabels,
//            out
//    );
    out << std::endl;

    std::string localSearchClusterFilePath = hdfPath;
    localSearchClusterFilePath.replace(hdfPath.find(".h5"), std::string(".h5").length(), "_localSearchSolution.csv");
    writeSolutionFile(
            localSearchClusterFilePath,
            localSearchNodeLabels,
            subsetData,
            labelData,
            fileIds
    );

    std::vector<size_t> optimalEdgeLabels(graph.numberOfEdges(), 1);
    std::vector<double> doubleEdgeCosts(edgeCosts.begin(), edgeCosts.end());

    andres::graph::multicut::ilp<andres::ilp::Gurobi>(graph, doubleEdgeCosts, optimalEdgeLabels, optimalEdgeLabels);


    std::vector<size_t> optimalSolutionNodeLabels = edgeLabelsToNodeLabels(graph, optimalEdgeLabels);

    out << "4.4. Metrics for optimal solution" << std::endl;
    analyzeSubsetMetrics(
            graph,
            subsetData,
            uniqueSubsets,
            optimalEdgeLabels,
            truthPartitionEdgeLabels,
            out
    );
//    analyzeSubsetVariationOfInformation(
//            subsetData,
//            uniqueSubsets,
//            optimalSolutionNodeLabels,
//            groundTruthNodeLabels,
//            out
//    );
    out << std::endl;


    std::string optimalClusterFilePath = hdfPath;
    optimalClusterFilePath.replace(hdfPath.find(".h5"), std::string(".h5").length(),
                                   "_independentDecisions_optimalSolution.csv");
    writeSolutionFile(
            optimalClusterFilePath,
            optimalSolutionNodeLabels,
            subsetData,
            labelData,
            fileIds
    );

    if (writeToLogFile) {
        realOutFile.close();
    }
}

void runCorrelationClusteringAnalysis() {
    std::vector<std::string> validationSets{
            "C3-nc", "C4-nc", "C5-nc",
            "C6-nc", "C7-nc", "C8-nc"
    };

    std::vector<std::string> decisionFiles{
            "/home/dstein/GitRepos/PhD/organoid_clustering_tool/pairwise_comparison_results/pairwise_comparison_15_11_10_40_C03_no_garbage.csv",
            "/home/dstein/GitRepos/PhD/organoid_clustering_tool/pairwise_comparison_results/pairwise_comparison_15_11_12_58_C04_no_garbage.csv",
            "/home/dstein/GitRepos/PhD/organoid_clustering_tool/pairwise_comparison_results/pairwise_comparison_15_11_15_22_C05_no_garbage.csv",
            "/home/dstein/GitRepos/PhD/organoid_clustering_tool/pairwise_comparison_results/pairwise_comparison_15_11_14_14_C06_no_garbage.csv",
            "/home/dstein/GitRepos/PhD/organoid_clustering_tool/pairwise_comparison_results/pairwise_comparison_16_11_11_0_C07_no_garbage.csv",
            "/home/dstein/GitRepos/PhD/organoid_clustering_tool/pairwise_comparison_results/pairwise_comparison_16_11_16_15_C08_no_garbage.csv"
    };

    std::vector<std::string> models{
            "organoids-p0.0-256x256",
            "organoids-p0.0-256x256-squarePad",
            "organoids-p0.0-256x256-squarePad-res34",
            "organoids-p0.2-256x256",
            "organoids-p0.2-256x256-squarePad",
            "organoids-p0.4-256x256",
            "organoids-p0.4-256x256-squarePad",
            "organoids-p0.4-256x256-squarePad-res34"
    };

    for (auto &model: models) {
        for (size_t i = 0; i < validationSets.size(); ++i) {
            std::string validationSet = validationSets[i];
            std::string decisionFile = decisionFiles[i];
            std::string path = "/home/dstein/GitRepos/PhD/organoid-matching/models/";
            path += model;
            path += "/analysis/val-";
            path += validationSet;
            path += ".h5";

            fromHDF5<float>(path, decisionFile, true);
        }
    }
}


int main() {
    runCorrelationClusteringAnalysis();
}
