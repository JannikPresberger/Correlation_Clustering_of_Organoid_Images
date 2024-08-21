#pragma once

#include "global_data_types.h"
#include "image_processing.h"


void calculate_clustering_using_graph_implementation(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters);

void calculate_clustering(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters, bool use_optimized = true);

std::vector<int>* calculate_clustering_using_gurobi(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters, double threshold);

std::vector<int>* calculate_clustering_using_gurobi_with_lazy_constraints(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters, double threshold);

void find_organoid_cluster_representative(std::vector<Matching_Result>& all_matching_results, std::vector<Cluster>& all_clusters, std::vector<Cluster_Representative_Pair>& selected_cluster_representatives);