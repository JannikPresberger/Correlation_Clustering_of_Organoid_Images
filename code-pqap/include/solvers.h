#pragma once

#include "gurobi_c++.h"
#include "cost_definitions.h"


void grb_print_quadratic_expr(GRBQuadExpr* quad_expr);

void solution_vector_from_gurobi_variables(std::vector<int>& solution_vector, GRBVar* all_model_variables, int num_features_points_in_model, int num_candidates_in_model);

std::vector<int>* assign_candidates_to_features_using_gurobi(std::vector<Feature_Point_Data>* features, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors, std::vector<int>* initial_solution = nullptr);

std::vector<int>* assign_candidates_to_features_using_gurobi_linearized(std::vector<Feature_Point_Data>* features, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors);

Matching_Result find_matching_between_two_organoids(std::vector<Feature_Point_Data>* features, std::vector<Feature_Point_Data>* candidates, All_Cost_Parameters& cost_params, bool solve_to_optimality = false , Assignment_Visualization_Data* viz_data = nullptr, int index_features_in_viz_data = 0, int index_candidates_in_viz_data = 0, std::vector<Op_Numbers_And_Runtime>* op_numbers = nullptr, Memory_Manager_Fixed_Size* mem_manager = nullptr);

void evaluate_all_possible_matchings(std::filesystem::path image_folder, const Input_Arguments args);

std::vector<int>* assign_candidates_to_features_using_qap_solver(std::vector<Feature_Point_Data>* features, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors);