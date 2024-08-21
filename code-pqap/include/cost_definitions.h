#pragma once

typedef struct Feature_Point_Data Feature_Point_Data;

#include "utils.h"
#include "image_processing.h"
#include "memory_manager.h"


//#define STATIC_COST_OFFSET -8//-4.0
//#define DISTANCE_DIFFERENCE_COST_COEFFICIENT 10.0
//#define DISTANCE_DIFFERENCE_COST_COEFFICIENT 16.0
//#define COLOR_DIFFERENCE_COST_COEFFICIENT 10.0
//#define COLOR_DIFFERENCE_COST_COEFFICIENT 8.0

// currently in the calculation for the pairwise costs we only look at the angle difference which gets normalized by 180 degrees
// hence subtracting eg. 0.1 means we reward everything that is closer then 0.1 x 180  = 18 degrees
//#define ANGLE_DIFFERENCE_COST_COEFFICIENT 1.0
//#define STATIC_PAIRWISE_COST_OFFSET -4//-0.05

#define DEFAULT_COLOR_COST_OFFSET 0.2
#define DEFAULT_DIST_COST_OFFSET 0.2
#define DEFAULT_ANGLE_COST_OFFSET 0.2
#define DEFAULT_COLOR_TO_DIST_WEIGHT 0.5
#define DEFAULT_UNARY_TO_QUADR_WEIGHT 0.5

typedef struct Cost_Breakdown{
    double total_cost;
    double theoretical_optimal_cost;
    double cost_from_color_differences;
    double cost_from_distance_differences;
    double cost_from_angle_differences;
    double unary_cost_offsets;
    double pairwise_cost_offsets;
    double percentage_not_assigned_features;
}Cost_Breakdown;

typedef struct Instance_Cost_Pair{
    std::vector<int>* instance;
    double cost;
    double heuristic_cost_for_remaining_decision;
    double total_cost;
    uint16_t next_row_to_make_decision;
    uint16_t total_num_assignments; 
}Instance_Cost_Pair;

typedef struct Instance_Cost_Pair_Cus_Mem{
    Memory_Element instance_mem_elem;
    double cost;
    double heuristic_cost_for_remaining_decision;
    double total_cost;
    uint16_t next_row_to_make_decision;
    uint16_t total_num_assignments; 
}Instance_Cost_Pair_Cus_Mem;

typedef enum {
    BaB_BASIC_SEARCH,
    BaB_GREEDY_SEARCH_FIRST,
    BaB_GREEDY_SEARCH_ONLY,
    BaB_CYCLIC_GREEDY_SEARCH_ONLY
}BaB_Search_Modes;

typedef enum {
    SINGLE_INSTANCE_LINEAR_COST,
    SINGLE_INSTANCE_QUADRATIC_COST,
    SINGLE_INSTANCE_TOTAL_COST
}Single_Instance_Cost_Type;

typedef struct Cost_Rescale_Factors{
    //double unary_cost_rescale;
    //double unary_color_rescale;
    //double unary_distance_rescale;
    //double pairwise_cost_rescale;
    //double pairwise_angle_rescale;


    double color_offset; // gamma
    double dist_offset; // gamma' 
    double angle_offset; // gamma'
    double color_to_dist_weight; // theta
    double unary_to_to_quadr_weight; // lambda

    double unary_prefactor; // (1 - lambda)/ num_unary_terms
    double quadr_prefactor; // lambda/ num_quadr_terms

    double heuristic_unary_cost; // unary_prefactor * ( theta * color_offset + (1 - theta) * dist_offset)
    double heurisitc_quadr_cost; // quadr_prefactor * angle_offset
}Cost_Rescale_Factors;

struct Queue_Cost_Compare{

    bool operator()(const Instance_Cost_Pair& lhs, const Instance_Cost_Pair& rhs){

        if(lhs.next_row_to_make_decision == rhs.next_row_to_make_decision){
            return lhs.total_cost > rhs.total_cost;
        }
        return lhs.next_row_to_make_decision < rhs.next_row_to_make_decision;
    }
};

struct Queue_Cost_Compare_Cus_Mem{

    bool operator()(const Instance_Cost_Pair_Cus_Mem& lhs, const Instance_Cost_Pair_Cus_Mem& rhs){

        if(lhs.next_row_to_make_decision == rhs.next_row_to_make_decision){
            return lhs.total_cost > rhs.total_cost;
        }
        return lhs.next_row_to_make_decision < rhs.next_row_to_make_decision;
    }
};


void apply_greedy_assignment_transposition(std::vector<int>* best_instance, Cost_Rescale_Factors cost_rescale_factors, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates);

Cost_Rescale_Factors get_cost_rescale_factors(int instance_size, All_Cost_Parameters& all_cost_params);

float get_distance_between_two_feature_points(Feature_Point_Data& feature_point_1,Feature_Point_Data& feature_point_2);

double calculate_cost_for_two_feature_points(Feature_Point_Data& feature_point_1,Feature_Point_Data& feature_point_2, Cost_Rescale_Factors cost_rescale_factors);
double calculate_cost_for_two_feature_points_using_cost_params(Feature_Point_Data& feature_point_1,Feature_Point_Data& feature_point_2, Cost_Rescale_Factors cost_rescale_factors,All_Cost_Parameters& cost_params);

double calculate_cost_for_single_feature_point_pair_assignment(Feature_Point_Data& x1, Feature_Point_Data& d1, Feature_Point_Data& x2, Feature_Point_Data& d2, Cost_Rescale_Factors cost_rescale_factors);
double calculate_cost_for_single_feature_point_pair_assignment_using_cost_params(Feature_Point_Data& x1, Feature_Point_Data& d1, Feature_Point_Data& x2, Feature_Point_Data& d2, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params);

void get_cost_breakdown_for_two_feature_points(Feature_Point_Data& feature_point_1,Feature_Point_Data& feature_point_2, double& cost_from_color_diff, double& cost_from_dist_diff, double& cost_offset, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params);

void get_cost_breakdown_for_single_feature_point_pair_assignment(Feature_Point_Data& x1, Feature_Point_Data& d1, Feature_Point_Data& x2, Feature_Point_Data& d2, double& cost_from_angle_diff, double& cost_offset, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params);

void calculate_costs_for_all_possible_pairs(std::vector<Feature_Point_Data>* feature_vector_1,std::vector<Feature_Point_Data>* feature_vector_2, Cost_Rescale_Factors cost_rescale_factors);

void calculate_costs_for_all_feature_points(std::vector<Feature_Point_Data>* feature_vector_1,std::vector<Feature_Point_Data>* feature_vector_2, Cost_Rescale_Factors cost_rescale_factors);

double get_cost_for_single_problem_instance(std::vector<int>* instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type, Cost_Rescale_Factors cost_rescale_factors);
double get_cost_for_single_problem_instance_with_debug(std::vector<int>* instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type, Cost_Rescale_Factors cost_rescale_factors, bool active_debug_print);
double get_cost_for_single_problem_instance_using_cost_params(std::vector<int>* instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params);
double get_cost_for_single_problem_instance_custom_memory(Memory_Element instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type, Cost_Rescale_Factors cost_rescale_factors);

double get_cost_for_single_additional_assignment_custom_memory(Memory_Element instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type,int start_row, int new_assignment_row, Cost_Rescale_Factors cost_rescale_factors);

Cost_Breakdown get_cost_breakdown_for_single_problem_instance(std::vector<int>* instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, All_Cost_Parameters cost_params);

std::vector<int>* assign_candidates_to_features_using_branch_and_bound(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, BaB_Search_Modes search_mode, Cost_Rescale_Factors cost_rescale_factors, Memory_Manager_Fixed_Size* mem_manager = nullptr, double cost_bound_of_already_found_solution = 0.0, const std::vector<int>* already_found_solution = nullptr);

void put_new_instances_on_queue(std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queues, std::vector<std::vector<int>*>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row , uint16_t num_previous_assignments, Instance_Cost_Pair* best_icp, int* num_deleted_instances, Cost_Rescale_Factors cost_rescale_factors);

Instance_Cost_Pair get_best_new_instance_and_put_rest_on_queue(std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queue, std::vector<std::vector<int>*>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row , uint16_t num_previous_assignments, Cost_Rescale_Factors cost_rescale_factors);

Instance_Cost_Pair get_best_new_instance(std::vector<std::vector<int>*>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row , uint16_t num_previous_assignments, int* num_deleted_instances, int num_remaining_decisions, Cost_Rescale_Factors cost_rescale_factors);
Instance_Cost_Pair_Cus_Mem get_best_new_instance_custom_memory(std::vector<Memory_Element>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row , uint16_t num_previous_assignments, int* num_deleted_instances, int num_remaining_decisions, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors);

void list_subinstances_for_next_decision_and_push_to_queue(     std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queue,
                                                                std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates,
                                                                uint16_t decision_row , uint16_t num_previous_assignments, Instance_Cost_Pair* best_icp, 
                                                                std::vector<int>* instance_vector, int* num_created_instances, Cost_Rescale_Factors cost_rescale_factors);

double get_heuristic_unary_costs(Cost_Rescale_Factors cost_rescale_factors);

double get_heuristic_pairwise_costs(Cost_Rescale_Factors cost_rescale_factors);

double get_heuristic_unary_costs_using_cost_params(Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params);

double get_heuristic_pairwise_costs_using_cost_params(Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params);

double get_heuristic_cost_for_remaining_decisions(Instance_Cost_Pair& icp, int total_num_decision, Cost_Rescale_Factors cost_rescale_factors, int num_remaining_decisions = -1);
double get_heuristic_cost_for_remaining_decisions_using_cost_params(Instance_Cost_Pair& icp, int total_num_decision, Cost_Rescale_Factors cost_rescale_factors, int num_remaining_decisions, All_Cost_Parameters& cost_params);
double get_heuristic_cost_for_remaining_decisions_custom_memory(Instance_Cost_Pair_Cus_Mem& icp, int total_num_decision, Cost_Rescale_Factors cost_rescale_factors, int num_remaining_decisions = -1);

double get_optimal_cost_for_problem_size(int problem_size, Cost_Rescale_Factors cost_rescale_factors);

double get_optimal_cost_for_problem_size_using_cost_params(int problem_size, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params);

std::vector<int> find_initial_solution_by_rotating_and_greedy_distance_matching(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, int& best_angle, Cost_Rescale_Factors cost_rescale_factors);

std::vector<int> find_initial_solution_by_rotating_and_greedy_distance_matching_neighborhood_cone(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, int& best_angle, Cost_Rescale_Factors cost_rescale_factors);

std::vector<int> find_initial_solution_by_rotating_and_greedy_cost_matching(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors);

std::vector<int> find_initial_solution_by_rotating_and_greedy_cost_matching_in_angle_range(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, int start_angle, int angle_range, Cost_Rescale_Factors cost_rescale_factors);