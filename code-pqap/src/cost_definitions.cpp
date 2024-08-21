#include "cost_definitions.h"
#include <chrono>
#include <thread>

#define PRINT_BaB_DEBUG false

#define BaB_DEBUG_PRINT(expr) if(PRINT_BaB_DEBUG){expr}

typedef struct ID_and_Dist{
    int neighbor_id;
    float dist_to_neighbor;
}ID_and_Dist;

typedef struct Vector_IDs_and_Dist{
    int index_of_current_best_feature_point;
    int index_of_current_best_candidate;

    int index_in_local_features;
    int index_in_local_candidates;

    float distance;
    float cost;

}Vector_IDs_and_Dist;

typedef struct Polar_Coords_and_ID{
    float angle_radians;
    float radius;
    float radius_squared;
    float sin_angle;
    float cos_angle;
    float color_value;
    int id;
    Channel_Type channel;
}Polar_Coords_and_ID;

struct Vector_IDs_and_Dist_Compare{

    bool operator()(const Vector_IDs_and_Dist& lhs, const Vector_IDs_and_Dist& rhs){

        return lhs.distance < rhs.distance;
    }
};

double get_cost_of_single_assignment_in_instance(std::vector<int>* instance, int assignment_index, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors);

void check_and_sort_into_ids_and_dist_vector(std::vector<Vector_IDs_and_Dist>& vec_ids_and_dists, Vector_IDs_and_Dist& new_element, int max_num_elements);

float get_distance_between_two_points_polar(Polar_Coords_and_ID& point_1,Polar_Coords_and_ID& point_2);

void print_instance_queue(std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queue,int instance_size_rows, int instance_size_cols, bool sparse_print);


double get_change_of_heuristic_cost_after_decision(unsigned int total_num, unsigned int decision_row, int decision_var, unsigned int total_num_previous_assignments, Cost_Rescale_Factors cost_rescale_factors);


void cyclic_search_custom_memory(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair_Cus_Mem* best_icp, int* num_created_instances, int* num_deleted_instances, int start_row, int end_row,Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors);

void get_best_instance_of_single_decision_row(int total_rows, int total_cols, Instance_Cost_Pair_Cus_Mem& local_best_icp, int num_previous_assignments, int num_remaining_decisions, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors);

void get_best_instance_of_single_decision_row_with_iterative_cost(int total_rows, int total_cols, Instance_Cost_Pair_Cus_Mem& local_best_icp, int num_previous_assignments, int num_remaining_decisions, int start_row, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors);

void get_best_instance_of_single_decision_row_with_iterative_cost_in_col_range(int total_rows, int total_cols, Instance_Cost_Pair_Cus_Mem& local_best_icp, int num_previous_assignments, int num_remaining_decisions, int start_row, int& col_from_previous_assignment, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors);

void check_occupancy_and_cost_of_single_assignment(Instance_Cost_Pair_Cus_Mem& local_best_icp, Memory_Element& local_instance_vector,int total_rows, int start_row, int decision_row, int col, Instance_Cost_Pair_Cus_Mem new_icp, double heuristic_cost_change, double& current_best_cost, double& current_best_cost_from_assignments, double& current_best_heuristic_cost_for_remaining_decisions, int& best_col_assignment, int& new_total_num_assignements, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors);

double cost_from_distance_difference(double distance_difference, Cost_Rescale_Factors cost_rescale_factors){

    return distance_difference - cost_rescale_factors.dist_offset;
}


double cost_from_color_difference(double color_difference, Cost_Rescale_Factors cost_rescale_factors){

    return color_difference - cost_rescale_factors.color_offset;
}


double cost_from_distance_difference_using_cost_params(double distance_difference, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters cost_params){
    //std::cout << "using cost_from_distance_difference_using_cost_params" << std::endl;
    return distance_difference - cost_rescale_factors.dist_offset;
}


double cost_from_color_difference_using_cost_params(double color_difference, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters cost_params){
    //std::cout << "using cost_from_color_difference_using_cost_params" << std::endl;
    return color_difference - cost_rescale_factors.color_offset;
}

void fill_polar_coords_and_id_vec_from_feature_point_vec(std::vector<Polar_Coords_and_ID>& pc_vec, std::vector<Feature_Point_Data>& features, float angle_offset, std::vector<int>* index_vector = nullptr, std::vector<int>* validity_vector = nullptr);

void greedy_search_for_initial_solution(std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queues, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair* best_icp, int* num_created_instances, int* num_deleted_instances, Cost_Rescale_Factors cost_rescale_factors);
void greedy_search_for_initial_solution_custom_memory(std::priority_queue<Instance_Cost_Pair_Cus_Mem,std::vector<Instance_Cost_Pair_Cus_Mem>,Queue_Cost_Compare_Cus_Mem>* instance_queue, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair_Cus_Mem* best_icp, int* num_created_instances, int* num_deleted_instances, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors);

void cyclic_greedy_search_for_initial_solution(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair* best_icp, int* num_created_instances, int* num_deleted_instances, Cost_Rescale_Factors cost_rescale_factors, Memory_Manager_Fixed_Size* mem_manager = nullptr, double cost_bound_of_already_found_solution = 0.0,const std::vector<int>* already_found_solution = nullptr);

void greedy_search_for_initial_solution_in_row_range(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair* best_icp, int* num_created_instances, int* num_deleted_instances, int start_row, int end_row, Cost_Rescale_Factors cost_rescale_factors);

Instance_Cost_Pair_Cus_Mem get_best_new_instance_and_put_rest_on_queue_custom_memory(std::priority_queue<Instance_Cost_Pair_Cus_Mem,std::vector<Instance_Cost_Pair_Cus_Mem>,Queue_Cost_Compare_Cus_Mem>* instance_queues, std::vector<Memory_Element>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row , uint16_t num_previous_assignments, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors);


std::vector<std::vector<int>*>* create_neighborhood_lists_for_candidates(std::vector<Polar_Coords_and_ID>& local_candidates, int num_neighbors_per_candidate);

void check_all_transpositions_in_range(std::vector<int>& instance, int base_transposition_index, int range_start, int range_end, const double base_cost, double& current_best_cost, int& index_a_of_best_transposition, int& index_b_of_best_transposition,Cost_Rescale_Factors cost_rescale_factors, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates){

      for(int col = range_start; col < range_end; col++){
                    //std::cout << col << " ";
        if(col == base_transposition_index){
            continue;
        }

        //std::cout << "instance before transposition of : " << col << " and " << i << std::endl;
        //print_instance_vector_as_vector(&instance);

        double cost_contribution_of_base = get_cost_of_single_assignment_in_instance(&instance,base_transposition_index,feature_points,candidates,cost_rescale_factors);
        double cost_contribution_of_col = get_cost_of_single_assignment_in_instance(&instance,col,feature_points,candidates,cost_rescale_factors);
        double total_of_both = cost_contribution_of_base + cost_contribution_of_col;// - calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[base_transposition_index],(*candidates)[instance[base_transposition_index]],(*feature_points)[col],(*candidates)[instance[col]],cost_rescale_factors);

        if(instance[base_transposition_index] != -1 && instance[col] != -1){
            total_of_both -= calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[base_transposition_index],(*candidates)[instance[base_transposition_index]],(*feature_points)[col],(*candidates)[instance[col]],cost_rescale_factors);
        }

        int previous_assignment_of_base = instance[base_transposition_index]; 

        instance[base_transposition_index] = instance[col];
        instance[col] = previous_assignment_of_base;

        double cost_contribution_of_base_after_t = get_cost_of_single_assignment_in_instance(&instance,base_transposition_index,feature_points,candidates,cost_rescale_factors);
        double cost_contribution_of_col_after_t = get_cost_of_single_assignment_in_instance(&instance,col,feature_points,candidates,cost_rescale_factors); 
        double total_of_both_after_t = cost_contribution_of_base_after_t + cost_contribution_of_col_after_t;// - calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[base_transposition_index],(*candidates)[instance[base_transposition_index]],(*feature_points)[col],(*candidates)[instance[col]],cost_rescale_factors);

        if(instance[base_transposition_index] != -1 && instance[col] != -1){
            total_of_both_after_t -= calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[base_transposition_index],(*candidates)[instance[base_transposition_index]],(*feature_points)[col],(*candidates)[instance[col]],cost_rescale_factors);
        }
        //std::cout << "instance after transposition of : " << col << " and " << i << std::endl;
        //print_instance_vector_as_vector(&instance);

        double cost_after_transposition = get_cost_for_single_problem_instance(&instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);

        double cost_after_transpostion_using_diff = base_cost + (total_of_both_after_t - total_of_both);

        //std::cout << "transpostions of: " << col << " and: " << base_transposition_index << " mapped:" << instance[col] << " and: " << instance[base_transposition_index]  << "improves the cost to: " << cost_after_transposition << " from: " << current_best_cost << std::endl;
        //std::cout << "diff: " << cost_after_transposition - base_cost << " total_both: " << total_of_both << " total_both_after_T: " << total_of_both_after_t << " diff: " << total_of_both_after_t - total_of_both << std::endl; 
        //std::cout << cost_after_transposition << " " << cost_after_transpostion_using_diff  << std::endl; 

        
        if((fabs(cost_after_transposition - cost_after_transpostion_using_diff) > FLT_EPSILON) && (cost_after_transposition < 1.0)){
            std::cout << "conflict! : " << cost_after_transposition << " " << cost_after_transpostion_using_diff << std::endl;
            std::cout << base_transposition_index << " " << instance[base_transposition_index] << " " << col << " " <<  instance[col] << std::endl ;
        }     

        if(cost_after_transpostion_using_diff < current_best_cost){
            current_best_cost = cost_after_transpostion_using_diff;
            index_a_of_best_transposition = base_transposition_index;
            index_b_of_best_transposition = col;           
        }

        instance[col] = instance[base_transposition_index];
        instance[base_transposition_index] = previous_assignment_of_base;

        //std::cout << "instance after reset of : " << col << " and " << i << std::endl;
        //print_instance_vector_as_vector(&instance);
    }
}

void apply_greedy_assignment_transposition(std::vector<int>* best_instance, Cost_Rescale_Factors cost_rescale_factors, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates){

    //std::cout << "begin greedy transposition" << std::endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


    std::vector<int> instance = *(best_instance);


    //print_instance_vector_as_vector(&instance);

    int intervall_max_range = instance.size();// - 1; 

    int col_range_radius = std::max(COL_SEARCH_RANGE_IN_CYCLIC_BAB_SEARCH_MIN_SIZE,(int)std::round(instance.size() * COL_SEARCH_RANGE_IN_CYCLIC_BAB_SEARCH));

    int found_improvment = true;


    while(found_improvment){
        double base_cost = get_cost_for_single_problem_instance(&instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);    
        double best_cost_of_transposed_instance = base_cost;

        int best_transposition_index_a = -1;
        int best_transposition_index_b = -1;

        for(int i = 0; i < instance.size();i++){
            //std::cout << i << " " << instance[i] << std::endl;
            /*
            int instance_i_prev = instance[i];
            bool debug_print = false;
            if(i == 1){
                debug_print = true;
            }

            base_cost = get_cost_for_single_problem_instance_with_debug(&instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors,debug_print);

            double cost_contribution_of_base = get_cost_of_single_assignment_in_instance(&instance,i,feature_points,candidates,cost_rescale_factors);

            double cost_without_base = base_cost - cost_contribution_of_base;

            instance[i] = -1;

            double cost_after_unset = get_cost_for_single_problem_instance_with_debug(&instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors,debug_print);

            if((fabs(cost_without_base - cost_after_unset) > FLT_EPSILON) && (cost_after_unset < 1.0)){
                std::cout << "conflict! of unset: " << cost_without_base << " " << cost_after_unset << std::endl;
            }else{
                //std::cout << "NO conflict in Unset!" << std::endl;
            }

            instance[i] = instance_i_prev;
            
            
            */

            if(col_range_radius == -1){
                //implement checking entire assignment vector
            }else{
                int first_intervall_start = 0;
                int first_intervall_end = intervall_max_range;

                int second_intervall_start = 0;
                int second_intervall_end = 0;


                first_intervall_start = (i - col_range_radius + intervall_max_range) % intervall_max_range;
                first_intervall_end = (i + (col_range_radius+1) + intervall_max_range) % intervall_max_range; 


                
                if(first_intervall_start > i){
                    second_intervall_start = first_intervall_start;
                    second_intervall_end = intervall_max_range;

                    first_intervall_start = 0;
                    //first_intervall_end -= 1; // we subtract to 1 since we need to take the zero into account and this would make the range one element larger.
                    check_all_transpositions_in_range(instance,i,second_intervall_start,second_intervall_end,base_cost,best_cost_of_transposed_instance,best_transposition_index_a,best_transposition_index_b,cost_rescale_factors,feature_points,candidates);

                }else if(first_intervall_end < i){
                    second_intervall_start = 0;
                    second_intervall_end = first_intervall_end + 1;// - 1; // we subtract to 1 since we need to take the zero into account and this would make the range one element larger.

                    first_intervall_end = intervall_max_range;
                    check_all_transpositions_in_range(instance,i,second_intervall_start,second_intervall_end,base_cost,best_cost_of_transposed_instance,best_transposition_index_a,best_transposition_index_b,cost_rescale_factors,feature_points,candidates);
                }  

                /*
                if(first_intervall_end < i){
                    second_intervall_start = 0;
                    second_intervall_end = first_intervall_end;

                    first_intervall_end = intervall_max_range;

                    check_all_transpositions_in_range(instance,i,second_intervall_start,second_intervall_end,base_cost,best_cost_of_transposed_instance,best_transposition_index_a,best_transposition_index_b,cost_rescale_factors,feature_points,candidates);
                }
                */
                //std::cout << first_intervall_start << " " << first_intervall_end << " " << second_intervall_start << " " << second_intervall_end << std::endl;
                
                check_all_transpositions_in_range(instance,i,first_intervall_start,first_intervall_end,base_cost,best_cost_of_transposed_instance,best_transposition_index_a,best_transposition_index_b,cost_rescale_factors,feature_points,candidates);


            }
        }

        if(best_cost_of_transposed_instance < base_cost){
            found_improvment = true;

            //std::cout << "found improvement: " << best_cost_of_transposed_instance << " " << best_transposition_index_a << " " << instance[best_transposition_index_a] << " " << best_transposition_index_b << " " << instance[best_transposition_index_b] << std::endl;

            int previous_assignment_of_a = instance[best_transposition_index_a];
            instance[best_transposition_index_a] = instance[best_transposition_index_b];
            instance[best_transposition_index_b] = previous_assignment_of_a;


            (*(best_instance))[best_transposition_index_a] = instance[best_transposition_index_a];
            (*(best_instance))[best_transposition_index_b] = instance[best_transposition_index_b];


        }else{
            found_improvment = false;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "greedy transpostion took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]"  << std::endl;

}

Cost_Rescale_Factors get_cost_rescale_factors(int instance_size, All_Cost_Parameters& all_cost_params){

    int num_pairs = calculate_binomial_coefficient(instance_size,2);


    Cost_Rescale_Factors rescale_factors;

    rescale_factors.color_offset = all_cost_params.color_offset;
    rescale_factors.dist_offset = all_cost_params.dist_offset;
    rescale_factors.angle_offset = all_cost_params.angle_offset;
    rescale_factors.color_to_dist_weight = all_cost_params.color_to_dist_weight;
    rescale_factors.unary_to_to_quadr_weight = all_cost_params.unary_to_to_quadr_weight;

    rescale_factors.unary_prefactor = (1.0 - rescale_factors.unary_to_to_quadr_weight) / (double)instance_size;

    rescale_factors.quadr_prefactor = rescale_factors.unary_to_to_quadr_weight / (double)num_pairs;

    rescale_factors.heuristic_unary_cost = rescale_factors.unary_prefactor * (rescale_factors.color_to_dist_weight * -rescale_factors.color_offset + (1.0 - rescale_factors.color_to_dist_weight) * -rescale_factors.dist_offset);

    rescale_factors.heurisitc_quadr_cost = rescale_factors.quadr_prefactor * -rescale_factors.angle_offset;

    //std::cout << num_pairs << " " << rescale_factors.unary_cost_rescale << " " << rescale_factors.pairwise_cost_rescale << std::endl;
    //rescale_factors.pairwise_cost_rescale = 1.0;
    //rescale_factors.unary_cost_rescale = 1.0;

    //double theoretical_maximal_cost = get_optimal_cost_for_problem_size(instance_size);

    return rescale_factors;
}

double calculate_cost_for_two_feature_points_using_cost_params(Feature_Point_Data& feature_point_1,Feature_Point_Data& feature_point_2, Cost_Rescale_Factors cost_rescale_factors,All_Cost_Parameters& cost_params){

    //std::cout << "using calculate_cost_for_two_feature_points_using_cost_params" << std::endl;

    double total_cost = 0.0;

    if(feature_point_1.channel != feature_point_2.channel){
        return COST_MISMATCHING_CHANNELS;
    }

    double distance_difference = abs(feature_point_1.relative_distance_center_max_distance - feature_point_2.relative_distance_center_max_distance);

    distance_difference = cost_from_distance_difference_using_cost_params(distance_difference,cost_rescale_factors,cost_params);

    double color_difference = abs(feature_point_1.normalize_peak_value - feature_point_2.normalize_peak_value);

    color_difference = cost_from_color_difference_using_cost_params(color_difference,cost_rescale_factors,cost_params);

    total_cost = cost_rescale_factors.unary_prefactor * (cost_rescale_factors.color_to_dist_weight * color_difference + (1.0 - cost_rescale_factors.color_to_dist_weight) * distance_difference);

    //total_cost *= cost_rescale_factors.unary_cost_rescale;

    return total_cost;
}

double calculate_cost_for_two_feature_points(Feature_Point_Data& feature_point_1,Feature_Point_Data& feature_point_2, Cost_Rescale_Factors cost_rescale_factors){

    double total_cost = 0.0;

    if(feature_point_1.channel != feature_point_2.channel){
        return COST_MISMATCHING_CHANNELS;
    }

    double distance_difference = abs(feature_point_1.relative_distance_center_max_distance - feature_point_2.relative_distance_center_max_distance);

    distance_difference = cost_from_distance_difference(distance_difference,cost_rescale_factors);

    double color_difference = abs(feature_point_1.normalize_peak_value - feature_point_2.normalize_peak_value);

    color_difference = cost_from_color_difference(color_difference,cost_rescale_factors);

    total_cost = cost_rescale_factors.unary_prefactor * (cost_rescale_factors.color_to_dist_weight * color_difference + (1.0 - cost_rescale_factors.color_to_dist_weight) * distance_difference);

    //total_cost *= cost_rescale_factors.unary_cost_rescale;

    return total_cost;
}

void calculate_costs_for_all_feature_points(std::vector<Feature_Point_Data>* feature_vector_1,std::vector<Feature_Point_Data>* feature_vector_2, Cost_Rescale_Factors cost_rescale_factors){

    size_t shape[] = {feature_vector_1->size() , feature_vector_2->size()};

	andres::Marray<double> marray_unary_cost_matrix(shape,shape+2,0,andres::FirstMajorOrder);

    for(int i = 0; i < feature_vector_1->size(); i++){
        for(int j = 0; j < feature_vector_2->size(); j++){
            double cost_ij = calculate_cost_for_two_feature_points((*feature_vector_1)[i],(*feature_vector_2)[j],cost_rescale_factors);

            marray_unary_cost_matrix(i,j) = cost_ij;

            //std::cout << i  << "," << j << " " << cost_ij << " "; 
        }
        //std::cout << std::endl;   
    }

    write_cost_arrays_as_hdf5(marray_unary_cost_matrix,marray_unary_cost_matrix,"test_costs.h5");
}

void calculate_costs_for_all_possible_pairs(std::vector<Feature_Point_Data>* feature_vector_1,std::vector<Feature_Point_Data>* feature_vector_2, Cost_Rescale_Factors cost_rescale_factors){

    for(int i = 0; i < feature_vector_1->size();i++){
        Feature_Point_Data feature_point_1 = (*feature_vector_1)[i];

        for(int j = 0; j < feature_vector_2->size();j++){
            Feature_Point_Data candidate_point_1 = (*feature_vector_2)[j];

            for(int ii = 0; ii < feature_vector_1->size();ii++){
                Feature_Point_Data feature_point_2 = (*feature_vector_1)[ii];

                for(int jj = 0; jj < feature_vector_2->size();jj++){
                    Feature_Point_Data candidate_point_2 = (*feature_vector_2)[jj];
                
                    //std::cout << i << "," << j << " -> " << ii << "," << jj << " | ";

                    if(ii == i || jj == j){
                        std::cout << "D" << " | " ;
                    }else{
                        std::cout << calculate_cost_for_single_feature_point_pair_assignment(feature_point_1,candidate_point_1,feature_point_2,candidate_point_2,cost_rescale_factors) << " | ";
                    }

                
                }
            }

            std::cout << std::endl;
        }
    }

}

double calculate_cost_for_single_feature_point_pair_assignment(Feature_Point_Data& x1, Feature_Point_Data& d1, Feature_Point_Data& x2, Feature_Point_Data& d2, Cost_Rescale_Factors cost_rescale_factors){
    // we have the assignements x1 -> d1 and x2 -> d2,
    // we construct the pairs s.t. the angle of x1 is smaller then x2
    // to calculate the cost we take the angle difference x2 - x1


    if(x1.channel != d1.channel || x2.channel != d2.channel){
        return COST_MISMATCHING_CHANNELS;
    }

    double diff_x2_x1 = x2.angle - x1.angle;

    double optimal_d2_angle = d1.angle + diff_x2_x1;

    if(optimal_d2_angle > 360.0){
        optimal_d2_angle -= 360.0;
    }else if(optimal_d2_angle < 0.0){
        optimal_d2_angle += 360.0;
    }

    double angle_from_d2_to_optimal = 180.0 - abs(abs(optimal_d2_angle - d2.angle) - 180.0);

    // we normalize by 180.0 because 180.0 is the largest possible value angle_from_d2_to_optimal can possibly take
    // as this would mean the point is on the exact opposite site of the circle. 

    double normalized_angle_diff = fabs(angle_from_d2_to_optimal / 180.0); 

    double normalized_cost = cost_rescale_factors.quadr_prefactor * (normalized_angle_diff - cost_rescale_factors.angle_offset);

    //normalized_cost *= cost_rescale_factors.pairwise_cost_rescale;

    return normalized_cost;
}

double calculate_cost_for_single_feature_point_pair_assignment_using_cost_params(Feature_Point_Data& x1, Feature_Point_Data& d1, Feature_Point_Data& x2, Feature_Point_Data& d2, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params){
    // we have the assignements x1 -> d1 and x2 -> d2,
    // we construct the pairs s.t. the angle of x1 is smaller then x2
    // to calculate the cost we take the angle difference x2 - x1


    if(x1.channel != d1.channel || x2.channel != d2.channel){
        return COST_MISMATCHING_CHANNELS;
    }

    double diff_x2_x1 = x2.angle - x1.angle;

    double optimal_d2_angle = d1.angle + diff_x2_x1;

    if(optimal_d2_angle > 360.0){
        optimal_d2_angle -= 360.0;
    }

    double angle_from_d2_to_optimal = 180.0 - abs(abs(optimal_d2_angle - d2.angle) - 180.0);

    // we normalize by 180.0 because 180.0 is the largest possible value angle_from_d2_to_optimal can possibly take
    // as this would mean the point is on the exact opposite site of the circle. 

    double normalized_angle_diff = fabs(angle_from_d2_to_optimal / 180.0); 

    double normalized_cost = cost_rescale_factors.quadr_prefactor * (normalized_angle_diff - cost_rescale_factors.angle_offset);

    //normalized_cost *= cost_rescale_factors.pairwise_cost_rescale;

    return normalized_cost;
}

double get_cost_for_single_additional_assignment_custom_memory(Memory_Element instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type,int start_row, int new_assignment_row, Cost_Rescale_Factors cost_rescale_factors){
    int rows = feature_points->size();
    int cols = candidates->size();

    double linear_cost = 0.0f;
    double quadratic_cost = 0.0f;

    double total_cost = 0.0f;


    int candidate_index = MEM_ELEM_ACCESS(instance,int,new_assignment_row);//(*instance)[row];

    if(candidate_index != -1){

        linear_cost += calculate_cost_for_two_feature_points((*feature_points)[new_assignment_row],(*candidates)[candidate_index],cost_rescale_factors);

        int loop_count = 0;

        
        if(new_assignment_row < start_row){
            loop_count = rows - start_row + new_assignment_row;
        }else{
            loop_count = new_assignment_row - start_row;
        }
        
        //loop_count = new_assignment_row - start_row;
        //std::cout << "from " << start_row << " to " << new_assignment_row << " over " << loop_count << " elements";  

        for(int row_offset = 0; row_offset < loop_count; row_offset++){

            int previous_row = (start_row + row_offset) % rows;

            //std::cout << " " << previous_row;
            int previous_candidate_index = MEM_ELEM_ACCESS(instance,int,previous_row);//(*instance)[previous_row];

            if(previous_candidate_index != -1){
                quadratic_cost += calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[new_assignment_row],
                                                                                            (*candidates)[candidate_index],
                                                                                            (*feature_points)[previous_row],
                                                                                            (*candidates)[previous_candidate_index],
                                                                                            cost_rescale_factors);
            }
        }
        //std::cout << std::endl;
    }


    total_cost = linear_cost + quadratic_cost;

    if(cost_type == SINGLE_INSTANCE_TOTAL_COST){
        return total_cost;
    }

    if(cost_type == SINGLE_INSTANCE_LINEAR_COST){
        return linear_cost;
    }

    if(cost_type == SINGLE_INSTANCE_QUADRATIC_COST){
        return quadratic_cost;
    }

    return total_cost;
}

double get_cost_for_single_problem_instance_custom_memory(Memory_Element instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type, Cost_Rescale_Factors cost_rescale_factors){
    int rows = feature_points->size();
    int cols = candidates->size();

    double linear_cost = 0.0f;
    double quadratic_cost = 0.0f;

    double total_cost = 0.0f;

    for(int row = 0; row < rows; row++){
        int candidate_index = MEM_ELEM_ACCESS(instance,int,row);//(*instance)[row];

        if(candidate_index != -1){

            linear_cost += calculate_cost_for_two_feature_points((*feature_points)[row],(*candidates)[candidate_index],cost_rescale_factors);

            for(int previous_row = 0; previous_row < row; previous_row++){
                int previous_candidate_index = MEM_ELEM_ACCESS(instance,int,previous_row);//(*instance)[previous_row];

                if(previous_candidate_index != -1){
                    quadratic_cost += calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[row],
                                                                                            (*candidates)[candidate_index],
                                                                                            (*feature_points)[previous_row],
                                                                                            (*candidates)[previous_candidate_index],
                                                                                            cost_rescale_factors);
                }
            }
        }
    }

    total_cost = linear_cost + quadratic_cost;

    if(cost_type == SINGLE_INSTANCE_TOTAL_COST){
        return total_cost;
    }

    if(cost_type == SINGLE_INSTANCE_LINEAR_COST){
        return linear_cost;
    }

    if(cost_type == SINGLE_INSTANCE_QUADRATIC_COST){
        return quadratic_cost;
    }

    return total_cost;

}

double get_cost_for_single_problem_instance_using_cost_params(std::vector<int>* instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params){
    if(instance == nullptr){
        return 1.0;
    }
    
    int rows = feature_points->size();
    int cols = candidates->size();

    if(rows > instance->size()){
        std::cerr << "Number of feature points " << rows << " exceeded the length of the assignment instance " << instance->size() <<  " ! (get_cost_for_single_problem_instance)" << std::endl;
        return 1.0;
    }

    double linear_cost = 0.0f;
    double quadratic_cost = 0.0f;

    double total_cost = 0.0f;

    for(int row = 0; row < rows; row++){
        int candidate_index = (*instance)[row];

        if(candidate_index != -1){

            linear_cost += calculate_cost_for_two_feature_points_using_cost_params((*feature_points)[row],(*candidates)[candidate_index],cost_rescale_factors,cost_params);

            for(int previous_row = 0; previous_row < row; previous_row++){
                int previous_candidate_index = (*instance)[previous_row];

                if(previous_candidate_index != -1){
                    quadratic_cost += calculate_cost_for_single_feature_point_pair_assignment_using_cost_params(  (*feature_points)[row],
                                                                                            (*candidates)[candidate_index],
                                                                                            (*feature_points)[previous_row],
                                                                                            (*candidates)[previous_candidate_index],
                                                                                            cost_rescale_factors,cost_params);
                }
            }
        }
    }

    total_cost = linear_cost + quadratic_cost;

    if(cost_type == SINGLE_INSTANCE_TOTAL_COST){
        return total_cost;
    }

    if(cost_type == SINGLE_INSTANCE_LINEAR_COST){
        return linear_cost;
    }

    if(cost_type == SINGLE_INSTANCE_QUADRATIC_COST){
        return quadratic_cost;
    }

    return total_cost;

}

double get_cost_of_single_assignment_in_instance(std::vector<int>* instance, int assignment_index, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors){

    if((*instance)[assignment_index] == -1){
        return 0.0;
    }

    int candidate_index = (*instance)[assignment_index];
    double unary_cost = calculate_cost_for_two_feature_points((*feature_points)[assignment_index],(*candidates)[candidate_index],cost_rescale_factors);

    double quadratic_cost = 0.0;

    for(int row = 0; row < instance->size(); row++){

        if(row == assignment_index){
            continue;
        }

        int second_candidate_index = (*instance)[row];

        //std::cout << assignment_index << " " << candidate_index << " " << row << " " << second_candidate_index << std::endl;

        if(second_candidate_index != -1){
                quadratic_cost += calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[assignment_index],
                                                                                            (*candidates)[candidate_index],
                                                                                            (*feature_points)[row],
                                                                                            (*candidates)[second_candidate_index],
                                                                                            cost_rescale_factors);
    /*
                double quadratic_cost_1 = calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[assignment_index],
                                                                                            (*candidates)[candidate_index],
                                                                                            (*feature_points)[row],
                                                                                            (*candidates)[second_candidate_index],
                                                                                            cost_rescale_factors);

                double quadratic_cost_2 = calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[row],
                                                                                            (*candidates)[second_candidate_index],
                                                                                            (*feature_points)[assignment_index],
                                                                                            (*candidates)[candidate_index],
                                                                                            cost_rescale_factors);

            if((fabs(quadratic_cost_1 - quadratic_cost_2) > FLT_EPSILON) && (quadratic_cost_1 < 1.0)){
                std::cout << "conflict! in quadratic costs : " << quadratic_cost_1 << " " << quadratic_cost_2 << std::endl;
            }
    */

        }
            
    }
    
    return unary_cost + quadratic_cost;
}

double get_cost_for_single_problem_instance(std::vector<int>* instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type, Cost_Rescale_Factors cost_rescale_factors){
    if(instance == nullptr){
        return 1.0;
    }
    
    int rows = feature_points->size();
    int cols = candidates->size();

    if(rows > instance->size()){
        std::cerr << "Number of feature points " << rows << " exceeded the length of the assignment instance " << instance->size() <<  " ! (get_cost_for_single_problem_instance)" << std::endl;
        return 1.0;
    }

    double linear_cost = 0.0f;
    double quadratic_cost = 0.0f;

    double total_cost = 0.0f;

    for(int row = 0; row < rows; row++){
        int candidate_index = (*instance)[row];

        if(candidate_index != -1){

            //std::cout << "unary: " << row << " " << candidate_index << " " << calculate_cost_for_two_feature_points((*feature_points)[row],(*candidates)[candidate_index],cost_rescale_factors) << std::endl;

            linear_cost += calculate_cost_for_two_feature_points((*feature_points)[row],(*candidates)[candidate_index],cost_rescale_factors);

            for(int previous_row = 0; previous_row < row; previous_row++){
                int previous_candidate_index = (*instance)[previous_row];

                if(previous_candidate_index != -1){
                    quadratic_cost += calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[row],
                                                                                            (*candidates)[candidate_index],
                                                                                            (*feature_points)[previous_row],
                                                                                            (*candidates)[previous_candidate_index],
                                                                                            cost_rescale_factors);
                }
            }
        }
    }

    total_cost = linear_cost + quadratic_cost;

    if(cost_type == SINGLE_INSTANCE_TOTAL_COST){
        return total_cost;
    }

    if(cost_type == SINGLE_INSTANCE_LINEAR_COST){
        return linear_cost;
    }

    if(cost_type == SINGLE_INSTANCE_QUADRATIC_COST){
        return quadratic_cost;
    }

    return total_cost;

}

double get_cost_for_single_problem_instance_with_debug(std::vector<int>* instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Single_Instance_Cost_Type cost_type, Cost_Rescale_Factors cost_rescale_factors, bool active_debug_print){
    if(instance == nullptr){
        return 1.0;
    }
    
    int rows = feature_points->size();
    int cols = candidates->size();

    if(rows > instance->size()){
        std::cerr << "Number of feature points " << rows << " exceeded the length of the assignment instance " << instance->size() <<  " ! (get_cost_for_single_problem_instance)" << std::endl;
        return 1.0;
    }

    double linear_cost = 0.0f;
    double quadratic_cost = 0.0f;

    double total_cost = 0.0f;

    int num_linear_terms = 0;
    int num_quadr_terms = 0;

    for(int row = 0; row < rows; row++){
        int candidate_index = (*instance)[row];

        if(candidate_index != -1){
            /*
            if(active_debug_print){
                std::cout << "unary: " << row << " " << candidate_index << " " << calculate_cost_for_two_feature_points((*feature_points)[row],(*candidates)[candidate_index],cost_rescale_factors) << std::endl;
            }
            */

            linear_cost += calculate_cost_for_two_feature_points((*feature_points)[row],(*candidates)[candidate_index],cost_rescale_factors);

            num_linear_terms++;

            for(int previous_row = 0; previous_row < row; previous_row++){
                int previous_candidate_index = (*instance)[previous_row];

                if(previous_candidate_index != -1){
                    quadratic_cost += calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[row],
                                                                                            (*candidates)[candidate_index],
                                                                                            (*feature_points)[previous_row],
                                                                                            (*candidates)[previous_candidate_index],
                                                                                            cost_rescale_factors);
                    /*
                    if(active_debug_print){
                        std::cout << "quadr: " << row << " " << candidate_index << " " << previous_row << " " << previous_candidate_index << " " << calculate_cost_for_single_feature_point_pair_assignment(  (*feature_points)[row],(*candidates)[candidate_index],(*feature_points)[previous_row],(*candidates)[previous_candidate_index],cost_rescale_factors) << std::endl;
                    }
                    */
                   num_quadr_terms++;
                }
            }
        }
    }

    if(active_debug_print){
        std::cout << "linear: " << num_linear_terms << " quadr: " << num_quadr_terms << std::endl;
    }

    total_cost = linear_cost + quadratic_cost;

    if(cost_type == SINGLE_INSTANCE_TOTAL_COST){
        return total_cost;
    }

    if(cost_type == SINGLE_INSTANCE_LINEAR_COST){
        return linear_cost;
    }

    if(cost_type == SINGLE_INSTANCE_QUADRATIC_COST){
        return quadratic_cost;
    }

    return total_cost;

}

void get_cost_breakdown_for_two_feature_points(Feature_Point_Data& feature_point_1,Feature_Point_Data& feature_point_2, double& cost_from_color_diff, double& cost_from_dist_diff, double& cost_offset, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params){
    double distance_difference = abs(feature_point_1.relative_distance_center_max_distance - feature_point_2.relative_distance_center_max_distance);

    cost_from_dist_diff = cost_rescale_factors.unary_prefactor * cost_from_distance_difference_using_cost_params(distance_difference,cost_rescale_factors,cost_params);

    double color_difference = abs(feature_point_1.normalize_peak_value - feature_point_2.normalize_peak_value);

    cost_from_color_diff = cost_rescale_factors.unary_prefactor * cost_from_color_difference_using_cost_params(color_difference,cost_rescale_factors,cost_params);

    cost_offset = cost_rescale_factors.unary_prefactor * (cost_rescale_factors.color_to_dist_weight * -cost_rescale_factors.color_offset + (1.0 - cost_rescale_factors.color_to_dist_weight) * -cost_rescale_factors.dist_offset);
}

void get_cost_breakdown_for_single_feature_point_pair_assignment(Feature_Point_Data& x1, Feature_Point_Data& d1, Feature_Point_Data& x2, Feature_Point_Data& d2, double& cost_from_angle_diff, double& pairwise_cost_offset, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params){
    if(x1.channel != d1.channel || x2.channel != d2.channel){
        return; //COST_MISMATCHING_CHANNELS;
    }

    double diff_x2_x1 = x2.angle - x1.angle;

    double optimal_d2_angle = d1.angle + diff_x2_x1;

    if(optimal_d2_angle > 360.0){
        optimal_d2_angle -= 360.0;
    }

    double angle_from_d2_to_optimal = 180.0 - abs(abs(optimal_d2_angle - d2.angle) - 180.0);

    // we normalize by 180.0 because 180.0 is the largest possible value angle_from_d2_to_optimal can possibly take
    // as this would mean the point is on the exact opposite site of the circle. 

    double normalized_angle = fabs(angle_from_d2_to_optimal / 180.0); 

    double normalized_cost = cost_rescale_factors.quadr_prefactor * (normalized_angle - cost_rescale_factors.angle_offset);

    cost_from_angle_diff = normalized_cost;

    pairwise_cost_offset = cost_rescale_factors.quadr_prefactor * -cost_rescale_factors.angle_offset;
    return;
}

Cost_Breakdown get_cost_breakdown_for_single_problem_instance(std::vector<int>* instance, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, All_Cost_Parameters cost_params){
    Cost_Breakdown cost_breakdown{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

    print_cost_parameters(cost_params);

    All_Cost_Parameters unit_cost_params{1.0,1.0,1.0,1.0,1.0,1.0,1.0};

    Cost_Rescale_Factors cost_rescale_factors = get_cost_rescale_factors(instance->size(), cost_params);

    cost_breakdown.theoretical_optimal_cost = get_optimal_cost_for_problem_size_using_cost_params(instance->size(),cost_rescale_factors,cost_params);
    cost_breakdown.total_cost = get_cost_for_single_problem_instance_using_cost_params(instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors,cost_params);

    int num_not_assigned_features = 0;

    for(int row = 0; row < instance->size(); row++){
        int candidate_index = (*instance)[row];

        if(candidate_index != -1){

            double current_cost_from_color_diff = 0.0;
            double current_cost_from_dist_diff = 0.0;
            double current_cost_offset = 0.0;

            get_cost_breakdown_for_two_feature_points((*feature_points)[row],(*candidates)[candidate_index],current_cost_from_color_diff,current_cost_from_dist_diff,current_cost_offset,cost_rescale_factors, cost_params);

            cost_breakdown.cost_from_color_differences += current_cost_from_color_diff;
            cost_breakdown.cost_from_distance_differences += current_cost_from_dist_diff;
            cost_breakdown.unary_cost_offsets += current_cost_offset;

            for(int previous_row = 0; previous_row < row; previous_row++){
                int previous_candidate_index = (*instance)[previous_row];

                double current_cost_from_angle_diff = 0.0;
                double current_pairwise_cost_offset = 0.0;

                if(previous_candidate_index != -1){
                    get_cost_breakdown_for_single_feature_point_pair_assignment(  (*feature_points)[row],
                                                                                            (*candidates)[candidate_index],
                                                                                            (*feature_points)[previous_row],
                                                                                            (*candidates)[previous_candidate_index],current_cost_from_angle_diff, current_pairwise_cost_offset,cost_rescale_factors, cost_params);
                }

                cost_breakdown.cost_from_angle_differences += current_cost_from_angle_diff;
                cost_breakdown.pairwise_cost_offsets += current_pairwise_cost_offset;

            }
        }else{
            num_not_assigned_features++;
        }
    }

    double optimal_cost_for_assignment = cost_breakdown.unary_cost_offsets + cost_breakdown.pairwise_cost_offsets;

    double color_cost_relative_to_unary = cost_breakdown.cost_from_color_differences / cost_breakdown.unary_cost_offsets;
    double dist_cost_relative_to_unary = cost_breakdown.cost_from_distance_differences / cost_breakdown.unary_cost_offsets;
    double angle_cost_relative_to_pairwise = cost_breakdown.cost_from_angle_differences / cost_breakdown.pairwise_cost_offsets;

    cost_breakdown.percentage_not_assigned_features = (float)num_not_assigned_features / (float)feature_points->size();

    std::cout << "Cost Breakdown: " << std::endl;
    std::cout << "Normalized Instance Cost: " << cost_breakdown.total_cost / cost_breakdown.theoretical_optimal_cost << std::endl;
    std::cout << "Instance Cost: " << cost_breakdown.total_cost << std::endl;
    std::cout << "Instance Sizes: " << instance->size() << " " << feature_points->size() << " " << candidates->size() << std::endl;
    std::cout << "Number not assigned Features: " << num_not_assigned_features << " as % of total: " <<cost_breakdown.percentage_not_assigned_features << std::endl; 
    std::cout << "Theoretical Optimum of Matching: " << cost_breakdown.theoretical_optimal_cost << std::endl;
    std::cout << "Theoretical Optimum of Assignment: " << optimal_cost_for_assignment << std::endl;
    std::cout << "Cost from Color Differences: " << cost_breakdown.cost_from_color_differences << std::endl;
    std::cout << "Cost from Distance Differences: " << cost_breakdown.cost_from_distance_differences << std::endl;
    std::cout << "Cost from Angle Differences: " << cost_breakdown.cost_from_angle_differences << std::endl;
    std::cout << "Total Unary Cost Offsets: " << cost_breakdown.unary_cost_offsets << std::endl;
    std::cout << "Total Pairwise Cost Offsets: " << cost_breakdown.pairwise_cost_offsets << std::endl;
    std::cout << "Cost from Color Differences relative to Unary Cost: " << color_cost_relative_to_unary << std::endl;
    std::cout << "Cost from Distance Differences relative to Unary Cost: " << dist_cost_relative_to_unary << std::endl;
    std::cout << "Cost from Angle Differences relative to Pairwise Cost: " << angle_cost_relative_to_pairwise << std::endl;
    std::cout << std::endl;

    return cost_breakdown;
}

std::vector<int>* assign_candidates_to_features_using_branch_and_bound(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, BaB_Search_Modes search_mode, Cost_Rescale_Factors cost_rescale_factors, Memory_Manager_Fixed_Size* mem_manager, double cost_bound_of_already_found_solution,const std::vector<int>* already_found_solution){

    std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare> instance_queue;

    std::vector<std::vector<int>*> subinstances;

    int num_feature_points = feature_points->size();
    int num_candidates = candidates->size();

    int max_num_points = std::max<int>(num_feature_points,num_candidates);

    //std::cout << num_feature_points << " " << num_candidates << " " << max_num_points << std::endl;

    double theoretical_optimal_cost = get_optimal_cost_for_problem_size(max_num_points,cost_rescale_factors);

    //std::cout << "Theoretical optimum: " << theoretical_optimal_cost << std::endl;

    int num_created_instances = 0;
    int num_deleted_instances = 0;
    int accumulator = 0;

    std::vector<int>* initial_instance = new std::vector<int>;//(num_feature_points);
    num_created_instances++;

    initial_instance->reserve(num_feature_points);

    for(int i = 0; i < num_feature_points;i++){
        initial_instance->push_back(-1);
    }

    Instance_Cost_Pair best_icp;
    best_icp.instance = initial_instance;
    best_icp.total_cost = 0.0;


    if(search_mode == BaB_BASIC_SEARCH){

        list_all_subinstances_recursively(initial_instance,0,0,num_feature_points,num_candidates,subinstances,&num_created_instances);

        put_new_instances_on_queue(&instance_queue,&subinstances,feature_points,candidates,0,0,&best_icp,&num_deleted_instances,cost_rescale_factors);
        
    }

    BaB_DEBUG_PRINT(
        print_instance_queue(&instance_queue,num_feature_points,num_candidates,false);
        print_multiple_instance_vectors_as_matrices(subinstances,num_feature_points,num_candidates);
    )
    //subinstances.clear();

    bool max_iterations_reached = false;
    int iterations_count = 0;


    if(search_mode == BaB_GREEDY_SEARCH_FIRST){
        greedy_search_for_initial_solution(&instance_queue,feature_points,candidates,&best_icp,&num_created_instances,&num_deleted_instances,cost_rescale_factors);

        //std::cout << "greedy initial solution, with cost: " << best_icp.total_cost << std::endl;
        //print_instance_vector_as_matrix(best_icp.instance,num_feature_points,num_candidates);

        //print_instance_queue(&instance_queue,num_feature_points,num_candidates,true);
    }

    if (search_mode == BaB_GREEDY_SEARCH_ONLY) {

        //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        if(!USE_CUSTOM_MEMORY_HANDLER){
            greedy_search_for_initial_solution(&instance_queue, feature_points, candidates, &best_icp, &num_created_instances, &num_deleted_instances,cost_rescale_factors);

            //std::cout << "greedy initial solution, with cost: " << best_icp.total_cost << std::endl;
            //print_instance_vector_as_matrix(best_icp.instance, num_feature_points, num_candidates);
            while (!instance_queue.empty()) {
                Instance_Cost_Pair current_icp = instance_queue.top();
                instance_queue.pop();

                delete(current_icp.instance);
                num_deleted_instances++;
            }

            //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            //std::cout << "greedy BaB = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

        }else{

            //begin = std::chrono::steady_clock::now();
            Memory_Element initial_instance_mem_elem = allocate_new_memory_element(mem_manager);
            std::priority_queue<Instance_Cost_Pair_Cus_Mem,std::vector<Instance_Cost_Pair_Cus_Mem>,Queue_Cost_Compare_Cus_Mem> instance_queue_for_custom_memory;

            for(int i = 0; i < num_feature_points;i++){
                MEM_ELEM_WRITE(initial_instance_mem_elem,int,i,-1);
            }

            int num_created_mem_elems = 0;
            int num_deleted_mem_elems = 0;

            Instance_Cost_Pair_Cus_Mem icp_cm;
            icp_cm.total_cost = 0.0;
            icp_cm.instance_mem_elem = initial_instance_mem_elem;

            greedy_search_for_initial_solution_custom_memory(&instance_queue_for_custom_memory,feature_points,candidates,&icp_cm,&num_created_mem_elems,&num_deleted_mem_elems,mem_manager,cost_rescale_factors);

            //end = std::chrono::steady_clock::now();
            //std::cout << "greedy BaB custom memory = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
            for(int i = 0; i < best_icp.instance->size();i++){
                (*(best_icp.instance))[i] = MEM_ELEM_ACCESS(icp_cm.instance_mem_elem,int,i);
                best_icp.total_cost = icp_cm.total_cost;
            }

        }




        //print_instance_queue(&instance_queue,num_feature_points,num_candidates,true);


        //std::cout << icp_cm.total_cost << " " << best_icp.total_cost << std::endl;
        //std::cout << num_created_instances << " " << num_created_mem_elems << std::endl;

        //print_instance_array_as_vector((int*)icp_cm.instance_mem_elem.memory_pointer,num_feature_points);
        //print_instance_vector_as_vector(best_icp.instance);
        max_iterations_reached = true;

    }

    if(search_mode == BaB_CYCLIC_GREEDY_SEARCH_ONLY){
        cyclic_greedy_search_for_initial_solution(feature_points, candidates, &best_icp, &num_created_instances, &num_deleted_instances,cost_rescale_factors,mem_manager,cost_bound_of_already_found_solution,already_found_solution);

        max_iterations_reached = true;
    }

    //std::cout << "Num created instances after initialization: " << num_created_instances << std::endl;

    while (!instance_queue.empty() && !max_iterations_reached){

        Instance_Cost_Pair current_icp = instance_queue.top();
        instance_queue.pop();
        subinstances.clear();

        BaB_DEBUG_PRINT(
            std::cout << "Instance chosen to be expanded: " << current_icp.next_row_to_make_decision << " " << current_icp.total_cost << " " << current_icp.heuristic_cost_for_remaining_decision << std::endl;
            print_instance_vector_as_matrix(current_icp.instance,num_feature_points,num_candidates);
        )

        if(current_icp.next_row_to_make_decision >= num_feature_points){
            //we have made one decision for each feature point and now have to evaluate if we have found a better solution

            if(best_icp.total_cost > current_icp.total_cost){
                std::cout << "new best: " << current_icp.total_cost << std::endl;

                // NOTE: since we only use this to check if we delete all instances that we have created. In the final version this if statement can be omitted.
                if(best_icp.instance != nullptr){

                    num_deleted_instances++;
                }

                delete(best_icp.instance);

                best_icp = current_icp;
            }else{
                BaB_DEBUG_PRINT(
                    std::cout << "cost of instance exceeded the total cost of the best solution" << std::endl;
                )
                delete(current_icp.instance);
                num_deleted_instances++;
            }



        }else{

            if(current_icp.total_cost < best_icp.total_cost){
                list_subinstances_for_next_decision_and_push_to_queue(&instance_queue,feature_points,candidates,current_icp.next_row_to_make_decision,current_icp.total_num_assignments,&best_icp,current_icp.instance,&num_created_instances,cost_rescale_factors);

                //list_all_subinstances_recursively(current_icp.instance,current_icp.next_row_to_make_decision,current_icp.next_row_to_make_decision,num_feature_points,num_candidates,subinstances,num_created_instances);
                //put_new_instances_on_queue(&instance_queue,&subinstances,feature_points,candidates,current_icp.next_row_to_make_decision,current_icp.total_num_assignments,&best_icp,num_deleted_instances);
            }else{
                BaB_DEBUG_PRINT(
                    std::cout << "cost of instance exceeded the total cost of the best solution" << std::endl;
                )
            }


            //print_instance_queue(&instance_queue,num_feature_points,num_candidates);
            //print_multiple_instance_vectors_as_matrices(subinstances,num_feature_points,num_candidates);

            delete(current_icp.instance);
            num_deleted_instances++;
        }

        iterations_count++;
        if(iterations_count > 10000000){
            std::cout << "reached max iterations" << std::endl;
            max_iterations_reached = true;
        }
        
    }


    if(best_icp.instance != nullptr){
        //std::cout << "Best Instance with Cost: " << best_icp.total_cost << std::endl;
        //std::cout << "Best Instance Cost percentage from theoretical optimum: " << (best_icp.total_cost/theoretical_optimal_cost) * 100.0 << "%" << std::endl;
        //std::cout << "Cost of Linear terms normalized by canidadates: " << get_cost_for_single_problem_instance(best_icp.instance,feature_points,candidates,SINGLE_INSTANCE_LINEAR_COST)/(double)candidates->size() << std::endl;
        //std::cout << "Cost of Linear terms normalized by features: " << get_cost_for_single_problem_instance(best_icp.instance,feature_points,candidates,SINGLE_INSTANCE_LINEAR_COST)/(double)feature_points->size() << std::endl;
        //print_instance_vector_as_matrix(best_icp.instance,num_feature_points,num_candidates);
        //delete(best_icp.instance);
        //num_deleted_instances++;
    }else{
        std::cout << "best solution was null" << std::endl;
    }

    /*
    for(int i = 0; i < subinstances.size();i++){
        delete(subinstances[i]);
    }
    */

    //if(search_mode == BaB_CYCLIC_GREEDY_SEARCH_ONLY){
        //std::cout << "Num created instances: " << num_created_instances << "   Num deleted instances: " << num_deleted_instances << " num features: " << num_feature_points << std::endl;
    //}


   return best_icp.instance;


}

void put_new_instances_on_queue(std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queues, std::vector<std::vector<int>*>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row, uint16_t num_previous_assignments, Instance_Cost_Pair* best_icp, int* num_deleted_instances, Cost_Rescale_Factors cost_rescale_factors){

    for(int i = 0; i < new_instances->size();i++){
        std::vector<int>* current_instance = (*new_instances)[i];

        BaB_DEBUG_PRINT(
            print_instance_vector_as_matrix(current_instance,feature_points->size(),candidates->size());
        )
        
        Instance_Cost_Pair new_icp;

        new_icp.instance = current_instance;
        new_icp.next_row_to_make_decision = decision_row + 1;
        new_icp.total_num_assignments = num_previous_assignments;

        if((*current_instance)[decision_row] != -1){
            new_icp.total_num_assignments++;
        }

        new_icp.cost = get_cost_for_single_problem_instance(current_instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
        new_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions(new_icp,feature_points->size(),cost_rescale_factors);
        new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

        if(new_icp.total_cost < best_icp->total_cost){

            instance_queues->push(new_icp);
        }else{
            // the total cost of the new icp already is larger then the cost of the currently best_icp (in which we have already taken all decisions for every feature point)
            // and since the cost of the new_icp is a heuristic estimation that can only get larger, we don't need to push the new_icp on the queue.
            delete(new_icp.instance);
            (*num_deleted_instances)++;
        }

    }
}

Instance_Cost_Pair get_best_new_instance(std::vector<std::vector<int>*>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row , uint16_t num_previous_assignments, int* num_deleted_instances, int num_remaining_decisions, Cost_Rescale_Factors cost_rescale_factors){

    std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare> tmp_instance_queue;

    Instance_Cost_Pair best_icp_from_new_subinstances;

    for(int i = 0; i < new_instances->size();i++){
        std::vector<int>* current_instance = (*new_instances)[i];

        BaB_DEBUG_PRINT(
            print_instance_vector_as_matrix(current_instance,feature_points->size(),candidates->size());
        )
        
        Instance_Cost_Pair new_icp;

        new_icp.instance = current_instance;
        new_icp.next_row_to_make_decision = decision_row + 1;
        new_icp.total_num_assignments = num_previous_assignments;

        if((*current_instance)[decision_row] != -1){
            new_icp.total_num_assignments++;
        }

        new_icp.cost = get_cost_for_single_problem_instance(current_instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
        new_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions(new_icp,feature_points->size(),cost_rescale_factors,num_remaining_decisions);
        new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

        tmp_instance_queue.push(new_icp);
    }

    best_icp_from_new_subinstances = tmp_instance_queue.top();
    tmp_instance_queue.pop();

    while(!tmp_instance_queue.empty()){
        Instance_Cost_Pair current_icp = tmp_instance_queue.top();
        delete(current_icp.instance);
        (*num_deleted_instances)++;
        
        tmp_instance_queue.pop();
    }

    return best_icp_from_new_subinstances;

}

void check_occupancy_and_cost_of_single_assignment(Instance_Cost_Pair_Cus_Mem& local_best_icp, Memory_Element& local_instance_vector,int total_rows, int start_row, int decision_row, int col, Instance_Cost_Pair_Cus_Mem new_icp, double heuristic_cost_change, double& current_best_cost, double& current_best_cost_from_assignments, double& current_best_heuristic_cost_for_remaining_decisions, int& best_col_assignment, int& new_total_num_assignements, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors){

    if(!check_if_instance_array_column_is_occupied((int*)local_instance_vector.memory_pointer,total_rows,col)){
            MEM_ELEM_WRITE(local_instance_vector,int,decision_row,col);

            new_icp.cost = local_best_icp.cost + get_cost_for_single_additional_assignment_custom_memory(local_instance_vector,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,start_row,decision_row,cost_rescale_factors);
            new_icp.heuristic_cost_for_remaining_decision = local_best_icp.heuristic_cost_for_remaining_decision - heuristic_cost_change;
            //new_icp.heuristic_cost_for_remaining_decision = heuristic_cost;
            //std::cout << new_icp.heuristic_cost_for_remaining_decision << std::endl;
            new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

            if(new_icp.total_cost < current_best_cost){
                current_best_cost = new_icp.total_cost;
                current_best_cost_from_assignments = new_icp.cost;
                current_best_heuristic_cost_for_remaining_decisions = new_icp.heuristic_cost_for_remaining_decision;
                best_col_assignment = col;
                new_total_num_assignements = new_icp.total_num_assignments;
            }

        }

}

void get_best_instance_of_single_decision_row_with_iterative_cost_in_col_range(int total_rows, int total_cols, Instance_Cost_Pair_Cus_Mem& local_best_icp, int num_previous_assignments, int num_remaining_decisions, int start_row, int& col_from_previous_assignment, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors){

    int original_value = ((int*)(local_best_icp.instance_mem_elem.memory_pointer))[local_best_icp.next_row_to_make_decision];

    Memory_Element local_instance_vector = local_best_icp.instance_mem_elem;

    int decision_row = local_best_icp.next_row_to_make_decision;
    int new_total_num_assignements = num_previous_assignments;
    double current_best_cost = 0.0;
    double current_best_cost_from_assignments = 0.0;
    double current_best_heuristic_cost_for_remaining_decisions = 0.0;
    int best_col_assignment = -1;

    Instance_Cost_Pair_Cus_Mem new_icp;
    new_icp.instance_mem_elem = local_instance_vector;

    new_icp.next_row_to_make_decision = 0;
    new_icp.total_num_assignments = 0;

    //std::cout << get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),feature_points->size()) << std::endl;

    // the check and the write in the next two lines might not be needed.
    if(!check_if_instance_array_column_is_occupied((int*)local_instance_vector.memory_pointer,total_rows,-1)){
        MEM_ELEM_WRITE(local_instance_vector,int,decision_row,-1);

        new_icp.next_row_to_make_decision = decision_row + 1;
        new_icp.total_num_assignments = num_previous_assignments;

        new_icp.cost = local_best_icp.cost + get_cost_for_single_additional_assignment_custom_memory(local_instance_vector,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,start_row,decision_row,cost_rescale_factors);
        //std::cout << "0 decision: ";
        new_icp.heuristic_cost_for_remaining_decision = local_best_icp.heuristic_cost_for_remaining_decision - get_change_of_heuristic_cost_after_decision(feature_points->size(),feature_points->size() - num_remaining_decisions,-1,new_icp.total_num_assignments,cost_rescale_factors);
        //new_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),num_remaining_decisions);
        //std::cout << "0 decision: " << new_icp.heuristic_cost_for_remaining_decision << " " << get_change_of_heuristic_cost_after_decision(feature_points->size(),feature_points->size() - num_remaining_decisions,-1,new_icp.total_num_assignments) << std::endl;
        new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

        if(new_icp.total_cost < current_best_cost){
            current_best_cost = new_icp.total_cost;
            current_best_cost_from_assignments = new_icp.cost;
            current_best_heuristic_cost_for_remaining_decisions = new_icp.heuristic_cost_for_remaining_decision;
            best_col_assignment = -1;
            new_total_num_assignements = new_icp.total_num_assignments;
        }
    }

    int intervall_max_range = total_cols;// - 1; 

    int col_range_radius = std::max(COL_SEARCH_RANGE_IN_CYCLIC_BAB_SEARCH_MIN_SIZE,(int)std::round(intervall_max_range * COL_SEARCH_RANGE_IN_CYCLIC_BAB_SEARCH));;

    double heuristic_cost_change = get_change_of_heuristic_cost_after_decision(feature_points->size(),feature_points->size() - num_remaining_decisions,1,new_icp.total_num_assignments,cost_rescale_factors);

    if(col_range_radius == -1){
        for(int col = 0; col < total_cols; col++){
            check_occupancy_and_cost_of_single_assignment(local_best_icp,local_instance_vector,total_rows,start_row,decision_row,col,new_icp,heuristic_cost_change,current_best_cost,current_best_cost_from_assignments,current_best_heuristic_cost_for_remaining_decisions,best_col_assignment,new_total_num_assignements,feature_points,candidates,cost_rescale_factors);
        }
    }else{
        int first_intervall_start = 0;
        int first_intervall_end = intervall_max_range;

        int second_intervall_start = 0;
        int second_intervall_end = 0;

        if(num_previous_assignments){
            first_intervall_start = (col_from_previous_assignment - col_range_radius + intervall_max_range) % intervall_max_range;
            first_intervall_end = (col_from_previous_assignment + (col_range_radius+1) + intervall_max_range) % intervall_max_range;

            if(first_intervall_start > col_from_previous_assignment){
                second_intervall_start = first_intervall_start;
                second_intervall_end = intervall_max_range;

                first_intervall_start = 0;
                //first_intervall_end -= 1; // we subtract to 1 since we need to take the zero into account and this would make the range one element larger.

            }else if(first_intervall_end < col_from_previous_assignment){
                second_intervall_start = 0;
                second_intervall_end = first_intervall_end;// - 1; // we subtract to 1 since we need to take the zero into account and this would make the range one element larger.

                first_intervall_end = intervall_max_range;

            }
        }

        //std::cout << "col_from_prev: " << col_from_previous_assignment << " range: " << col_range_radius << " 1st_start: " << first_intervall_start << " 1st_end: " << first_intervall_end << " 2nd_start: " << second_intervall_start << " 2nd_end: " << second_intervall_end << std::endl;

        new_icp.next_row_to_make_decision = decision_row + 1;
        new_icp.total_num_assignments = num_previous_assignments + 1;
        //new_icp.instance_mem_elem = local_instance_vector;

        //std::cout << "1 decision: "; 
        //double heuristic_cost = get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),num_remaining_decisions);

        //std::cout << "1 decision: " << new_icp.heuristic_cost_for_remaining_decision << " " << heuristic_cost_change << std::endl;

        for(int col = first_intervall_start; col < first_intervall_end; col++){
            check_occupancy_and_cost_of_single_assignment(local_best_icp,local_instance_vector,total_rows,start_row,decision_row,col,new_icp,heuristic_cost_change,current_best_cost,current_best_cost_from_assignments,current_best_heuristic_cost_for_remaining_decisions,best_col_assignment,new_total_num_assignements,feature_points,candidates,cost_rescale_factors);
        }

        for(int col = second_intervall_start; col < second_intervall_end; col++){
            check_occupancy_and_cost_of_single_assignment(local_best_icp,local_instance_vector,total_rows,start_row,decision_row,col,new_icp,heuristic_cost_change,current_best_cost,current_best_cost_from_assignments,current_best_heuristic_cost_for_remaining_decisions,best_col_assignment,new_total_num_assignements,feature_points,candidates,cost_rescale_factors);
        }

    }

             

    col_from_previous_assignment = best_col_assignment;

    original_value = best_col_assignment;

    MEM_ELEM_WRITE(local_instance_vector,int,decision_row,original_value);
    local_best_icp.total_cost = current_best_cost;
    local_best_icp.cost = current_best_cost_from_assignments;
    local_best_icp.heuristic_cost_for_remaining_decision = current_best_heuristic_cost_for_remaining_decisions;
    local_best_icp.total_num_assignments = new_total_num_assignements;
    local_best_icp.next_row_to_make_decision++;

    //print_instance_array_as_vector((int*)local_instance_vector.memory_pointer,total_rows);
}

void get_best_instance_of_single_decision_row_with_iterative_cost(int total_rows, int total_cols, Instance_Cost_Pair_Cus_Mem& local_best_icp, int num_previous_assignments, int num_remaining_decisions, int start_row, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors){

    int original_value = ((int*)(local_best_icp.instance_mem_elem.memory_pointer))[local_best_icp.next_row_to_make_decision];

    Memory_Element local_instance_vector = local_best_icp.instance_mem_elem;

    int decision_row = local_best_icp.next_row_to_make_decision;
    int new_total_num_assignements = num_previous_assignments;
    double current_best_cost = 0.0;
    double current_best_cost_from_assignments = 0.0;
    double current_best_heuristic_cost_for_remaining_decisions = 0.0;
    int best_col_assignment = -1;

    Instance_Cost_Pair_Cus_Mem new_icp;
    new_icp.instance_mem_elem = local_instance_vector;

    new_icp.next_row_to_make_decision = 0;
    new_icp.total_num_assignments = 0;

    //std::cout << get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),feature_points->size()) << std::endl;

    if(!check_if_instance_array_column_is_occupied((int*)local_instance_vector.memory_pointer,total_rows,-1)){
        MEM_ELEM_WRITE(local_instance_vector,int,decision_row,-1);

        new_icp.next_row_to_make_decision = decision_row + 1;
        new_icp.total_num_assignments = num_previous_assignments;

        new_icp.cost = local_best_icp.cost + get_cost_for_single_additional_assignment_custom_memory(local_instance_vector,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,start_row,decision_row,cost_rescale_factors);
        //std::cout << "0 decision: ";
        new_icp.heuristic_cost_for_remaining_decision = local_best_icp.heuristic_cost_for_remaining_decision - get_change_of_heuristic_cost_after_decision(feature_points->size(),feature_points->size() - num_remaining_decisions,-1,new_icp.total_num_assignments,cost_rescale_factors);
        //new_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),num_remaining_decisions);
        //std::cout << "0 decision: " << new_icp.heuristic_cost_for_remaining_decision << " " << get_change_of_heuristic_cost_after_decision(feature_points->size(),feature_points->size() - num_remaining_decisions,-1,new_icp.total_num_assignments) << std::endl;
        new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

        if(new_icp.total_cost < current_best_cost){
            current_best_cost = new_icp.total_cost;
            current_best_cost_from_assignments = new_icp.cost;
            current_best_heuristic_cost_for_remaining_decisions = new_icp.heuristic_cost_for_remaining_decision;
            best_col_assignment = -1;
            new_total_num_assignements = new_icp.total_num_assignments;
        }
    }

    new_icp.next_row_to_make_decision = decision_row + 1;
    new_icp.total_num_assignments = num_previous_assignments + 1;
    //new_icp.instance_mem_elem = local_instance_vector;

    //std::cout << "1 decision: "; 
    //double heuristic_cost = get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),num_remaining_decisions);
    double heuristic_cost_change = get_change_of_heuristic_cost_after_decision(feature_points->size(),feature_points->size() - num_remaining_decisions,1,new_icp.total_num_assignments,cost_rescale_factors);
    //std::cout << "1 decision: " << heuristic_cost << " " << heuristic_cost_change << std::endl;


    for(int col = 0; col < total_cols; col++){
        if(!check_if_instance_array_column_is_occupied((int*)local_instance_vector.memory_pointer,total_rows,col)){
            MEM_ELEM_WRITE(local_instance_vector,int,decision_row,col);

            new_icp.cost = local_best_icp.cost + get_cost_for_single_additional_assignment_custom_memory(local_instance_vector,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,start_row,decision_row,cost_rescale_factors);
            new_icp.heuristic_cost_for_remaining_decision = local_best_icp.heuristic_cost_for_remaining_decision - heuristic_cost_change;
            //new_icp.heuristic_cost_for_remaining_decision = heuristic_cost;
            //std::cout << new_icp.heuristic_cost_for_remaining_decision << std::endl;
            new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

            if(new_icp.total_cost < current_best_cost){
                current_best_cost = new_icp.total_cost;
                current_best_cost_from_assignments = new_icp.cost;
                current_best_heuristic_cost_for_remaining_decisions = new_icp.heuristic_cost_for_remaining_decision;
                best_col_assignment = col;
                new_total_num_assignements = new_icp.total_num_assignments;
            }

        }
    }        

    original_value = best_col_assignment;

    MEM_ELEM_WRITE(local_instance_vector,int,decision_row,original_value);
    local_best_icp.total_cost = current_best_cost;
    local_best_icp.cost = current_best_cost_from_assignments;
    local_best_icp.heuristic_cost_for_remaining_decision = current_best_heuristic_cost_for_remaining_decisions;
    local_best_icp.total_num_assignments = new_total_num_assignements;
    local_best_icp.next_row_to_make_decision++;
}

void get_best_instance_of_single_decision_row(int total_rows, int total_cols, Instance_Cost_Pair_Cus_Mem& local_best_icp, int num_previous_assignments, int num_remaining_decisions, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors){

    int original_value = ((int*)(local_best_icp.instance_mem_elem.memory_pointer))[local_best_icp.next_row_to_make_decision];

    Memory_Element local_instance_vector = local_best_icp.instance_mem_elem;

    int decision_row = local_best_icp.next_row_to_make_decision;
    int new_total_num_assignements = num_previous_assignments;
    double current_best_cost = 0.0;
    int best_col_assignment = -1;

    Instance_Cost_Pair_Cus_Mem new_icp;
    new_icp.instance_mem_elem = local_instance_vector;

    if(!check_if_instance_array_column_is_occupied((int*)local_instance_vector.memory_pointer,total_rows,-1)){
        MEM_ELEM_WRITE(local_instance_vector,int,decision_row,-1);

        new_icp.next_row_to_make_decision = decision_row + 1;
        new_icp.total_num_assignments = num_previous_assignments;

        new_icp.cost = get_cost_for_single_problem_instance_custom_memory(local_instance_vector,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
        new_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),cost_rescale_factors,num_remaining_decisions);
        //std::cout << new_icp.heuristic_cost_for_remaining_decision << std::endl;
        new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

        if(new_icp.total_cost < current_best_cost){
            current_best_cost = new_icp.total_cost;
            best_col_assignment = -1;
            new_total_num_assignements = new_icp.total_num_assignments;
        }
    }

    new_icp.next_row_to_make_decision = decision_row + 1;
    new_icp.total_num_assignments = num_previous_assignments + 1;
    //new_icp.instance_mem_elem = local_instance_vector;

    double heuristic_cost = get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),cost_rescale_factors,num_remaining_decisions);
    //std::cout << heuristic_cost<< std::endl;

    for(int col = 0; col < total_cols; col++){
        if(!check_if_instance_array_column_is_occupied((int*)local_instance_vector.memory_pointer,total_rows,col)){
            MEM_ELEM_WRITE(local_instance_vector,int,decision_row,col);

            new_icp.cost = get_cost_for_single_problem_instance_custom_memory(local_instance_vector,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
            new_icp.heuristic_cost_for_remaining_decision = heuristic_cost;
            //std::cout << new_icp.heuristic_cost_for_remaining_decision << std::endl;
            new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

            if(new_icp.total_cost < current_best_cost){
                current_best_cost = new_icp.total_cost;
                best_col_assignment = col;
                new_total_num_assignements = new_icp.total_num_assignments;
            }

        }
    }        

    original_value = best_col_assignment;

    MEM_ELEM_WRITE(local_instance_vector,int,decision_row,original_value);
    local_best_icp.total_cost = current_best_cost;
    local_best_icp.total_num_assignments = new_total_num_assignements;
    local_best_icp.next_row_to_make_decision++;
}

Instance_Cost_Pair_Cus_Mem get_best_new_instance_custom_memory(std::vector<Memory_Element>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row , uint16_t num_previous_assignments, int* num_deleted_instances, int num_remaining_decisions, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors){

    std::priority_queue<Instance_Cost_Pair_Cus_Mem,std::vector<Instance_Cost_Pair_Cus_Mem>,Queue_Cost_Compare_Cus_Mem> tmp_instance_queue;

    Instance_Cost_Pair_Cus_Mem best_icp_from_new_subinstances;

    for(int i = 0; i < new_instances->size();i++){
        Memory_Element current_instance = (*new_instances)[i];

        BaB_DEBUG_PRINT(
            print_instance_array_as_matrix((int*)current_instance.memory_pointer,feature_points->size(),candidates->size());
        )
        
        Instance_Cost_Pair_Cus_Mem new_icp;

        new_icp.instance_mem_elem = current_instance;
        new_icp.next_row_to_make_decision = decision_row + 1;
        new_icp.total_num_assignments = num_previous_assignments;

        if((MEM_ELEM_ACCESS(current_instance,int,decision_row)) != -1){
            new_icp.total_num_assignments++;
        }

        new_icp.cost = get_cost_for_single_problem_instance_custom_memory(current_instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
        new_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),cost_rescale_factors,num_remaining_decisions);
        //std::cout << new_icp.heuristic_cost_for_remaining_decision << std::endl;
        new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

        tmp_instance_queue.push(new_icp);
    }

    //std::cout << "loop end" << std::endl;

    best_icp_from_new_subinstances = tmp_instance_queue.top();
    tmp_instance_queue.pop();

    while(!tmp_instance_queue.empty()){
        Instance_Cost_Pair_Cus_Mem current_icp = tmp_instance_queue.top();
        free_memory_element(mem_manager,&(current_icp.instance_mem_elem));
        (*num_deleted_instances)++;
        
        tmp_instance_queue.pop();
    }

    return best_icp_from_new_subinstances;

}

Instance_Cost_Pair_Cus_Mem get_best_new_instance_and_put_rest_on_queue_custom_memory(std::priority_queue<Instance_Cost_Pair_Cus_Mem,std::vector<Instance_Cost_Pair_Cus_Mem>,Queue_Cost_Compare_Cus_Mem>* instance_queues, std::vector<Memory_Element>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row , uint16_t num_previous_assignments, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors){

    std::priority_queue<Instance_Cost_Pair_Cus_Mem,std::vector<Instance_Cost_Pair_Cus_Mem>,Queue_Cost_Compare_Cus_Mem> tmp_instance_queue;

    Instance_Cost_Pair_Cus_Mem best_icp_from_new_subinstances;

    for(int i = 0; i < new_instances->size();i++){
        Memory_Element current_instance = (*new_instances)[i];

        BaB_DEBUG_PRINT(
            //print_instance_array_as_matrix((int*)current_instance.memory_pointer,feature_points->size(),candidates->size());
            print_instance_array_as_vector((int*)current_instance.memory_pointer,feature_points->size());
        )
        
        Instance_Cost_Pair_Cus_Mem new_icp;

        new_icp.instance_mem_elem = current_instance;
        new_icp.next_row_to_make_decision = decision_row + 1;
        new_icp.total_num_assignments = num_previous_assignments;

        int value_in_decision_row = MEM_ELEM_ACCESS(current_instance,int,decision_row);

        if(value_in_decision_row != -1){
            new_icp.total_num_assignments++;
        }

        new_icp.cost = get_cost_for_single_problem_instance_custom_memory(current_instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
        new_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions_custom_memory(new_icp,feature_points->size(),cost_rescale_factors);
        new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

        tmp_instance_queue.push(new_icp);
    }


    best_icp_from_new_subinstances = tmp_instance_queue.top();
    tmp_instance_queue.pop();

    while(!tmp_instance_queue.empty()){
        instance_queues->push(tmp_instance_queue.top());
        tmp_instance_queue.pop();
    }

    return best_icp_from_new_subinstances;
}

Instance_Cost_Pair get_best_new_instance_and_put_rest_on_queue(std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queues, std::vector<std::vector<int>*>* new_instances, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, uint16_t decision_row , uint16_t num_previous_assignments, Cost_Rescale_Factors cost_rescale_factors){

    std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare> tmp_instance_queue;

    Instance_Cost_Pair best_icp_from_new_subinstances;

    for(int i = 0; i < new_instances->size();i++){
        std::vector<int>* current_instance = (*new_instances)[i];

        BaB_DEBUG_PRINT(
            print_instance_vector_as_matrix(current_instance,feature_points->size(),candidates->size());
        )
        
        Instance_Cost_Pair new_icp;

        new_icp.instance = current_instance;
        new_icp.next_row_to_make_decision = decision_row + 1;
        new_icp.total_num_assignments = num_previous_assignments;

        if((*current_instance)[decision_row] != -1){
            new_icp.total_num_assignments++;
        }

        new_icp.cost = get_cost_for_single_problem_instance(current_instance,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
        new_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions(new_icp,feature_points->size(),cost_rescale_factors);
        new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

        tmp_instance_queue.push(new_icp);
    }

    best_icp_from_new_subinstances = tmp_instance_queue.top();
    tmp_instance_queue.pop();

    while(!tmp_instance_queue.empty()){
        instance_queues->push(tmp_instance_queue.top());
        tmp_instance_queue.pop();
    }

    return best_icp_from_new_subinstances;

}

void print_instance_queue(std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queue,int instance_size_rows, int instance_size_cols, bool sparse_print){

    std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare> local_instance_queue = *instance_queue;

    int index = 0;

    std::cout << "Start of Instance Queue: " << std::endl;

    while(!local_instance_queue.empty()){
        
        Instance_Cost_Pair current_icp = local_instance_queue.top();

        if(sparse_print){
            std::cout << current_icp.next_row_to_make_decision;
            print_instance_vector_as_vector(current_icp.instance);

        }else{

            std::cout << std::endl;
            std::cout << "Queue index: " << index << std::endl;
            std::cout << "Instance cost: " << current_icp.total_cost << "  Heuristic cost: " << current_icp.heuristic_cost_for_remaining_decision << "  " << std::endl;
            std::cout << "Next decision in row: " << current_icp.next_row_to_make_decision << std::endl;
            std::cout << "Total num assignments: " << current_icp.total_num_assignments << std::endl;
            print_instance_vector_as_matrix(current_icp.instance,instance_size_rows,instance_size_cols);
        }

        local_instance_queue.pop();

        index++;

    }



}

double get_optimal_cost_for_problem_size(int problem_size, Cost_Rescale_Factors cost_rescale_factors){

    Instance_Cost_Pair dummy_icp;
    dummy_icp.next_row_to_make_decision = 0;
    dummy_icp.total_num_assignments = 0;

    return get_heuristic_cost_for_remaining_decisions(dummy_icp,problem_size,cost_rescale_factors);

}

double get_optimal_cost_for_problem_size_using_cost_params(int problem_size, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params){

    Instance_Cost_Pair dummy_icp;
    dummy_icp.next_row_to_make_decision = 0;
    dummy_icp.total_num_assignments = 0;

    return get_heuristic_cost_for_remaining_decisions_using_cost_params(dummy_icp,problem_size,cost_rescale_factors,-1,cost_params);

}

double calc_heuristic_cost_for_remaining_decisions_using_cost_params(int total_num_decision, int num_remaining_decisions,int next_decision_row, int total_num_assignments, Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params){

    double total_heuristic_cost = 0.0;

    if(num_remaining_decisions == -1){
        num_remaining_decisions = total_num_decision - next_decision_row;
    }

    //total_heuristic_cost += num_remaining_decisions * get_heuristic_unary_costs();
    double heuristic_cost_for_remaining_unary_decisions = num_remaining_decisions * get_heuristic_unary_costs_using_cost_params(cost_rescale_factors,cost_params);
    total_heuristic_cost += heuristic_cost_for_remaining_unary_decisions;

    int num_potentially_perfect_pairs = num_remaining_decisions * total_num_assignments;

    for(int i = 0; i < num_remaining_decisions; i++){
        num_potentially_perfect_pairs += i;
    }

    //total_heuristic_cost += num_potentially_perfect_pairs * get_heuristic_pairwise_costs();
    double total_heuristic_cost_from_pairwise_decisions = num_potentially_perfect_pairs * get_heuristic_pairwise_costs_using_cost_params(cost_rescale_factors,cost_params);
    total_heuristic_cost += total_heuristic_cost_from_pairwise_decisions;

    //std::cout << "total_num_decision: " << total_num_decision << " num_remaining_decisions: " << num_remaining_decisions << " next_decision_row: " << next_decision_row << " total_num_assignments: " << total_num_assignments << " total_heuristic_cost: " << total_heuristic_cost << std::endl;

    return total_heuristic_cost;
}

double calc_heuristic_cost_for_remaining_decisions(int total_num_decision, int num_remaining_decisions,int next_decision_row, int total_num_assignments, Cost_Rescale_Factors cost_rescale_factors){

    double total_heuristic_cost = 0.0;

    if(num_remaining_decisions == -1){
        num_remaining_decisions = total_num_decision - next_decision_row;
    }

    //total_heuristic_cost += num_remaining_decisions * get_heuristic_unary_costs();
    double heuristic_cost_for_remaining_unary_decisions = num_remaining_decisions * get_heuristic_unary_costs(cost_rescale_factors);
    total_heuristic_cost += heuristic_cost_for_remaining_unary_decisions;

    int num_potentially_perfect_pairs = num_remaining_decisions * total_num_assignments;

    for(int i = 0; i < num_remaining_decisions; i++){
        num_potentially_perfect_pairs += i;
    }

    //total_heuristic_cost += num_potentially_perfect_pairs * get_heuristic_pairwise_costs();
    double total_heuristic_cost_from_pairwise_decisions = num_potentially_perfect_pairs * get_heuristic_pairwise_costs(cost_rescale_factors);
    total_heuristic_cost += total_heuristic_cost_from_pairwise_decisions;

    //std::cout << "total_num_decision: " << total_num_decision << " num_remaining_decisions: " << num_remaining_decisions << " next_decision_row: " << next_decision_row << " total_num_assignments: " << total_num_assignments << " total_heuristic_cost: " << total_heuristic_cost << std::endl;

    return total_heuristic_cost;
}

double get_heuristic_cost_for_remaining_decisions(Instance_Cost_Pair& icp, int total_num_decision, Cost_Rescale_Factors cost_rescale_factors, int num_remaining_decisions){

    return calc_heuristic_cost_for_remaining_decisions(total_num_decision,num_remaining_decisions,icp.next_row_to_make_decision,icp.total_num_assignments,cost_rescale_factors);
}

double get_heuristic_cost_for_remaining_decisions_using_cost_params(Instance_Cost_Pair& icp, int total_num_decision, Cost_Rescale_Factors cost_rescale_factors, int num_remaining_decisions, All_Cost_Parameters& cost_params){

    return calc_heuristic_cost_for_remaining_decisions_using_cost_params(total_num_decision,num_remaining_decisions,icp.next_row_to_make_decision,icp.total_num_assignments,cost_rescale_factors,cost_params);
}

double get_heuristic_cost_for_remaining_decisions_custom_memory(Instance_Cost_Pair_Cus_Mem& icp, int total_num_decision, Cost_Rescale_Factors cost_rescale_factors, int num_remaining_decisions){

    return calc_heuristic_cost_for_remaining_decisions(total_num_decision,num_remaining_decisions,icp.next_row_to_make_decision,icp.total_num_assignments,cost_rescale_factors);
}

double get_change_of_heuristic_cost_after_decision(unsigned int total_num, unsigned int decision_row, int decision_var, unsigned int total_num_previous_assignments, Cost_Rescale_Factors cost_rescale_factors){

    //at first we "remove" the unary cost
    double cost_change = get_heuristic_unary_costs(cost_rescale_factors);

    // then we remove all the potentially perfect pairs we could have formed with all previous assignments 
    // and which of course now have been replace with the actual costs for the pairs with these assignments in the costs variable of the icp

    // TODO: this obviously returns a very large number if total_num_previous_assignemnts is equal to 0
    // we should investigate if we can find a workaround without using another if statement.
    if(total_num_previous_assignments > 0){
        cost_change += (total_num_previous_assignments - 1) * get_heuristic_pairwise_costs(cost_rescale_factors);
    }

    // lastly if we have chosen not to assign a candidate to this feature then of course all the follwing decisions cannot form a pair with a non existent assignment and therefore we have to subtract them aswell
    if(decision_var == -1){
        cost_change += (total_num - decision_row) * get_heuristic_pairwise_costs(cost_rescale_factors);
    }

    //std::cout << "cost_change in heuristic after decision: " << cost_change << std::endl;

    return cost_change;

}

double get_heuristic_unary_costs(Cost_Rescale_Factors cost_rescale_factors){

    return cost_rescale_factors.heuristic_unary_cost;
}

double get_heuristic_pairwise_costs(Cost_Rescale_Factors cost_rescale_factors){

    return cost_rescale_factors.heurisitc_quadr_cost;
}

double get_heuristic_unary_costs_using_cost_params(Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params){

    return cost_rescale_factors.heuristic_unary_cost;
}

double get_heuristic_pairwise_costs_using_cost_params(Cost_Rescale_Factors cost_rescale_factors, All_Cost_Parameters& cost_params){

    return cost_rescale_factors.heurisitc_quadr_cost;
}

void greedy_search_for_initial_solution_custom_memory(std::priority_queue<Instance_Cost_Pair_Cus_Mem,std::vector<Instance_Cost_Pair_Cus_Mem>,Queue_Cost_Compare_Cus_Mem>* instance_queue, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair_Cus_Mem* best_icp, int* num_created_instances, int* num_deleted_instances, Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors){

    int num_feature_points = feature_points->size();
    int num_candidates = candidates->size();

    Instance_Cost_Pair_Cus_Mem local_best_icp;

    std::vector<Memory_Element> subinstances;

    /*
    subinstances.resize(num_candidates);

    for(int i = 0; i < num_candidates;i++){
        subinstances[i] = allocate_new_memory_element(mem_manager);
    }
    */

    Memory_Element initial_instance = allocate_new_memory_element(mem_manager);
    (*num_created_instances)++;

    for(int i = 0; i < num_feature_points;i++){
        ((int*)initial_instance.memory_pointer)[i] = -1;
    }

    list_all_subinstances_recursively_custom_memory(&initial_instance,0,0,num_feature_points,num_candidates,subinstances,num_created_instances,mem_manager);

    free_memory_element(mem_manager,&initial_instance);
    (*num_deleted_instances)++;

    local_best_icp = get_best_new_instance_and_put_rest_on_queue_custom_memory(instance_queue,&subinstances,feature_points,candidates,0,0,mem_manager,cost_rescale_factors);

    while(local_best_icp.next_row_to_make_decision < num_feature_points){
        subinstances.clear();

        list_all_subinstances_recursively_custom_memory(&(local_best_icp.instance_mem_elem),local_best_icp.next_row_to_make_decision,local_best_icp.next_row_to_make_decision,num_feature_points,num_candidates,subinstances,num_created_instances,mem_manager);

        free_memory_element(mem_manager,&(local_best_icp.instance_mem_elem));
        (*num_deleted_instances)++;

        local_best_icp = get_best_new_instance_and_put_rest_on_queue_custom_memory(instance_queue,&subinstances,feature_points,candidates,local_best_icp.next_row_to_make_decision,local_best_icp.total_num_assignments,mem_manager,cost_rescale_factors);
    }

    if(local_best_icp.total_cost < best_icp->total_cost){
        *best_icp = local_best_icp;
        //std::cout << "greedy search found a negative cost solution" << std::endl;
    }else {
        //std::cout << "greedy search DID NOT FIND a better solution" << std::endl;
    }
}

void greedy_search_for_initial_solution(std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queue, std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair* best_icp, int* num_created_instances, int* num_deleted_instances, Cost_Rescale_Factors cost_rescale_factors){

    int num_feature_points = feature_points->size();
    int num_candidates = candidates->size();

    Instance_Cost_Pair local_best_icp;

    std::vector<std::vector<int>*> subinstances;

    std::vector<int>* initial_instance = new std::vector<int>;//(num_feature_points);
    (*num_created_instances)++;

    initial_instance->reserve(num_feature_points);

    for(int i = 0; i < num_feature_points;i++){
        initial_instance->push_back(-1);
    }

    list_all_subinstances_recursively(initial_instance,0,0,num_feature_points,num_candidates,subinstances,num_created_instances);

    delete(initial_instance);
    (*num_deleted_instances)++;

    
    local_best_icp = get_best_new_instance_and_put_rest_on_queue(instance_queue,&subinstances,feature_points,candidates,0,0,cost_rescale_factors);

    while(local_best_icp.next_row_to_make_decision < num_feature_points){
        subinstances.clear();

        list_all_subinstances_recursively(local_best_icp.instance,local_best_icp.next_row_to_make_decision,local_best_icp.next_row_to_make_decision,num_feature_points,num_candidates,subinstances,num_created_instances);

        delete(local_best_icp.instance);
        (*num_deleted_instances)++;

        local_best_icp = get_best_new_instance_and_put_rest_on_queue(instance_queue,&subinstances,feature_points,candidates,local_best_icp.next_row_to_make_decision,local_best_icp.total_num_assignments,cost_rescale_factors);

    }

    if(local_best_icp.total_cost < best_icp->total_cost){
        *best_icp = local_best_icp;
        //std::cout << "greedy search found a negative cost solution" << std::endl;
    }else {
        //std::cout << "greedy search DID NOT FIND a better solution" << std::endl;
    }


    //std::cout << "num deleted instances after greedy search: " << num_deleted_instances << std::endl;

}

void greedy_search_for_initial_solution_in_row_range(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair* best_icp, int* num_created_instances, int* num_deleted_instances, int start_row, int end_row, Cost_Rescale_Factors cost_rescale_factors){

    //std::cout << "search in range: " << start_row << " to " << end_row << std::endl;

    int num_feature_points = feature_points->size();
    int num_candidates = candidates->size();

    Instance_Cost_Pair local_best_icp;

    std::vector<std::vector<int>*> subinstances;

    std::vector<int>* initial_instance = new std::vector<int>;//(num_feature_points);
    (*num_created_instances)++;

    initial_instance->reserve(num_feature_points);

    for(int i = 0; i < num_feature_points;i++){
        initial_instance->push_back(-1);
    }

    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //std::cout << "time greedy rotating and distance matching = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[�s]" << std::endl;

    int num_unnecessary_iterations = 0;

    for(int i = start_row; i < end_row;i++){

        //std::cout << "start of iteration" << std::endl;
        /*
        if (num_feature_points > 100 && (i % 10 == 0) && i > 0) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            int total_num_iterations = end_row - start_row;

            std::cout << "at iteration: " << i-start_row << " of " << total_num_iterations << " elapsed time: " << (std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() / 60.0f) << " mins  estimated remaining time: " << ( (std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() / (float)i) * total_num_iterations)/60.0f << " mins" << std::endl;

        }
        */

        subinstances.clear();
        
        int start_row = i;

        list_all_subinstances_recursively(initial_instance,start_row,start_row,num_feature_points,num_candidates,subinstances,num_created_instances);
        int num_decisions_made = 1;

        local_best_icp = get_best_new_instance(&subinstances,feature_points,candidates,start_row,0,num_deleted_instances, num_feature_points-num_decisions_made,cost_rescale_factors);
        //std::cout << local_best_icp.next_row_to_make_decision << std::endl;


        while(num_decisions_made < num_feature_points){
            subinstances.clear();
            local_best_icp.next_row_to_make_decision = local_best_icp.next_row_to_make_decision % num_feature_points;

            list_all_subinstances_recursively(local_best_icp.instance,local_best_icp.next_row_to_make_decision,local_best_icp.next_row_to_make_decision,num_feature_points,num_candidates,subinstances,num_created_instances);
            num_decisions_made++;

            if(local_best_icp.instance != best_icp->instance){
                delete(local_best_icp.instance);
                (*num_deleted_instances)++;
            }

            local_best_icp = get_best_new_instance(&subinstances,feature_points,candidates,local_best_icp.next_row_to_make_decision,local_best_icp.total_num_assignments,num_deleted_instances, num_feature_points-num_decisions_made,cost_rescale_factors);
            //std::cout << local_best_icp.next_row_to_make_decision << std::endl;

            if (local_best_icp.total_cost > best_icp->total_cost) {
                break;
                //num_unnecessary_iterations++;
                //std::cout << "best new local "<< local_best_icp.total_cost << " was worse then global best" << best_icp->total_cost << std::endl;
            }
        }

        if (local_best_icp.total_cost < best_icp->total_cost) {
            if (best_icp->instance != nullptr) {
                delete(best_icp->instance);
                (*num_deleted_instances)++;
            }

            *best_icp = local_best_icp;
            //std::cout << "cyclic greedy search found a negative cost solution" << std::endl;
        }

        if(local_best_icp.instance != best_icp->instance){
            delete(local_best_icp.instance);
            (*num_deleted_instances)++;
        }
        //std::cout << "after while loop iteration created: " << *num_created_instances << " num deleted instances: " << *num_deleted_instances << " diff: " << *num_created_instances - *num_deleted_instances << std::endl; 

    }

    //std::cout << "end of cyclic greedy search for: " << num_feature_points << "  num unnecessary iterations was: " << num_unnecessary_iterations << std::endl;

    delete(initial_instance);
    (*num_deleted_instances)++;

}

void greedy_search_for_initial_solution_in_row_range_custom_memory(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair_Cus_Mem* best_icp, int* num_created_instances, int* num_deleted_instances, int start_row, int end_row,Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors){

    //std::cout << "search in range: " << start_row << " to " << end_row << std::endl;

    int num_feature_points = feature_points->size();
    int num_candidates = candidates->size();

    Instance_Cost_Pair_Cus_Mem local_best_icp;

    std::vector<Memory_Element> subinstances;

    Memory_Element initial_instance;// = allocate_new_memory_element(mem_manager);//(num_feature_points);
    initial_instance.bit_index = 0;
    initial_instance.leaf_index = 0;
    initial_instance.memory_pointer = (char*)malloc(feature_points->size() * sizeof(int));
    (*num_created_instances)++;

    for(int i = 0; i < num_feature_points;i++){
        MEM_ELEM_WRITE(initial_instance,int,i,-1);
    }

    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //std::cout << "time greedy rotating and distance matching = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[�s]" << std::endl;

    int num_unnecessary_iterations = 0;

    for(int i = start_row; i < end_row;i++){

        subinstances.clear();
        
        int start_row = i;

        list_all_subinstances_recursively_custom_memory(&initial_instance,start_row,start_row,num_feature_points,num_candidates,subinstances,num_created_instances,mem_manager);
        int num_decisions_made = 1;

        local_best_icp = get_best_new_instance_custom_memory(&subinstances,feature_points,candidates,start_row,0,num_deleted_instances, num_feature_points-num_decisions_made,mem_manager,cost_rescale_factors);
        //std::cout << local_best_icp.next_row_to_make_decision << std::endl;


        while(num_decisions_made < num_feature_points){
            subinstances.clear();
            local_best_icp.next_row_to_make_decision = local_best_icp.next_row_to_make_decision % num_feature_points;

            list_all_subinstances_recursively_custom_memory(&(local_best_icp.instance_mem_elem),local_best_icp.next_row_to_make_decision,local_best_icp.next_row_to_make_decision,num_feature_points,num_candidates,subinstances,num_created_instances,mem_manager);
            num_decisions_made++;

            /*
            if(local_best_icp.instance != best_icp->instance){
                delete(local_best_icp.instance);
                (*num_deleted_instances)++;
            }
            */

            local_best_icp = get_best_new_instance_custom_memory(&subinstances,feature_points,candidates,local_best_icp.next_row_to_make_decision,local_best_icp.total_num_assignments,num_deleted_instances, num_feature_points-num_decisions_made,mem_manager,cost_rescale_factors);

            if (local_best_icp.total_cost > best_icp->total_cost) {
                break;
            }
        }

        if (local_best_icp.total_cost < best_icp->total_cost) {
            /*
            if (best_icp->instance != nullptr) {
                delete(best_icp->instance);
                (*num_deleted_instances)++;
            }
            */
            memcpy(best_icp->instance_mem_elem.memory_pointer,local_best_icp.instance_mem_elem.memory_pointer,mem_manager->size_of_single_element);
            best_icp->total_cost = local_best_icp.total_cost;
            //*best_icp = local_best_icp;
            //std::cout << "cyclic greedy search found a negative cost solution" << std::endl;
        }

        /*
        if(local_best_icp.instance != best_icp->instance){
            delete(local_best_icp.instance);
            (*num_deleted_instances)++;
        }
        */
        //std::cout << "after while loop iteration created: " << *num_created_instances << " num deleted instances: " << *num_deleted_instances << " diff: " << *num_created_instances - *num_deleted_instances << std::endl; 

    }

    //std::cout << "end of cyclic greedy search for: " << num_feature_points << "  num unnecessary iterations was: " << num_unnecessary_iterations << std::endl;
    
    free(initial_instance.memory_pointer);
    (*num_deleted_instances)++;
    

}

void cyclic_search_custom_memory(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair_Cus_Mem* best_icp, int* num_created_instances, int* num_deleted_instances, int start_row, int end_row,Memory_Manager_Fixed_Size* mem_manager, Cost_Rescale_Factors cost_rescale_factors){

    int num_feature_points = feature_points->size();
    int num_candidates = candidates->size();

    Instance_Cost_Pair_Cus_Mem local_best_icp;

    Memory_Element initial_instance;
    initial_instance.bit_index = 0;
    initial_instance.leaf_index = 0;
    initial_instance.memory_pointer = (char*)malloc(feature_points->size() * sizeof(int));
    (*num_created_instances)++;

    local_best_icp.instance_mem_elem = initial_instance;
    //std::cout << "begin of single cyclic iteration" << std::endl;

    for(int i = start_row; i < end_row;i++){

        for(int j = 0; j < num_feature_points;j++){
            MEM_ELEM_WRITE(initial_instance,int,j,-1);
        }
     
        int num_decisions_made = 0;

        local_best_icp.next_row_to_make_decision = i;
        local_best_icp.total_num_assignments = 0;
        local_best_icp.total_cost = 0.0;
        local_best_icp.cost = 0.0;
        local_best_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions_custom_memory(local_best_icp,feature_points->size(),cost_rescale_factors,feature_points->size());

        int col_assigned = -1;

        while(num_decisions_made < num_feature_points){

            local_best_icp.next_row_to_make_decision = local_best_icp.next_row_to_make_decision % num_feature_points;

            num_decisions_made++;
            //get_best_instance_of_single_decision_row(num_feature_points,num_candidates,local_best_icp,local_best_icp.total_num_assignments,num_feature_points-num_decisions_made,feature_points,candidates,mem_manager);

            //get_best_instance_of_single_decision_row_with_iterative_cost(num_feature_points,num_candidates,local_best_icp,local_best_icp.total_num_assignments,num_feature_points-num_decisions_made,i,feature_points,candidates,mem_manager);

            get_best_instance_of_single_decision_row_with_iterative_cost_in_col_range(num_feature_points,num_candidates,local_best_icp,local_best_icp.total_num_assignments,num_feature_points-num_decisions_made,i,col_assigned,feature_points,candidates,mem_manager,cost_rescale_factors);

            if(local_best_icp.total_cost > best_icp->total_cost){
                //std::cout << "break" << std::endl;
                break;
            }

        }

        //std::cout << local_best_icp.total_cost << std::endl;

        if(local_best_icp.total_cost < best_icp->total_cost){
            best_icp->total_cost = local_best_icp.total_cost;
            best_icp->cost = local_best_icp.cost;
            best_icp->heuristic_cost_for_remaining_decision = local_best_icp.heuristic_cost_for_remaining_decision;
            memcpy(best_icp->instance_mem_elem.memory_pointer,local_best_icp.instance_mem_elem.memory_pointer,mem_manager->size_of_single_element);
        }
    }

    //std::cout << best_icp->total_cost << " " << best_icp->cost << " " << best_icp->heuristic_cost_for_remaining_decision << std::endl;
    //print_instance_array_as_vector((int*)best_icp->instance_mem_elem.memory_pointer,feature_points->size());
    //std::cout << get_cost_for_single_problem_instance_custom_memory(best_icp->instance_mem_elem,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST) << std::endl; 

}

void cyclic_greedy_search_for_initial_solution(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Instance_Cost_Pair* best_icp, int* num_created_instances, int* num_deleted_instances, Cost_Rescale_Factors cost_rescale_factors, Memory_Manager_Fixed_Size* mem_manager, double cost_bound_of_already_found_solution,const std::vector<int>* already_found_solution){

    bool parallelization_useful = false;//feature_points->size() > 250 && candidates->size() > 250;

    //std::cout << "parallelization useful: " << parallelization_useful << std::endl;

    if(parallelization_useful){
        unsigned int max_num_hardware_threads = std::thread::hardware_concurrency();


        unsigned int num_additional_threads = max_num_hardware_threads - 1;
        std::vector<std::thread*> additional_threads;
        additional_threads.resize(num_additional_threads);

        std::vector<Instance_Cost_Pair*> best_icps_per_thread;
        std::vector<int*> num_created_instances_per_thread;
        std::vector<int*> num_deleted_instances_per_thread;

        int num_rows_per_thread = feature_points->size() / max_num_hardware_threads;

        for(int i = 0; i < num_additional_threads;i++){

            Instance_Cost_Pair* new_icp = new Instance_Cost_Pair;
            new_icp->total_cost = 0.0;
            new_icp->instance = nullptr;

            best_icps_per_thread.push_back(new_icp);

            int* new_ci = new int;
            *new_ci = 0;

            int* new_di = new int;
            *new_di = 0;

            num_created_instances_per_thread.push_back(new_ci);
            num_deleted_instances_per_thread.push_back(new_di);

            int start_row = i * num_rows_per_thread;
            int end_row = (i+1) * num_rows_per_thread;

            additional_threads[i] = new std::thread(greedy_search_for_initial_solution_in_row_range,feature_points,candidates,best_icps_per_thread[i],num_created_instances_per_thread[i],num_deleted_instances_per_thread[i],start_row,end_row,cost_rescale_factors);
        }

        int start_row_for_original_thread = num_additional_threads * num_rows_per_thread;
        int end_row_for_original_thread = feature_points->size();

        greedy_search_for_initial_solution_in_row_range(feature_points,candidates,best_icp,num_created_instances,num_deleted_instances,start_row_for_original_thread,end_row_for_original_thread,cost_rescale_factors); 

        for(int i = 0; i < num_additional_threads;i++){
            additional_threads[i]->join();
        }

        for(int i = 0; i < num_additional_threads;i++){
            if(best_icps_per_thread[i]->total_cost < best_icp->total_cost){
                if(best_icp->instance != nullptr){
                    delete(best_icp->instance);
                    (*num_deleted_instances)++;
                }

                *best_icp = *(best_icps_per_thread[i]);
            }else{
                if(best_icps_per_thread[i]->instance != nullptr){
                    delete(best_icps_per_thread[i]->instance);
                    (*num_deleted_instances)++;
                }

            }

            delete(best_icps_per_thread[i]);

            (*num_created_instances) += *(num_created_instances_per_thread[i]);
            (*num_deleted_instances) += *(num_deleted_instances_per_thread[i]);

            delete(num_created_instances_per_thread[i]);
            delete(num_deleted_instances_per_thread[i]);

        }

    }else{

        if(!USE_CUSTOM_MEMORY_HANDLER){

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            greedy_search_for_initial_solution_in_row_range(feature_points,candidates,best_icp,num_created_instances,num_deleted_instances,0,feature_points->size(),cost_rescale_factors);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        }else{
            Memory_Element best_mem_elem;
            best_mem_elem.bit_index = 0;
            best_mem_elem.leaf_index = 0;
            best_mem_elem.memory_pointer = (char*)malloc(mem_manager->size_of_single_element);

            for(int i = 0; i < feature_points->size();i++){
                int value = -1;

                if(already_found_solution != nullptr && i < already_found_solution->size()){
                    value = (*already_found_solution)[i];
                }

                MEM_ELEM_WRITE(best_mem_elem,int,i,value);
            }

            int num_created_in_custom = 0;
            int num_deleted_in_custom = 0;

            Instance_Cost_Pair_Cus_Mem best_icp_cus_mem;
            best_icp_cus_mem.total_cost = cost_bound_of_already_found_solution;
            best_icp_cus_mem.instance_mem_elem = best_mem_elem;


            //std::chrono::steady_clock::time_point custom_begin = std::chrono::steady_clock::now();
            //greedy_search_for_initial_solution_in_row_range_custom_memory(feature_points,candidates,&best_icp_cus_mem,&num_created_in_custom,&num_deleted_in_custom,0,feature_points->size(),mem_manager);
            cyclic_search_custom_memory(feature_points,candidates,&best_icp_cus_mem,&num_created_in_custom,&num_deleted_in_custom,0,feature_points->size(),mem_manager,cost_rescale_factors);
            //std::chrono::steady_clock::time_point custom_end = std::chrono::steady_clock::now();
            //std::cout << "custom mem cyclic took: = " << std::chrono::duration_cast<std::chrono::microseconds>(custom_end - custom_begin).count() << "[�s]" << "  default mem cyclic took: = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[�s]" << std::endl;
            //std::cout << best_icp->total_cost << " " << best_icp_cus_mem.total_cost << std::endl;

            for(int i = 0; i < best_icp->instance->size();i++){
                (*(best_icp->instance))[i] = MEM_ELEM_ACCESS(best_icp_cus_mem.instance_mem_elem,int,i);
                best_icp->total_cost = best_icp_cus_mem.total_cost;
            }


            free(best_mem_elem.memory_pointer);
        }




    }

   
    //std::cout << feature_points->size() << " to " << candidates->size() << " cyclic greedy search solution cost: " << best_icp->total_cost << std::endl;

}

void list_subinstances_for_next_decision_and_push_to_queue(     std::priority_queue<Instance_Cost_Pair,std::vector<Instance_Cost_Pair>,Queue_Cost_Compare>* instance_queue,
                                                                std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates,
                                                                uint16_t decision_row , uint16_t num_previous_assignments, Instance_Cost_Pair* best_icp, 
                                                                std::vector<int>* instance_vector, int* num_created_instances, Cost_Rescale_Factors cost_rescale_factors){

    int num_candidates = candidates->size();

    std::vector<int> local_instance_vector = *instance_vector;

    for(int col = -1; col < num_candidates; col++){
        if(!check_if_instance_vector_column_is_occupied(local_instance_vector,col)){
            local_instance_vector[decision_row] = col;

            Instance_Cost_Pair new_icp;

            new_icp.instance = &local_instance_vector;
            new_icp.next_row_to_make_decision = decision_row + 1;
            new_icp.total_num_assignments = num_previous_assignments;

            if(local_instance_vector[decision_row] != -1){
                new_icp.total_num_assignments++;
            }

            new_icp.cost = get_cost_for_single_problem_instance(&local_instance_vector,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
            new_icp.heuristic_cost_for_remaining_decision = get_heuristic_cost_for_remaining_decisions(new_icp,feature_points->size(),cost_rescale_factors);
            new_icp.total_cost = new_icp.cost + new_icp.heuristic_cost_for_remaining_decision;

            if(new_icp.total_cost < best_icp->total_cost){

                new_icp.instance = new std::vector<int>;
                (*num_created_instances)++;
                *new_icp.instance = local_instance_vector;

                instance_queue->push(new_icp);

            }
            
        }        
    }

}

std::vector<int> find_initial_solution_by_rotating_and_greedy_distance_matching(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, int& best_angle, Cost_Rescale_Factors cost_rescale_factors){

    std::vector<int> best_solution;
    double cost_of_best_solution = 0;


    std::vector<Polar_Coords_and_ID> local_features; 
    std::vector<Polar_Coords_and_ID> local_candidates;

    for(int angle_offset = 0; angle_offset < 360; angle_offset += 1){

    std::vector<int> current_solution;

    for(int i = 0; i < feature_points->size();i++){

        current_solution.push_back(-1);
    }

    fill_polar_coords_and_id_vec_from_feature_point_vec(local_features,*feature_points, 0.0);
    fill_polar_coords_and_id_vec_from_feature_point_vec(local_candidates, *candidates, angle_offset);

        while(local_features.size() > 0){

            int index_of_current_best_feature_point = -1;
            int index_of_current_best_candidate = -1;

            int index_in_local_features = -1;
            int index_in_local_candidates = -1;

            float shortest_distance = FLT_MAX;

            for(int i = 0; i < local_features.size();i++){
                //std::cout << "i: " << i << std::endl;
                for(int j = 0; j < local_candidates.size();j++){

                    if(local_features[i].channel != local_candidates[j].channel){
                        continue;
                    }

                    float current_distance = get_distance_between_two_points_polar(local_features[i],local_candidates[j]);



                    if(current_distance < shortest_distance){
                        shortest_distance = current_distance;

                        index_of_current_best_feature_point = local_features[i].id;
                        index_of_current_best_candidate = local_candidates[j].id;

                        index_in_local_features = i;
                        index_in_local_candidates = j;
                    }
                }
            }
            //std::cout << "end of loop" << std::endl;

            if(index_in_local_features == -1 || index_in_local_candidates == -1 || index_of_current_best_candidate == -1 || index_of_current_best_feature_point == -1){
                break;
            }

            //std::cout << index_in_local_features << " " << index_in_local_candidates << " " << index_of_current_best_candidate << " " << index_of_current_best_feature_point << std::endl;

            current_solution[index_of_current_best_feature_point] = index_of_current_best_candidate;

            local_features.erase(local_features.begin() + index_in_local_features);
            local_candidates.erase(local_candidates.begin() + index_in_local_candidates);

            double cost_of_current_instance = get_cost_for_single_problem_instance(&current_solution,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);

            if(cost_of_current_instance < cost_of_best_solution){
                cost_of_best_solution = cost_of_current_instance;
                best_angle = angle_offset;
                best_solution = current_solution;

            }

        }

    }
 
    BaB_DEBUG_PRINT(
        std::cout << "Cost of best greedy rotate and find closest solution: " << get_cost_for_single_problem_instance(&best_solution,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors) << std::endl;
        std::cout << "found best greedy dist solution at: " << best_angle << " with cost: " << cost_of_best_solution << " " << feature_points->size() << " " << candidates->size() << std::endl;
        
    )

    return best_solution;
}

std::vector<std::vector<int>*>* create_neighborhood_lists_for_candidates(std::vector<Polar_Coords_and_ID>& local_candidates, int num_neighbors_per_candidate){
    std::vector<std::vector<int>*>* neighborhood_lists = new std::vector<std::vector<int>*>;

    if(num_neighbors_per_candidate <= 1){
        num_neighbors_per_candidate = 2;
    }

    std::vector<ID_and_Dist> tmp_ids_and_dists;

    for(int i = 0; i < local_candidates.size();i++){

        std::vector<int>* new_neighbor_list = new std::vector<int>;

        tmp_ids_and_dists.clear();

        Polar_Coords_and_ID current_candidate = local_candidates[i];

        for(int j = 0; j < local_candidates.size();j++){
            if(i == j){
                continue;
            }

            Polar_Coords_and_ID current_neighbor = local_candidates[j];

            if(current_candidate.channel != current_neighbor.channel){
                continue;

                //we use this list to check from a previous assignment the neighboring points to find a new assignment.
                // a valid assignment only has those in which the feature and the candidate have the same channel
                // hence we know that if we had assigned a feature to the 'current canidate' then we only need those 'current neighbors' that have the same channel
            }

            float dist = get_distance_between_two_points_polar(current_candidate,current_neighbor);

            ID_and_Dist id_and_dist_of_neighbor;
            id_and_dist_of_neighbor.dist_to_neighbor = dist;
            id_and_dist_of_neighbor.neighbor_id = current_neighbor.id;

            //std::cout << dist << " " << current_neighbor.id << " "; 

            if(tmp_ids_and_dists.size() == 0){
                tmp_ids_and_dists.push_back(id_and_dist_of_neighbor);
                continue;
            }

            int max_needed_iterations = num_neighbors_per_candidate;
            if(tmp_ids_and_dists.size() < max_needed_iterations){
                max_needed_iterations = tmp_ids_and_dists.size();
            }

            bool inserted_element = false;

            for(int k = 0; k < max_needed_iterations;k++){
                if(id_and_dist_of_neighbor.dist_to_neighbor < tmp_ids_and_dists[k].dist_to_neighbor){
                    tmp_ids_and_dists.emplace(tmp_ids_and_dists.begin() + k, id_and_dist_of_neighbor);
                    inserted_element = true;
                    break;
                }
            }

            if(!inserted_element && (tmp_ids_and_dists.size() < num_neighbors_per_candidate)){
                tmp_ids_and_dists.push_back(id_and_dist_of_neighbor);
            }
        }
        //std::cout << std::endl;



        for(int j = 0; j < std::min<int>(tmp_ids_and_dists.size(),num_neighbors_per_candidate);j++){
            new_neighbor_list->push_back(tmp_ids_and_dists[j].neighbor_id);
        }

        new_neighbor_list->push_back(current_candidate.id);

        neighborhood_lists->push_back(new_neighbor_list);
    }

    return neighborhood_lists;
}

std::vector<int> find_initial_solution_by_rotating_and_greedy_distance_matching_neighborhood_cone(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, int& best_angle, Cost_Rescale_Factors cost_rescale_factors){

    std::vector<int> best_solution;
    double cost_of_best_solution = 0;

    std::vector<int>* previous_solution = new std::vector<int>;
    std::vector<int>* current_solution = new std::vector<int>;

    for(int i = 0; i < feature_points->size();i++){
        current_solution->push_back(-1);
        previous_solution->push_back(-1);
    }

    std::vector<Polar_Coords_and_ID> local_features; 
    std::vector<Polar_Coords_and_ID> local_candidates;
    fill_polar_coords_and_id_vec_from_feature_point_vec(local_candidates, *candidates, 0.0);

    std::vector<std::vector<int>*>* neighborhood_lists = create_neighborhood_lists_for_candidates(local_candidates,local_candidates.size() * ANGLE_RANGE_SEARCH_NEIGHBORHOOD_LIST_SIZE);


    std::vector<int> indices_into_local_features;
    std::vector<int> indices_into_local_candidates;
    std::vector<int> local_candidates_validity;

    local_candidates_validity.resize(local_candidates.size());

    fill_polar_coords_and_id_vec_from_feature_point_vec(local_features,*feature_points, 0.0);


    for(int angle_offset = 0; angle_offset < 360; angle_offset += ANGLE_RANGE_SEARCH_INCREMENT){
        
        indices_into_local_features.clear();
        indices_into_local_candidates.clear();

        for(int i = 0; i < feature_points->size();i++){
            (*current_solution)[i] = -1;
            indices_into_local_features.push_back(i);
        }

        fill_polar_coords_and_id_vec_from_feature_point_vec(local_candidates, *candidates, angle_offset,&indices_into_local_candidates,&local_candidates_validity);

        while(indices_into_local_features.size() > 0){

            int index_of_current_best_feature_point = -1;
            int index_of_current_best_candidate = -1;

            int offset_in_features_index_vector = -1;
            int offset_in_candidates_index_vector = -1;

            float shortest_distance = FLT_MAX;

            for(int i = 0; i < indices_into_local_features.size();i++){
                //std::cout << "i: " << i << std::endl;

                Polar_Coords_and_ID current_feature = local_features[indices_into_local_features[i]];

                int previous_candidate_id = (*previous_solution)[current_feature.id];

                if(previous_candidate_id != -1){

                    std::vector<int>* candidate_neighborhood = (*neighborhood_lists)[previous_candidate_id];

                    for(int j = 0; j < candidate_neighborhood->size();j++){
                        int neighboring_candidate_index = (*candidate_neighborhood)[j];

                        Polar_Coords_and_ID neighbor = local_candidates[neighboring_candidate_index];

                        if(current_feature.channel != neighbor.channel){
                            continue;
                        }

                        if(local_candidates_validity[neighboring_candidate_index] == 1){

                            float current_distance = get_distance_between_two_points_polar(current_feature,neighbor);

                            if(current_distance < shortest_distance){
                                shortest_distance = current_distance;

                                index_of_current_best_feature_point = current_feature.id;
                                index_of_current_best_candidate = neighboring_candidate_index;

                                offset_in_features_index_vector = i;
                                offset_in_candidates_index_vector = neighboring_candidate_index;
                            }
                        }
                    }
                    
                }else{
                    for(int j = 0; j < indices_into_local_candidates.size();j++){

                        Polar_Coords_and_ID current_candidate = local_candidates[indices_into_local_candidates[j]];

                        if(current_feature.channel != current_candidate.channel){
                            continue;
                        }

                        float current_distance = get_distance_between_two_points_polar(current_feature,current_candidate);

                        if(current_distance < shortest_distance){
                            shortest_distance = current_distance;

                            index_of_current_best_feature_point = current_feature.id;
                            index_of_current_best_candidate = current_candidate.id;

                            offset_in_features_index_vector = i;
                            offset_in_candidates_index_vector = current_candidate.id;
                        }
                    }
                }

            }
            //std::cout << "end of loop" << std::endl;

            if(offset_in_features_index_vector == -1 || offset_in_candidates_index_vector == -1 || index_of_current_best_candidate == -1 || index_of_current_best_feature_point == -1){
                break;
            }

            //std::cout << offset_in_features_index_vector << " " << offset_in_candidates_index_vector << " " << index_of_current_best_candidate << " " << index_of_current_best_feature_point << std::endl;

            (*current_solution)[index_of_current_best_feature_point] = index_of_current_best_candidate;

            indices_into_local_features.erase(indices_into_local_features.begin() + offset_in_features_index_vector);
            //indices_into_local_candidates.erase(indices_into_local_candidates.begin() + offset_in_candidates_index_vector);


            /*
            int id_to_erase = get_index_of_element_in_ordered_vector(indices_into_local_candidates,offset_in_candidates_index_vector);
            if(id_to_erase > 0){
                indices_into_local_candidates.erase(indices_into_local_candidates.begin() + id_to_erase);
            }
            */
            for(int k = 0; k < indices_into_local_candidates.size(); k++){
                if(indices_into_local_candidates[k] == offset_in_candidates_index_vector){
                    indices_into_local_candidates.erase(indices_into_local_candidates.begin() + k);
                    break;
                }
            }
            
            
            local_candidates_validity[offset_in_candidates_index_vector] = 0;


            
            double cost_of_current_instance = get_cost_for_single_problem_instance(current_solution,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);

            if(cost_of_current_instance < cost_of_best_solution){
                cost_of_best_solution = cost_of_current_instance;
                best_angle = angle_offset;
                best_solution = *current_solution;

            }

        }

        std::vector<int>* tmp = previous_solution;
        previous_solution = current_solution;
        current_solution = tmp;

    }

    for(int i = 0; i < neighborhood_lists->size();i++){
        delete((*neighborhood_lists)[i]);
    }

    delete(neighborhood_lists);

    delete(current_solution);
    delete(previous_solution);
 
    BaB_DEBUG_PRINT(
        std::cout << "Cost of best greedy rotate and find closest solution: " << get_cost_for_single_problem_instance(&best_solution,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors) << std::endl;
        std::cout << "found best greedy dist neighborhood solution at: " << best_angle << " with cost: " << cost_of_best_solution << " " << feature_points->size() << " " << candidates->size() << std::endl;
    )
    //std::cout << "end of function" << std::endl;

    return best_solution;

}

void calculate_greedy_cost_matching_for_single_angle(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates,int angle, std::vector<int>& best_solution, double& cost_of_best_solution, Cost_Rescale_Factors cost_rescale_factors){
    std::vector<int> current_solution;

    std::vector<Polar_Coords_and_ID> local_features; 
    std::vector<Polar_Coords_and_ID> local_candidates;

    //std::priority_queue<Vector_IDs_and_Dist,std::vector<Vector_IDs_and_Dist>,Vector_IDs_and_Dist_Compare> local_distance_queue;

    std::vector<Vector_IDs_and_Dist> best_vec_ids_and_dists;
    int max_length_of_best_vec_ids_dist = 10;

    float distance_threshold = 0.2f;

    for(int i = 0; i < feature_points->size();i++){

        current_solution.push_back(-1);
    }

    fill_polar_coords_and_id_vec_from_feature_point_vec(local_features,*feature_points, 0.0);
    fill_polar_coords_and_id_vec_from_feature_point_vec(local_candidates, *candidates, angle);

    while(local_features.size() > 0){


        int index_of_current_best_feature_point = -1;
        int index_of_current_best_candidate = -1;

        int index_in_local_features = -1;
        int index_in_local_candidates = -1;

        float shortest_distance = FLT_MAX;
        float best_cost = FLT_MAX;

        best_vec_ids_and_dists.clear();

        for(int i = 0; i < local_features.size();i++){
            for(int j = 0; j < local_candidates.size();j++){

                if(local_features[i].channel != local_candidates[j].channel){
                    continue;
                }

                Vector_IDs_and_Dist new_vec_id_and_dist;

                new_vec_id_and_dist.index_in_local_candidates = j;
                new_vec_id_and_dist.index_in_local_features = i;

                new_vec_id_and_dist.index_of_current_best_candidate = local_candidates[j].id;
                new_vec_id_and_dist.index_of_current_best_feature_point = local_features[i].id;

                new_vec_id_and_dist.distance = get_distance_between_two_points_polar(local_features[i],local_candidates[j]);

                new_vec_id_and_dist.cost = calculate_cost_for_two_feature_points((*feature_points)[local_features[i].id],(*candidates)[local_candidates[j].id],cost_rescale_factors);

                
                check_and_sort_into_ids_and_dist_vector(best_vec_ids_and_dists,new_vec_id_and_dist,max_length_of_best_vec_ids_dist);
                //local_distance_queue.push(new_vec_id_and_dist);
            }
        }

        shortest_distance = best_vec_ids_and_dists[0].distance;

        for(int i = 0; i < best_vec_ids_and_dists.size();i++){
            if((best_vec_ids_and_dists[i].distance / shortest_distance) - 1.0f < distance_threshold){ 
                if(best_cost > best_vec_ids_and_dists[i].cost){

                best_cost = best_vec_ids_and_dists[i].cost;
                index_of_current_best_feature_point = best_vec_ids_and_dists[i].index_of_current_best_feature_point;
                index_of_current_best_candidate = best_vec_ids_and_dists[i].index_of_current_best_candidate;

                index_in_local_features = best_vec_ids_and_dists[i].index_in_local_features;
                index_in_local_candidates = best_vec_ids_and_dists[i].index_in_local_candidates;
                }

            }else{
                break;
            }
        }

        if(index_in_local_features == -1 || index_in_local_candidates == -1 || index_of_current_best_candidate == -1 || index_of_current_best_feature_point == -1){
            break;
        }

        current_solution[index_of_current_best_feature_point] = index_of_current_best_candidate;

        local_features.erase(local_features.begin() + index_in_local_features);
        local_candidates.erase(local_candidates.begin() + index_in_local_candidates);

        double cost_of_current_instance = get_cost_for_single_problem_instance(&current_solution,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);

        if(cost_of_current_instance < cost_of_best_solution){
            cost_of_best_solution = cost_of_current_instance;
            best_solution = current_solution;

        }

    }
}

std::vector<int> find_initial_solution_by_rotating_and_greedy_cost_matching(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors){

    std::vector<int> best_solution;
    double cost_of_best_solution = 0;

    for(int angle_offset = 0; angle_offset < 360; angle_offset++){

        calculate_greedy_cost_matching_for_single_angle(feature_points,candidates,angle_offset,best_solution,cost_of_best_solution,cost_rescale_factors);

    }
 
    BaB_DEBUG_PRINT(
        std::cout << "Cost of best greedy rotate and find best cost solution: " << get_cost_for_single_problem_instance(&best_solution,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors) << std::endl;
    )

    return best_solution;
}


std::vector<int> find_initial_solution_by_rotating_and_greedy_cost_matching_in_angle_range(std::vector<Feature_Point_Data>* feature_points, std::vector<Feature_Point_Data>* candidates, int start_angle, int angle_range, Cost_Rescale_Factors cost_rescale_factors){

    std::vector<int> best_solution;
    double cost_of_best_solution = 0;

    int first_intervall_start = 0;
    int first_intervall_end = 0;

    int second_intervall_start = 0;
    int second_intervall_end = 0;

    first_intervall_start = ((start_angle - angle_range + 360) % 360);
    first_intervall_end = ((start_angle + angle_range + 360) % 360);

    bool overflow_on_zero = first_intervall_start > start_angle;
    bool overflow_on_end = first_intervall_end < start_angle;

    if(overflow_on_zero){
        second_intervall_start = first_intervall_start;
        second_intervall_end = 360;

        first_intervall_start = 0;
        first_intervall_end = start_angle + angle_range; 
    }else if(overflow_on_end){
        first_intervall_start = start_angle - angle_range;
        first_intervall_end = 360;

        second_intervall_start = 0;
        second_intervall_end = first_intervall_end;
    }

    for(int angle_offset = first_intervall_start; angle_offset < first_intervall_end; angle_offset++){

        calculate_greedy_cost_matching_for_single_angle(feature_points,candidates,angle_offset,best_solution,cost_of_best_solution,cost_rescale_factors);

    }

    for(int angle_offset = second_intervall_start; angle_offset < second_intervall_end; angle_offset++){

        calculate_greedy_cost_matching_for_single_angle(feature_points,candidates,angle_offset,best_solution,cost_of_best_solution,cost_rescale_factors);

    }
 
    BaB_DEBUG_PRINT(
        std::cout << "Cost of best greedy rotate and find best cost solution: " << get_cost_for_single_problem_instance(&best_solution,feature_points,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors) << std::endl;
        std::cout << "finished greedy cost matching " << cost_of_best_solution << std::endl;
    )


    return best_solution;
}

void check_and_sort_into_ids_and_dist_vector(std::vector<Vector_IDs_and_Dist>& vec_ids_and_dists, Vector_IDs_and_Dist& new_element, int max_num_elements){

    if(vec_ids_and_dists.size() == 0){
        vec_ids_and_dists.push_back(new_element);
        return;
    }

    if(vec_ids_and_dists.size() < max_num_elements){
        int index_to_insert_at = 0;

        for(int i = 0; i < vec_ids_and_dists.size();i++){
            if(new_element.distance < vec_ids_and_dists[i].distance){
                break;
            }
            index_to_insert_at++;
        }

        vec_ids_and_dists.insert(vec_ids_and_dists.begin() + index_to_insert_at, 1,new_element);

        //std::cout << "new size: " << vec_ids_and_dists.size() << "after insert at: " << index_to_insert_at << std::endl; 

    }else{
        if(new_element.distance > vec_ids_and_dists[vec_ids_and_dists.size() - 1].distance){
            return;
        }else{
            int index_to_insert_at = 0;

            for(int i = 0; i < vec_ids_and_dists.size();i++){
                if(new_element.distance < vec_ids_and_dists[i].distance){
                    break;
                }
                index_to_insert_at++;
            }

            vec_ids_and_dists.insert(vec_ids_and_dists.begin() + index_to_insert_at, 1,new_element);
            vec_ids_and_dists.pop_back();
            
        }
    }

}

float get_distance_between_two_feature_points(Feature_Point_Data& feature_point_1,Feature_Point_Data& feature_point_2){

    // we can optimize this function by in the context of the find_initial_solution_by_rotating_and_greedy_matching by calculating the squared radius once 
    // as well as the sin and cos of those feature points that dont get rotated
    // we might also be able to use a look up table for the sin and cos values
    float radius_1_sqr = feature_point_1.relative_distance_center_max_distance * feature_point_1.relative_distance_center_max_distance;

    float radius_2_sqr = feature_point_2.relative_distance_center_max_distance * feature_point_2.relative_distance_center_max_distance;

    float r1_r2 = feature_point_1.relative_distance_center_max_distance * feature_point_2.relative_distance_center_max_distance;

    float alpha = (feature_point_1.angle / 180.0) * PI;
    float beta = (feature_point_2.angle / 180.0) * PI;
    
    float angle_term = sin(alpha) * sin(beta) + cos(alpha) * cos(beta);

    return sqrtf(radius_1_sqr + radius_2_sqr + -2 * r1_r2 * angle_term);

}

float get_distance_between_two_points_polar(Polar_Coords_and_ID& point_1,Polar_Coords_and_ID& point_2){

    float r1_r2 = point_1.radius * point_2.radius;

    float angle_term = point_1.sin_angle * point_2.sin_angle + point_1.cos_angle * point_2.cos_angle;

    return sqrtf(point_1.radius_squared + point_2.radius_squared + -2 * r1_r2 * angle_term);
}

void fill_polar_coords_and_id_vec_from_feature_point_vec(std::vector<Polar_Coords_and_ID>& pc_vec, std::vector<Feature_Point_Data>& features, float angle_offset, std::vector<int>* index_vector, std::vector<int>* validity_vector){

    pc_vec.clear();

    for(int i = 0; i < features.size();i++){
        Polar_Coords_and_ID new_pc;

        float angle_degrees = fmod((features[i].angle + angle_offset),360.0f);

        new_pc.angle_radians = (angle_degrees / 180.0f) * PI;
        new_pc.radius = features[i].relative_distance_center_max_distance;
        new_pc.radius_squared = features[i].relative_distance_center_max_distance * features[i].relative_distance_center_max_distance;

        new_pc.channel = features[i].channel;

        new_pc.color_value = features[i].normalize_peak_value;

        new_pc.sin_angle = sinf(new_pc.angle_radians);
        new_pc.cos_angle = cosf(new_pc.angle_radians);

        new_pc.id = i;

        if(index_vector != nullptr){
            index_vector->push_back(i);
        }

        if(validity_vector != nullptr){
            (*validity_vector)[i] = 1;
        }

        pc_vec.push_back(new_pc);
    }

}