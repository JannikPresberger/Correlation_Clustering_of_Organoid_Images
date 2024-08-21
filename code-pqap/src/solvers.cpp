#include "solvers.h"
#include "clustering.h"

#include "task_handler.h"
#include "python_handler.h"
#include "mpi_handler.h"

#include <chrono>
#include <thread>
#include <algorithm>
#include <fstream>

#define VISUALIZE_INDIVIDUAL_ASSIGNMENTS true

#define SOLVE_TO_OPTIMALITY_USING_GUROBI false

#define USE_QAP_SOLVER false

#define PARALLELIZE_MATCHING_CALCULATION true

#define PRINT_SOLVER_PROGRESS false

#define PRINT_GUROBI_LOGS 0
#define GUROBI_TIMELIMIT_SECONDS 60

#define RUNTIME_ESTIMATION_VAR_A 0.497
#define RUNTIME_ESTIMATION_VAR_B -4.207

#define POS_DIST_COST_STEP_FLAG 1
#define NEG_DIST_COST_STEP_FLAG 2

#define POS_COLOR_COST_STEP_FLAG 4
#define NEG_COLOR_COST_STEP_FLAG 8

#define POS_ANGLE_COST_STEP_FLAG 16
#define NEG_ANGLE_COST_STEP_FLAG 32

#define POS_UNARY_OFFSET_STEP_FLAG 64
#define NEG_UNARY_OFFSET_STEP_FLAG 128

#define POS_QUADR_OFFSET_STEP_FLAG 256
#define NEG_QUADR_OFFSET_STEP_FLAG 512

#define MODEL_PARAMETER_OFFSET_LOWER_BOUND 0.01
#define MODEL_PARAMETER_OFFSET_UPPER_BOUND 0.30

#define MODEL_PARAMETER_WEIGHT_LOWER_BOUND 0.10
#define MODEL_PARAMETER_WEIGHT_UPPER_BOUND 0.90

const int viz_threshold_slider_max = 1000;
int viz_threshold_slider = 500;
double viz_threshold;

int microsec_trackbar_cooldown = 10000;
std::chrono::steady_clock::time_point last_trackbar_call_timestamp = std::chrono::steady_clock::now();
//std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();


typedef struct Matching_Matrix_Input_Struct{
    std::vector<cv::Mat*>* square_organoid_images;
    std::vector<Matching_Result>* all_matching_results;
    std::vector<Image_Features_Pair>* all_feature_vectors;
    int individual_image_size;
    Matching_Visualization_Type viz_type;
    uint32_t flags;
    std::vector<int>* image_order;
    bool use_image_order;
    All_Cost_Parameters cost_params;
}Matching_Matrix_Input_Struct;

struct Matching_Calculation_Task_Compare{

    bool operator()(const Matching_Calculation_Task& lhs, const Matching_Calculation_Task& rhs){

        return lhs.runtime_estimation < rhs.runtime_estimation;
    }
};

struct Task_Vector_Compare{

    bool operator()(const Task_Vector& lhs, const Task_Vector& rhs){

        return lhs.total_runtime_estimation < rhs.total_runtime_estimation;
    }
}Task_Vector_Compare;

struct TPR_FPR_Tuple_Ordering_Compare{
    bool operator()(const TPR_FPR_Tuple& lhs, const TPR_FPR_Tuple& rhs){

        if(lhs.fpr == rhs.fpr){
            return lhs.tpr < rhs.tpr;
        }
        return lhs.fpr < rhs.fpr;
    }
}TPR_FPR_Tuple_Ordering_Compare;

struct Prec_Ordering_Compare{
    bool operator()(const TPR_FPR_Tuple& lhs, const TPR_FPR_Tuple& rhs){

        return lhs.threshold < rhs.threshold;
    }
}Prec_Ordering_Compare;

enum Extraction_By_Image_Set_Number_Mode{
    BOTH_MATCH,
    ONLY_ONE_MATCHES,
    ATLEAST_ONE_MATCHES
};

typedef struct Line_Search_Data_Struct{
    Search_Dim current_search_dim;
    int current_search_dir;
    bool dir_is_set;
    All_Cost_Parameters start_parameters;
    All_Cost_Parameters pos_search_dir_candidate;
    All_Cost_Parameters neg_search_dir_candidate;
}Line_Search_Data_Struct;

typedef struct Exhaustive_Adj_Search_Data_Struct{
    bool currently_search_for_best_adj;
    bool new_best_was_found;
}Exhaustive_Adj_Search_Data_Struct;

typedef struct Sim_Annealing_Search_Data_Struct{
    double temperature;
    double cooling_rate;
    double std_dev;
    double prev_metric;
    All_Cost_Parameters prev_accepted_parameters;
    double temp_at_last_best;
    double metric_of_best;
    All_Cost_Parameters last_best_parameters;
    int num_time_steps_since_last_best;
    int restart_time_step_threshold;
}Sim_Annealing_Search_Data_Struct;

typedef struct Metric_and_Threshold{
    double metric;
    double threshold;
}Metric_and_Threshold;

template<>
struct std::hash<All_Cost_Parameters>
{
    std::size_t operator()(const All_Cost_Parameters& key) const noexcept
    {

        size_t hash_key = std::hash<double>{}(key.color_offset);//-key.static_cost_offset;
        hash_key += std::hash<double>{}(key.dist_offset);//-key.static_pairwise_cost_offset * 100;
        hash_key += std::hash<double>{}(key.angle_offset);//key.angle_difference_cost_coefficient * 10000;
        hash_key += std::hash<double>{}(key.color_to_dist_weight);//key.color_difference_cost_coefficient * 1000000;
        hash_key += std::hash<double>{}(key.unary_to_to_quadr_weight);//key.distance_difference_cost_coefficient * 100000000;

        return std::hash<size_t>{}(hash_key);
    }

};

bool operator==(const All_Cost_Parameters& lhs, const All_Cost_Parameters& rhs){
    bool angle_match = lhs.color_offset == rhs.color_offset;
    bool dist_match = lhs.dist_offset == rhs.dist_offset;
    bool color_match = lhs.angle_offset == rhs.angle_offset;
    bool unary_match = lhs.color_to_dist_weight == rhs.color_to_dist_weight;
    bool quadr_match = lhs.unary_to_to_quadr_weight == rhs.unary_to_to_quadr_weight;

    return angle_match && dist_match && color_match && unary_match && quadr_match;/* your comparison code goes here */
}

void* get_initial_search_data_struct_by_search_strat(Model_Parameter_Selection_Strategy search_strat,const Input_Arguments& args);

Search_Dim get_next_dim_in_search_order(std::vector<Search_Dim>& search_order,int& current_search_order_index);

void finalize_output(std::ofstream& log_file, All_Cost_Parameters& best_parameters, Metric_and_Threshold& best_metric_and_threshold);

void output_results(std::ofstream& log_file, Model_Parameter_Selection_Strategy search_strat ,Learning_Metric_Types lm_type ,std::vector<Matching_Result>* matching_results, All_Cost_Parameters& cost_params, std::vector<Image_Features_Pair>& all_feature_vectors, std::string feature_vector_file_name, bool features_were_read_from_file, const Input_Arguments args, bool metric_is_new_best, std::string& filename_of_best_result, Metric_and_Threshold current_metric_and_threshold);

bool check_loop_termination_condition_and_distribute_cost_parameters_from_queue(int process_id, std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters* current_parameters, int& iteration_counter, int max_num_iterations, double& current_runtime, double runtime_of_last_mm_calc, double time_limit, bool reference_clustering_was_valid, std::chrono::steady_clock::time_point& loop_begin);

float get_initial_value_by_metric_type(Learning_Metric_Types lm_type);

float extract_best_threshold_and_metric_value_from_confusion_matrix(std::vector<Matching_Result>& all_matching_results, Learning_Metric_Types lm_type, double* best_threshold, const std::vector<Cluster> &clustering);

void reset_line_search_data_struct(Line_Search_Data_Struct* data_struct);

void start_line_search_in_next_dim(Line_Search_Data_Struct* data, All_Cost_Parameters current_parameters, double step_width, std::queue<All_Cost_Parameters>& cost_parameter_queue, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, std::vector<Search_Dim>& search_order, int& current_search_order_index);

void set_new_start_and_search_dim_in_line_search_data_struct(Line_Search_Data_Struct* data_struct, All_Cost_Parameters& new_start, Search_Dim prev_search_dim, std::vector<Search_Dim>& search_order, int& current_search_order_index);

bool is_metric_a_better_then_b(double metric_a, double metric_b, Learning_Metric_Types metric_type);

void add_step_and_bound_model_parameter(double& parameter, double step, double lower_bound, double upper_bound);

void add_both_line_search_dirs(Line_Search_Data_Struct* data, All_Cost_Parameters current_parameters, double step_width, std::queue<All_Cost_Parameters>& cost_parameter_queue, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map);

All_Cost_Parameters get_new_cost_parameters(All_Cost_Parameters old_cost_parameters, All_Cost_Parameters cost_steps);

bool add_parameters_to_queue_and_map(All_Cost_Parameters new_cost_parameters, All_Cost_Parameters prev_cost_parameter, std::queue<All_Cost_Parameters>& cost_parameter_queue, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map);

bool check_if_parameters_are_present_in_map(const All_Cost_Parameters& cost_parameters, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map);

void choose_next_parameters(std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters& current_parameters, Model_Parameter_Selection_Strategy sel_strat, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, double step_width, void* additional_data, Learning_Metric_Types lm_type, double prev_metric_val , double current_metric_val, std::vector<Search_Dim>& search_order,int& current_search_order_index,All_Cost_Parameters* best_parameters, double metric_of_best_parameters);

void choose_next_parameters_using_line_search(std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters& current_parameters, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, double step_width, void* additional_data, Learning_Metric_Types lm_type, bool current_was_better_then_prev, std::vector<Search_Dim>& search_order, int& current_search_order_index);

void choose_next_parameter_using_exhaustive_adj_search(std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters& current_parameters, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, double step_width, void* additional_data, Learning_Metric_Types lm_type, bool current_was_better_then_prev, All_Cost_Parameters* best_parameters);

void choose_next_parameter_using_simulated_annealing(std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters& current_parameters, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, double step_width, void* additional_data, Learning_Metric_Types lm_type, bool current_was_better_then_prev, All_Cost_Parameters* best_parameters,double current_metric_value, double metric_of_best_parameters);

double evaluate_learning_metric(std::vector<Matching_Result>& all_matching_results, Learning_Metric_Types lm_type, const std::vector<Cluster> &clustering, double* threshold_corresponding_to_best_metric = nullptr);

Metric_and_Threshold add_new_cost_parameters_to_queue(std::queue<All_Cost_Parameters>& cost_parameter_queue, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, All_Cost_Parameters& current_cost_parameters, std::vector<Matching_Result>& local_matching_results, std::vector<Matching_Result>* best_matching_results, double* metric_of_best, double* threshold_of_best, All_Cost_Parameters* best_parameters, Model_Parameter_Selection_Strategy search_strat, Learning_Metric_Types learning_metric_type, void* additional_search_data, bool& new_metric_is_best, const std::vector<Cluster> &clustering,std::vector<Search_Dim>& search_order, int& current_search_order_index, float step_width);

void calculate_matching_matrix(int process_id, int total_num_processes, All_Cost_Parameters& cost_params, Assignment_Visualization_Data& viz_data, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Matching_Result>& all_matching_results, std::vector<Op_Numbers_And_Runtime>& execution_times, Additional_Viz_Data_Input* viz_input, const Input_Arguments& args, int iteration_counter);

void subdivide_task_queue_into_task_vectors(std::priority_queue<Matching_Calculation_Task,std::vector<Matching_Calculation_Task>,Matching_Calculation_Task_Compare>& task_queue, std::vector<Task_Vector>& all_task_vectors, int num_task_vectors_to_subdivide_into);

void fill_task_queue_with_all_tasks(std::priority_queue<Matching_Calculation_Task,std::vector<Matching_Calculation_Task>,Matching_Calculation_Task_Compare>& task_queue, std::vector<Image_Features_Pair>& all_feature_vectors, Assignment_Visualization_Data* viz_data, All_Cost_Parameters* cost_params);

void process_task_vector(int thread_id, Task_Vector* task_vector, std::vector<Matching_Result>* partial_matching_results, std::vector<Image_Features_Pair>* all_feature_vectors);

void process_task_queue_from_task_handler(int thread_id, Task_Handler* task_handler, std::vector<Matching_Result>* partial_matching_results, std::vector<Op_Numbers_And_Runtime>* partial_execution_times, std::vector<Image_Features_Pair>* all_feature_vectors);

void free_task_vectors(std::vector<Task_Vector>& task_vectors);

void adapt_image_order_to_clusters(std::vector<int>& image_order, std::vector<Cluster>& all_clusters, std::vector<Image_Features_Pair>& all_feature_vectors);

void fill_op_number_struct(Op_Numbers_And_Runtime& op_number_struct, int num_features, int num_candidates, Algorithm_Type algo_type, double runtime, double result);

int extract_all_images_from_folder(std::filesystem::path image_folder_path, const Input_Arguments args, std::filesystem::path feature_vector_file_path,    
                                                                                std::vector<cv::Mat*>& square_organoid_images, 
                                                                                std::vector<Image_Features_Pair>& all_feature_vectors,
                                                                                std::vector<int>& image_number_index,
                                                                                std::vector<cv::Point>& image_centers,
                                                                                std::vector<cv::Point>& offsets_from_squaring,
                                                                                std::vector<std::filesystem::path>& image_paths,
                                                                                int set_number = 1, int img_number_offset = 0);

void read_image_numbers_and_feature_vectors_from_file(std::filesystem::path file_path,std::vector<Image_Features_Pair>& all_feature_vectors, int set_number);

void set_image_set_numbers_in_matching_results(std::vector<Matching_Result>& all_matching_results, std::filesystem::path image_base_folder, const Input_Arguments& input_args, int img_num_offset_set_2);

void select_cluster_representatives_and_update_matching_results(std::vector<Matching_Result>& all_matching_results, std::vector<Cluster>& all_current_clusters, std::vector<Matching_Result>& new_matching_results, std::vector<Cluster_Representative_Pair>& selected_cluster_representatives);

static void on_trackbar(int, void* user_data)
{
    std::chrono::steady_clock::time_point current_timestamp = std::chrono::steady_clock::now();

    if(std::chrono::duration_cast<std::chrono::microseconds>(current_timestamp - last_trackbar_call_timestamp).count() > microsec_trackbar_cooldown){
        last_trackbar_call_timestamp = current_timestamp;

        viz_threshold = (double) viz_threshold_slider/viz_threshold_slider_max ;

        //std::cout << viz_threshold << std::endl;

        Matching_Matrix_Input_Struct* input = (Matching_Matrix_Input_Struct*)user_data;

        //std::cout << "on trackbar use order: " << input->use_image_order << std::endl;

        show_matching_matrix(input->square_organoid_images,input->all_matching_results,input->all_feature_vectors,input->individual_image_size,viz_threshold,input->viz_type,input->flags, input->image_order, input->use_image_order, input->cost_params);   
    }
}

bool image_features_pair_compare(const Image_Features_Pair& lhs, const Image_Features_Pair& rhs){

    return lhs.image_number < rhs.image_number;
}


double calculate_runtime_estimation(int first_feature_vector_size, int second_feature_vector_size);

void merge_matching_results_and_convert_similarity_to_log_odds(std::vector<Matching_Result>& original_matching_result, std::vector<Matching_Result>& merged_matching_results, double threshold);

void merge_matching_results(std::vector<Matching_Result>& original_matching_result, std::vector<Matching_Result>& merged_matching_results, double threshold, bool normalize_intervall = false);

void merge_matching_results_from_ordered_results(std::vector<Matching_Result>& original_matching_result, std::vector<Matching_Result>& merged_matching_results, double threshold,int num_elems);

void extract_matching_results_by_image_set_numbers(std::vector<Matching_Result>& all_matching_results, std::vector<Matching_Result>& extracted_matching_results,std::vector<int>& set_numbers, Extraction_By_Image_Set_Number_Mode extraction_mode);

void sort_image_feature_pair_vector_by_image_number(std::vector<Image_Features_Pair>* ifp_vector){
    std::sort(ifp_vector->begin(),ifp_vector->end(),image_features_pair_compare);
}

std::vector<int>* assign_candidates_to_features_using_gurobi(std::vector<Feature_Point_Data>* features, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors, std::vector<int>* initial_solution){

    int num_features_points_in_model = features->size();
    int num_candidates_in_model = candidates->size();

    int max_num_points = std::max<int>(num_features_points_in_model,num_candidates_in_model);
    double theoretical_optimal_cost = get_optimal_cost_for_problem_size(max_num_points,cost_rescale_factors);

    //std::filesystem::path data_folder_path = get_data_folder_path();
    //data_folder_path.append("matching.dd");
    
    //std::ofstream model_file;
    //model_file.open(data_folder_path);

    //std::cout << "Theoretical optimum: " << theoretical_optimal_cost << std::endl;

    int total_num_variables = num_features_points_in_model * num_candidates_in_model;

    std::vector<int>* solution_instance = new std::vector<int>(num_features_points_in_model,-1);

    //std::cout << "start gurobi" << std::endl;

    try {

        // Create an environment
        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag,PRINT_GUROBI_LOGS);
        env.set("LogFile", "mip1.log");
        env.start();


        // Create an empty model
        GRBModel model = GRBModel(env);


        model.getEnv().set(GRB_DoubleParam_TimeLimit,GUROBI_TIMELIMIT_SECONDS);
        //model.getEnv().set(GRB_IntParam_OutputFlag,0);
        

        GRBVar* all_model_variables = (GRBVar*)malloc(sizeof(GRBVar) * total_num_variables);
        double* variable_costs = (double*)malloc(sizeof(double) * total_num_variables);

        // total_num_variables * total_num_variables gives us the full matrix of all combinations of pairs.
        // since we don't need the diagonal in that matrix since that contains only pairs of variables with themselves we subtract total_num_variables
        // finally we divide by 2 since the matrix is symmetric and we only need one side 
        int total_num_quadratic_expressions = (total_num_variables * total_num_variables - total_num_variables) / 2;

        GRBVar* quadratic_expressions_var1 = (GRBVar*)malloc(sizeof(GRBVar) * total_num_quadratic_expressions);
        GRBVar* quadratic_expressions_var2 = (GRBVar*)malloc(sizeof(GRBVar) * total_num_quadratic_expressions);
        double* quadratic_expressions_costs = (double*)malloc(sizeof(double) * total_num_quadratic_expressions);

        GRBQuadExpr model_objective;

        int num_valid_quadr_terms = 0;

        /*
        for(int i = 0; i < total_num_variables;i++){
            for(int j = i+1; j < total_num_variables;j++){

                int index_first_feature = i / num_candidates_in_model;
                int index_fist_candidate = i % num_candidates_in_model;

                int index_second_feature = j / num_candidates_in_model;
                int index_second_candidate = j % num_candidates_in_model;

                if(index_first_feature == index_second_feature || index_fist_candidate == index_second_candidate){
                    num_valid_quadr_terms++;
                }
            }
        }

        model_file << "p " << num_features_points_in_model << " " << num_candidates_in_model << " " << total_num_variables << " " << num_valid_quadr_terms << std::endl; 
        */

        for(int i = 0; i < num_features_points_in_model; i++){
            for(int j = 0; j < num_candidates_in_model; j++){

                int var_index = i * num_candidates_in_model + j;

                std::string new_variable_name = "x" + std::to_string(i+1) + std::to_string(j+1);
                //std::cout << "added new variable: " << new_variable_name << std::endl;
                all_model_variables[var_index] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, new_variable_name);

                if(initial_solution != nullptr){
                    if((*initial_solution)[i] == j){
                        all_model_variables[var_index].set(GRB_DoubleAttr_Start,1);
                    }else{
                        all_model_variables[var_index].set(GRB_DoubleAttr_Start,0);
                    }

                }
                
                variable_costs[var_index] = calculate_cost_for_two_feature_points((*features)[i],(*candidates)[j],cost_rescale_factors);

                //model_file << "a " << var_index << " " << i << " " << j << " " << variable_costs[var_index] << std::endl;

                //model_objective.addTerm(C,all_model_variables[var_index]);
                //model_objective.addTerms()
            
            }

        }

        int num_quadr_expr = 0;

        for(int i = 0; i < total_num_variables;i++){
            for(int j = i+1; j < total_num_variables;j++){

                int index_first_feature = i / num_candidates_in_model;
                int index_fist_candidate = i % num_candidates_in_model;

                int index_second_feature = j / num_candidates_in_model;
                int index_second_candidate = j % num_candidates_in_model;

                double new_cost = 0.0;

                if(index_first_feature == index_second_feature || index_fist_candidate == index_second_candidate){
                    new_cost = COST_DOUBLE_ASSIGNMENT;
                }else{
                    new_cost = calculate_cost_for_single_feature_point_pair_assignment((*features)[index_first_feature],(*candidates)[index_fist_candidate],(*features)[index_second_feature],(*candidates)[index_second_candidate],cost_rescale_factors);
                    //model_file << "e " << index_first_feature * num_candidates_in_model + index_fist_candidate << " " << index_second_feature * num_candidates_in_model + index_second_candidate << " " << (float)new_cost << std::endl;
                }


                quadratic_expressions_costs[num_quadr_expr] = new_cost;

                quadratic_expressions_var1[num_quadr_expr] = all_model_variables[i];
                quadratic_expressions_var2[num_quadr_expr] = all_model_variables[j];

                //std::cout << index_first_feature << " " << index_fist_candidate << " " << index_second_feature << " " << index_second_candidate << " " << new_cost << std::endl;

                num_quadr_expr++;

            }
        }


        model_objective.addTerms(variable_costs,all_model_variables,total_num_variables);

        model_objective.addTerms(quadratic_expressions_costs,quadratic_expressions_var1,quadratic_expressions_var2,num_quadr_expr);

        model.setObjective(model_objective,GRB_MINIMIZE);

        model.update();

        //grb_print_quadratic_expr(&model_objective);
        
        model.optimize();
        solution_vector_from_gurobi_variables(*solution_instance,all_model_variables,num_features_points_in_model,num_candidates_in_model);

        if(!PARALLELIZE_MATCHING_CALCULATION){
            std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
            std::cout << "Obj percentage from theoretical optimum: " << (model.get(GRB_DoubleAttr_ObjVal)/theoretical_optimal_cost) * 100.0 << "%" << std::endl;
            std::cout << "Cost of Linear terms normalized by canidadates: " << get_cost_for_single_problem_instance(solution_instance,features,candidates,SINGLE_INSTANCE_LINEAR_COST,cost_rescale_factors)/(double)candidates->size() << std::endl;
            std::cout << "Cost of Linear terms normalized by features: " << get_cost_for_single_problem_instance(solution_instance,features,candidates,SINGLE_INSTANCE_LINEAR_COST,cost_rescale_factors)/(double)features->size() << std::endl;
        }



        print_instance_vector_as_vector(solution_instance);

        free(all_model_variables);
        free(variable_costs);

        free(quadratic_expressions_var1);
        free(quadratic_expressions_var2);
        free(quadratic_expressions_costs);


    }
    catch (GRBException e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch (...) {
        std::cout << "Exception during optimization" << std::endl;
    }

    //std::cout << "end gurobi" << std::endl;

    //model_file.close();

    return solution_instance;
    //return NULL;

}

void grb_print_quadratic_expr(GRBQuadExpr* quad_expr){


    GRBLinExpr lin_exp = quad_expr->getLinExpr();

    std::cout << "Print Quadratic Expression: " << lin_exp.size() << " " << quad_expr->size() << std::endl;

    for(int i = 0; i < lin_exp.size();i++){

        std::cout << lin_exp.getCoeff(i) << " * " << lin_exp.getVar(i).get(GRB_StringAttr_VarName) << " + ";

    }

    for(int i = 0; i < quad_expr->size();i++){

        std::cout << quad_expr->getCoeff(i) << " * " << quad_expr->getVar1(i).get(GRB_StringAttr_VarName) << " * " << quad_expr->getVar2(i).get(GRB_StringAttr_VarName);

        if(i < quad_expr->size() - 1 ){
            std::cout << " + ";
        }

    }

    std::cout << std::endl;

}

std::vector<int>* assign_candidates_to_features_using_gurobi_linearized(std::vector<Feature_Point_Data>* features, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors){

    int num_features_points_in_model = features->size();
    int num_candidates_in_model = candidates->size();

    int max_num_points = std::max<int>(num_features_points_in_model,num_candidates_in_model);
    double theoretical_optimal_cost = get_optimal_cost_for_problem_size(max_num_points,cost_rescale_factors);

    std::cout << "Theoretical optimum: " << theoretical_optimal_cost << std::endl;

    int total_num_linear_variables = num_features_points_in_model * num_candidates_in_model;
    //for explanation of the calculation see the non linearized version of the function
    int total_num_linearized_quadratic_expressions = (total_num_linear_variables * total_num_linear_variables - total_num_linear_variables) / 2;

    int combined_num_variables = total_num_linear_variables + total_num_linearized_quadratic_expressions;

    std::vector<int>* solution_instance = new std::vector<int>(num_features_points_in_model,-1);

    try {

        /*
        GRBEnv env = GRBEnv(true);
        env.set("LogFile", "mip1.log");
        env.start();

        // Create an empty model
        GRBModel model = GRBModel(env);

        GRBVar model_vars[6];

        model_vars[0] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x1");
        model_vars[1] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x2");
        model_vars[2] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x3");

        model_vars[3] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x13");
        model_vars[4] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x23");
        model_vars[5] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x12");

        model.setObjective(-0.1 * model_vars[0] + -0.1 * model_vars[1] + -0.1 * model_vars[2] + 2.0 * model_vars[3] + 2.0 * model_vars[4] + -0.1 * model_vars[5]);

        model.addConstr(model_vars[3] <= model_vars[0]);
        model.addConstr(model_vars[3] <= model_vars[2]);
        model.addConstr(model_vars[3] >= model_vars[0] + model_vars[2] -1);

        model.addConstr(model_vars[4] <= model_vars[1]);
        model.addConstr(model_vars[4] <= model_vars[2]);
        model.addConstr(model_vars[4] >= model_vars[1] + model_vars[2] -1);

        model.addConstr(model_vars[5] <= model_vars[0]);
        model.addConstr(model_vars[5] <= model_vars[1]);
        model.addConstr(model_vars[5] >= model_vars[0] + model_vars[1] -1);

        model.optimize();

        std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
        std::cout << model_vars[0].get(GRB_DoubleAttr_X) << " " << model_vars[1].get(GRB_DoubleAttr_X) << " " << model_vars[2].get(GRB_DoubleAttr_X) << " " << model_vars[3].get(GRB_DoubleAttr_X) << " " << model_vars[4].get(GRB_DoubleAttr_X) << " " << model_vars[5].get(GRB_DoubleAttr_X) << std::endl; 

        */
        // Create an environment
        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag, PRINT_GUROBI_LOGS);
        env.set("LogFile", "mip1.log");
        env.start();

        // Create an empty model
        GRBModel model = GRBModel(env);

        model.getEnv().set(GRB_DoubleParam_TimeLimit, GUROBI_TIMELIMIT_SECONDS);

        GRBVar* all_model_variables = (GRBVar*)malloc(sizeof(GRBVar) * combined_num_variables);
        double* variable_costs = (double*)malloc(sizeof(double) * combined_num_variables);


        //GRBVar* quadratic_expressions_var1 = (GRBVar*)malloc(sizeof(GRBVar) * total_num_quadratic_expressions);
        //GRBVar* quadratic_expressions_var2 = (GRBVar*)malloc(sizeof(GRBVar) * total_num_quadratic_expressions);
        //double* quadratic_expressions_costs = (double*)malloc(sizeof(double) * total_num_quadratic_expressions);

        GRBQuadExpr model_objective;


        std::cout << "begin add unary terms" << std::endl;
        for(int i = 0; i < num_features_points_in_model; i++){
            for(int j = 0; j < num_candidates_in_model; j++){

                int var_index = i * num_candidates_in_model + j;

                std::string new_variable_name = "x" + std::to_string(i+1) + std::to_string(j+1);
                //std::cout << "added new variable: " << new_variable_name << " at index: " << var_index << std::endl;
                all_model_variables[var_index] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, new_variable_name);
                
                variable_costs[var_index] = calculate_cost_for_two_feature_points((*features)[i],(*candidates)[j],cost_rescale_factors);

                //model_objective.addTerm(C,all_model_variables[var_index]);
                //model_objective.addTerms()
            
            }

        }

        model.update();

        int num_quadr_expr = 0;

        for(int i = 0; i < total_num_linear_variables;i++){
            for(int j = i+1; j < total_num_linear_variables;j++){

                int index_first_feature = i / num_candidates_in_model;
                int index_first_candidate = i % num_candidates_in_model;

                int index_second_feature = j / num_candidates_in_model;
                int index_second_candidate = j % num_candidates_in_model;

                double new_cost = 0.0;

                if(index_first_feature == index_second_feature || index_first_candidate == index_second_candidate){
                    new_cost = COST_DOUBLE_ASSIGNMENT;
                }else{
                    new_cost = calculate_cost_for_single_feature_point_pair_assignment((*features)[index_first_feature],(*candidates)[index_first_candidate],(*features)[index_second_feature],(*candidates)[index_second_candidate],cost_rescale_factors);
                }

                int index_of_linearized_variable = total_num_linear_variables + num_quadr_expr;

                std::string new_variable_name = "x" + std::to_string(index_first_feature+1) + std::to_string(index_first_candidate+1) + "_" + std::to_string(index_second_feature+1) + std::to_string(index_second_candidate+1);
                //std::cout << "added new variable: " << new_variable_name << std::endl;
                all_model_variables[index_of_linearized_variable] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, new_variable_name);


                variable_costs[index_of_linearized_variable] = new_cost;

                int index_of_first_variable = index_first_candidate + num_candidates_in_model * index_first_feature;
                int index_of_second_variable = index_second_candidate + num_candidates_in_model * index_second_feature;

                //std::cout << "added new variable: " << new_variable_name << " indices of other vars: " << index_of_first_variable << " " << index_of_second_variable << std::endl;

                //model.update();
                /*
                GRBVar var_1 = all_model_variables[index_of_first_variable];
                GRBVar var_2 = all_model_variables[index_of_second_variable];
                GRBVar var_12 = all_model_variables[index_of_linearized_variable];

                std::cout << var_12.get(GRB_StringAttr_VarName) << " <= " << var_1.get(GRB_StringAttr_VarName) << std::endl;
                std::cout << var_12.get(GRB_StringAttr_VarName) << " <= " << var_2.get(GRB_StringAttr_VarName) << std::endl;
                std::cout << var_12.get(GRB_StringAttr_VarName) << " >= " << var_1.get(GRB_StringAttr_VarName) << " + " << var_2.get(GRB_StringAttr_VarName) << " - 1" << std::endl;
                std::cout << std::endl;
                */
                model.addConstr(all_model_variables[index_of_linearized_variable] <= all_model_variables[index_of_first_variable],new_variable_name + "_c1");
                model.addConstr(all_model_variables[index_of_linearized_variable] <= all_model_variables[index_of_second_variable],new_variable_name + "_c2");
                model.addConstr(all_model_variables[index_of_linearized_variable] >= all_model_variables[index_of_first_variable] + all_model_variables[index_of_second_variable] - 1,new_variable_name + "_c3");

                //quadratic_expressions_var1[num_quadr_expr] = all_model_variables[i];
                //quadratic_expressions_var2[num_quadr_expr] = all_model_variables[j];


                num_quadr_expr++;   
            }
        }


        model_objective.addTerms(variable_costs,all_model_variables,combined_num_variables);

        //model_objective.addTerms(quadratic_expressions_costs,quadratic_expressions_var1,quadratic_expressions_var2,num_quadr_expr);

        model.setObjective(model_objective,GRB_MINIMIZE);

        model.update();

        //grb_print_quadratic_expr(&model_objective);
        
        std::cout << "begin optimize" << std::endl;

        model.optimize();

        std::cout << "end optimize" << std::endl;
        solution_vector_from_gurobi_variables(*solution_instance,all_model_variables,num_features_points_in_model,num_candidates_in_model);

        if (!PARALLELIZE_MATCHING_CALCULATION) {
            std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
            std::cout << "Obj percentage from theoretical optimum: " << (model.get(GRB_DoubleAttr_ObjVal)/theoretical_optimal_cost) * 100.0 << "%" << std::endl;
            std::cout << "Cost of Linear terms normalized by canidadates: " << get_cost_for_single_problem_instance(solution_instance,features,candidates,SINGLE_INSTANCE_LINEAR_COST,cost_rescale_factors)/(double)candidates->size() << std::endl;
            std::cout << "Cost of Linear terms normalized by features: " << get_cost_for_single_problem_instance(solution_instance,features,candidates,SINGLE_INSTANCE_LINEAR_COST,cost_rescale_factors)/(double)features->size() << std::endl;
        }


        print_instance_vector_as_matrix(solution_instance,num_features_points_in_model,num_candidates_in_model);

        free(all_model_variables);
        free(variable_costs);

        //free(quadratic_expressions_var1);
        //free(quadratic_expressions_var2);
        //free(quadratic_expressions_costs);


    }
    catch (GRBException e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch (...) {
        std::cout << "Exception during optimization" << std::endl;
    }

    return solution_instance;

}

void solution_vector_from_gurobi_variables(std::vector<int>& solution_vector, GRBVar* all_model_variables, int num_features_points_in_model, int num_candidates_in_model){
    
    for(int i = 0; i < num_features_points_in_model;i++){
            for(int j = 0; j < num_candidates_in_model;j++){
                int var_index = i * num_candidates_in_model + j;

                GRBVar current_var = all_model_variables[var_index];


                if(current_var.get(GRB_DoubleAttr_X) > 0.0){
                    solution_vector[i] = j;
                }
           
            }

        }


}

Matching_Result find_matching_between_two_organoids(std::vector<Feature_Point_Data>* features, std::vector<Feature_Point_Data>* candidates, All_Cost_Parameters& cost_params, bool solve_to_optimality, Assignment_Visualization_Data* viz_data, int index_features_in_viz_data, int index_candidates_in_viz_data, std::vector<Op_Numbers_And_Runtime>* op_numbers, Memory_Manager_Fixed_Size* mem_manager){

    //std::vector<Feature_Point_Data>* smaller_feature_vector = features;
    //std::vector<Feature_Point_Data>* larger_feature_vector = candidates;

    


    Matching_Result matching_results;
    matching_results.id_1 = -1;
    matching_results.id_2 = -1;

    matching_results.assignment = new std::vector<int>;
    //std::cout << "allocates new assignment at: " << matching_results.assignment << std::endl;

    matching_results.additional_viz_data_id1_to_id2 = nullptr;
    matching_results.additional_viz_data_id2_to_id1 = nullptr;

    //print_model_parameters(cost_params,true);

    
    //std::cout << "Matching: " << features->size() << " to: " << candidates->size() << std::endl;
    Cost_Rescale_Factors cost_rescale_factors = get_cost_rescale_factors(features->size(), cost_params);



    std::vector<int> empty_solution(features->size(),-1);

    
    if(USE_HISTOGRAM_COMPARISON_FOR_MATCHING){

        cv::Mat image_feature;
        cv::Mat image_candidates;


        image_feature = cv::imread(viz_data->image_paths->at(index_features_in_viz_data).string());
        image_candidates = cv::imread(viz_data->image_paths->at(index_candidates_in_viz_data).string());

        std::vector<cv::Mat> histograms_features;
        std::vector<cv::Mat> histograms_candidates;

        calculate_histogram(image_feature,histograms_features,index_features_in_viz_data);
        calculate_histogram(image_candidates,histograms_candidates,index_candidates_in_viz_data);

        double histogram_similarity = compare_histograms(histograms_features,histograms_candidates);

        if(histogram_similarity < 0.0){
            histogram_similarity = 0;
        }

        if(histogram_similarity > 1.0){
            histogram_similarity = 1.0;
        }

        matching_results.rel_quadr_cost = histogram_similarity;

        return matching_results;

    }

    double theoretical_optimal_cost = get_optimal_cost_for_problem_size(features->size(),cost_rescale_factors);
    std::vector<int>* best_found_solution = &empty_solution;
    double cost_of_best_solution = 0.0;

    Op_Numbers_And_Runtime op_number_struct;
    op_number_struct.total_runtime = 0;

    op_number_struct.runtime[BAB_GREEDY_ALGORITHM] = 0;
    op_number_struct.runtime[BAB_GREEDY_CYCLIC_ALGORITHM] = 0;
    op_number_struct.runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM] = 0;
    op_number_struct.runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] = 0;
    op_number_struct.runtime[GREEDY_COST_MATCHING_ALGORITHM] = 0;
    op_number_struct.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] = 0;
    op_number_struct.runtime[GUROBI_ALGORITHM] = 0;

    op_number_struct.normalized_results[BAB_GREEDY_ALGORITHM] = 0;
    op_number_struct.normalized_results[BAB_GREEDY_CYCLIC_ALGORITHM] = 0;
    op_number_struct.normalized_results[GREEDY_DISTANCE_MATCHING_ALGORITHM] = 0;
    op_number_struct.normalized_results[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] = 0;
    op_number_struct.normalized_results[GREEDY_COST_MATCHING_ALGORITHM] = 0;
    op_number_struct.normalized_results[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] = 0;
    op_number_struct.normalized_results[GUROBI_ALGORITHM] = 1.0;

    std::chrono::steady_clock::time_point total_begin = std::chrono::steady_clock::now();


    std::vector<int>* gurobi_solution = nullptr;
    if(solve_to_optimality){
        std::chrono::steady_clock::time_point gurobi_begin = std::chrono::steady_clock::now();
        gurobi_solution = assign_candidates_to_features_using_gurobi(features,candidates,cost_rescale_factors);//,best_found_solution);
        std::chrono::steady_clock::time_point gurobi_end = std::chrono::steady_clock::now();

        double gurobi_time = std::chrono::duration_cast<std::chrono::microseconds>(gurobi_end - gurobi_begin).count();
        double cost_of_gurobi_solution = get_cost_for_single_problem_instance(gurobi_solution, features, candidates, SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);

        fill_op_number_struct(op_number_struct,features->size(),candidates->size(),GUROBI_ALGORITHM,gurobi_time,cost_of_gurobi_solution/theoretical_optimal_cost);
    }

    std::vector<int>* qap_solver_solution = nullptr;
    double cost_of_qap_solver_solution = 0;
    if(USE_QAP_SOLVER){
        qap_solver_solution = assign_candidates_to_features_using_qap_solver(features,candidates,cost_rescale_factors);
        cost_of_qap_solver_solution = get_cost_for_single_problem_instance(qap_solver_solution, features, candidates, SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
    }
    if(cost_of_qap_solver_solution < cost_of_best_solution){
        //std::cout << " qap solver was better: " << cost_of_qap_solver_solution << " than: " << cost_of_best_solution << std::endl;
        best_found_solution = qap_solver_solution;
        cost_of_best_solution = cost_of_qap_solver_solution;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time;

    /*
    begin = std::chrono::steady_clock::now();
    int best_angle_from_greedy_distance_matching = 0;
    std::vector<int> greedy_distance_solution = find_initial_solution_by_rotating_and_greedy_distance_matching(features,candidates,best_angle_from_greedy_distance_matching,cost_rescale_factors);
    end = std::chrono::steady_clock::now();

    double cost_of_greedy_distance_no_neighborhood = get_cost_for_single_problem_instance(&greedy_distance_solution, features, candidates, SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    fill_op_number_struct(op_number_struct,features->size(),candidates->size(),GREEDY_DISTANCE_MATCHING_ALGORITHM,time,cost_of_greedy_distance_no_neighborhood/theoretical_optimal_cost);
    */


    int best_angle_from_greedy_distance_matching_in_neighborhood = 0;
    begin = std::chrono::steady_clock::now();
    std::vector<int> greedy_distance_neighborhood_solution = find_initial_solution_by_rotating_and_greedy_distance_matching_neighborhood_cone(features,candidates,best_angle_from_greedy_distance_matching_in_neighborhood,cost_rescale_factors);
    end = std::chrono::steady_clock::now();

    //std::cout << "best angle: " << best_angle_from_greedy_distance_matching_in_neighborhood << std::endl;

    double cost_of_greedy_distance = get_cost_for_single_problem_instance(&greedy_distance_neighborhood_solution, features, candidates, SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    //std::cout << "cost_of_greedy_distance: " << cost_of_greedy_distance << std::endl;
    fill_op_number_struct(op_number_struct,features->size(),candidates->size(),GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD,time,cost_of_greedy_distance/theoretical_optimal_cost);


    if(cost_of_greedy_distance < cost_of_best_solution){
        best_found_solution = &greedy_distance_neighborhood_solution;
        cost_of_best_solution = cost_of_greedy_distance;
    }




    begin = std::chrono::steady_clock::now();
    std::vector<int> greedy_cost_solution_angle_range = find_initial_solution_by_rotating_and_greedy_cost_matching_in_angle_range(features,candidates,best_angle_from_greedy_distance_matching_in_neighborhood,ANGLE_RANGE_SEARCH_SIZE,cost_rescale_factors);
    end = std::chrono::steady_clock::now();
    double cost_of_greedy_cost_angle_range = get_cost_for_single_problem_instance(&greedy_cost_solution_angle_range, features, candidates, SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);
    //std::cout << "cost_of_greedy_cost_angle_range: " << cost_of_greedy_cost_angle_range << std::endl;
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    fill_op_number_struct(op_number_struct,features->size(),candidates->size(),GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE,time,cost_of_greedy_cost_angle_range/theoretical_optimal_cost);

    if(cost_of_greedy_cost_angle_range < cost_of_best_solution){
        best_found_solution = &greedy_cost_solution_angle_range;
        cost_of_best_solution = cost_of_greedy_cost_angle_range;
    }

    //std::cout << "cost of best rotate solution: " << cost_of_best_solution << std::endl;

    if(USE_GREEDY_ASSIGNMENT_TRANSPOSITIONING){
        std::vector<int> instance_of_angle_range_and_transpose = *best_found_solution;
        begin = std::chrono::steady_clock::now();
        apply_greedy_assignment_transposition(&instance_of_angle_range_and_transpose,cost_rescale_factors,features,candidates);
        end = std::chrono::steady_clock::now();
        double cost_of_transpose_instance = get_cost_for_single_problem_instance(&instance_of_angle_range_and_transpose, features, candidates, SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);

        time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        fill_op_number_struct(op_number_struct,features->size(),candidates->size(),GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION,time,cost_of_transpose_instance/theoretical_optimal_cost);

        if(cost_of_transpose_instance < cost_of_best_solution){
            cost_of_best_solution = cost_of_transpose_instance;
            *best_found_solution = instance_of_angle_range_and_transpose; 
        }
        //std::cout << "cost_of_transpose_instance: " << cost_of_transpose_instance << std::endl;
    }




    std::vector<int>* bab_solution = nullptr;
    std::vector<int>* bab_cyclic_greedy_solution = nullptr;

    double cost_of_greedy_cyclic = 0.0;
    double cost_of_greedy_bab = 0.0;

    begin = std::chrono::steady_clock::now();
    //bab_cyclic_greedy_solution = assign_candidates_to_features_using_branch_and_bound(features,candidates,BaB_CYCLIC_GREEDY_SEARCH_ONLY,cost_rescale_factors,mem_manager,cost_of_best_solution,best_found_solution);
     bab_cyclic_greedy_solution = assign_candidates_to_features_using_branch_and_bound(features,candidates,BaB_CYCLIC_GREEDY_SEARCH_ONLY,cost_rescale_factors,mem_manager,0.0,&empty_solution);
    end = std::chrono::steady_clock::now();

    cost_of_greedy_cyclic = get_cost_for_single_problem_instance(bab_cyclic_greedy_solution, features, candidates, SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);

    //std::cout << "greedy cyclic: " << cost_of_greedy_cyclic << " ratio: " << cost_of_greedy_cyclic/theoretical_optimal_cost << std::endl;
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    fill_op_number_struct(op_number_struct,features->size(),candidates->size(),BAB_GREEDY_CYCLIC_ALGORITHM,time,cost_of_greedy_cyclic/theoretical_optimal_cost);
    

    if (cost_of_greedy_cyclic < cost_of_best_solution) {
        //std::cout << " greedy cyclic was better: " << cost_of_greedy_cyclic << " than: " << cost_of_best_solution << std::endl;
        best_found_solution = bab_cyclic_greedy_solution;
        cost_of_best_solution = cost_of_greedy_cyclic;
    }
    


    if(USE_GREEDY_ASSIGNMENT_TRANSPOSITIONING){
        begin = std::chrono::steady_clock::now();
        apply_greedy_assignment_transposition(bab_cyclic_greedy_solution,cost_rescale_factors,features,candidates);
        end = std::chrono::steady_clock::now();

        double cost_of_greedy_cyclic_with_transposition = get_cost_for_single_problem_instance(bab_cyclic_greedy_solution, features, candidates, SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors);

        //std::cout << "greedy cyclic transpose: " << cost_of_greedy_cyclic_with_transposition << " ratio: " << cost_of_greedy_cyclic_with_transposition/theoretical_optimal_cost << std::endl;

        if (cost_of_greedy_cyclic_with_transposition < cost_of_best_solution) {
            //std::cout << " greedy cyclic transp was better: " << cost_of_greedy_cyclic_with_transposition << " than: " << cost_of_best_solution << std::endl;
            best_found_solution = bab_cyclic_greedy_solution;
            cost_of_best_solution = cost_of_greedy_cyclic_with_transposition;
        }

        time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        fill_op_number_struct(op_number_struct,features->size(),candidates->size(),BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION,time,cost_of_greedy_cyclic_with_transposition/theoretical_optimal_cost);
    }

    //std::cout << "end comparing: " << cost_of_greedy_distance << " " << cost_of_greedy_cost << " " << cost_of_greedy_cyclic << " " << features->size() << " " << candidates->size() << std::endl;


    std::chrono::steady_clock::time_point total_end = std::chrono::steady_clock::now();

    double total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_begin).count();
    op_number_struct.total_runtime = total_time;
    


    matching_results.rel_quadr_cost = get_cost_for_single_problem_instance(best_found_solution,features,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors)/theoretical_optimal_cost;

    double linear_cost = get_cost_for_single_problem_instance(best_found_solution,features,candidates,SINGLE_INSTANCE_LINEAR_COST,cost_rescale_factors);

    matching_results.linear_cost_per_feature = linear_cost/(double)features->size();
    matching_results.linear_cost_per_candidate = linear_cost/(double)candidates->size();

    if(solve_to_optimality){
        matching_results.rel_quadr_cost_optimal = get_cost_for_single_problem_instance(gurobi_solution,features,candidates,SINGLE_INSTANCE_TOTAL_COST,cost_rescale_factors)/theoretical_optimal_cost;
    }else{
        matching_results.rel_quadr_cost_optimal = 0.0;
    }


    
    if(op_numbers != nullptr){
    
        op_numbers->push_back(op_number_struct);
    }
    



    if(PRINT_SOLVER_PROGRESS){
        std::cout << "Solution:" << std::endl;
        std::cout << "rel quadr cost: " << matching_results.rel_quadr_cost << std::endl;
        std::cout << "linear cost per feature: " << matching_results.linear_cost_per_feature << std::endl;
        std::cout << "linear cost per candidate: " << matching_results.linear_cost_per_candidate << std::endl;
    }

    if (VISUALIZE_INDIVIDUAL_ASSIGNMENTS && viz_data != nullptr) {
        //print_instance_vector_as_vector(gurobi_solution);
        matching_results.additional_viz_data_id1_to_id2 = new Matching_Result_Additional_Viz_Data;//(Matching_Result_Additional_Viz_Data*)malloc(sizeof(Matching_Result_Additional_Viz_Data));
        matching_results.additional_viz_data_id2_to_id1 = nullptr;

        matching_results.additional_viz_data_id1_to_id2->features = features;
        matching_results.additional_viz_data_id1_to_id2->assigned_candidates = candidates;

        matching_results.additional_viz_data_id1_to_id2->feature_image = nullptr;//new cv::Mat;
        //*(matching_results.additional_viz_data_id1_to_id2->feature_image) = viz_data->images->at(index_features_in_viz_data)->clone();

        matching_results.additional_viz_data_id1_to_id2->candidate_image = nullptr;//new cv::Mat;
        //*(matching_results.additional_viz_data_id1_to_id2->candidate_image) = viz_data->images->at(index_candidates_in_viz_data)->clone();

        matching_results.additional_viz_data_id1_to_id2->path_to_feature_image = viz_data->image_paths->at(index_features_in_viz_data);
        matching_results.additional_viz_data_id1_to_id2->path_to_candidate_image = viz_data->image_paths->at(index_candidates_in_viz_data);

        matching_results.additional_viz_data_id1_to_id2->center_feature_image = viz_data->centers->at(index_features_in_viz_data);
        matching_results.additional_viz_data_id1_to_id2->center_candidate_image = viz_data->centers->at(index_candidates_in_viz_data);

        matching_results.additional_viz_data_id1_to_id2->offset_feature_image = viz_data->offsets->at(index_features_in_viz_data);
        matching_results.additional_viz_data_id1_to_id2->offset_candidate_image = viz_data->offsets->at(index_candidates_in_viz_data);

        //matching_results.additional_viz_data_id1_to_id2->assignment = new std::vector<int>;
        //*(matching_results.additional_viz_data_id1_to_id2->assignment) = *(best_found_solution);
        /*
        cv::Mat* image_from_smaller_vector = viz_data->images->at(index_features_in_viz_data);
        cv::Mat* image_from_larger_vector = viz_data->images->at(index_candidates_in_viz_data);

        cv::Point2i center_from_smaller_image = viz_data->centers->at(index_features_in_viz_data);
        cv::Point2i center_from_larger_image = viz_data->centers->at(index_candidates_in_viz_data);

        cv::Point2i offset_from_smaller_image = viz_data->offsets->at(index_features_in_viz_data);
        cv::Point2i offset_from_larger_image = viz_data->offsets->at(index_candidates_in_viz_data);

        if(features->size() > candidates->size()){
            image_from_smaller_vector = viz_data->images->at(index_candidates_in_viz_data);
            image_from_larger_vector = viz_data->images->at(index_features_in_viz_data);

            center_from_smaller_image = viz_data->centers->at(index_candidates_in_viz_data);
            center_from_larger_image = viz_data->centers->at(index_features_in_viz_data);

            offset_from_smaller_image = viz_data->offsets->at(index_candidates_in_viz_data);
            offset_from_larger_image = viz_data->offsets->at(index_features_in_viz_data);
        }
        */

       /*
        show_assignment(best_found_solution,
            features, candidates,
            viz_data->images->at(index_features_in_viz_data),
            viz_data->images->at(index_candidates_in_viz_data),
            viz_data->centers->at(index_features_in_viz_data),
            viz_data->centers->at(index_candidates_in_viz_data),
            viz_data->offsets->at(index_features_in_viz_data),
            viz_data->offsets->at(index_candidates_in_viz_data));
       */
    }

    *(matching_results.assignment) = *best_found_solution;
    //*(matching_results.assignment) = *gurobi_solution;

    if (bab_solution != nullptr) {
        //std::cout << "deleting bab_solution at: " << bab_solution << std::endl;
        delete(bab_solution);
    }

    if (bab_cyclic_greedy_solution != nullptr) {
        //std::cout << "deleting bab_cyclic_greedy_solution at: " << bab_cyclic_greedy_solution << std::endl;
        delete(bab_cyclic_greedy_solution);
    }

    if (gurobi_solution != nullptr) {
        //std::cout << "deleting gurobi_solution at: " << gurobi_solution << std::endl;
        delete(gurobi_solution);
    }

    if(qap_solver_solution != nullptr){
        delete(qap_solver_solution);
    }

    return matching_results;
}

void evaluate_all_possible_matchings(std::filesystem::path image_folder, const Input_Arguments args){

    std::chrono::steady_clock::time_point matching_startup_begin = std::chrono::steady_clock::now();


    std::string name_of_matching_results_file = "matching_results_15_3_17_10.csv";
    if(args.matching_results_file_name != ""){
        name_of_matching_results_file = args.matching_results_file_name;
    }

    std::string name_of_feature_vectors_file = "feature_vectors_15_3_13_56.csv";
    if(args.feature_vector_file_name != ""){
        name_of_feature_vectors_file = args.feature_vector_file_name;
    }

    std::filesystem::path matching_results_file_path = get_data_folder_path();
    matching_results_file_path.append("matching_results");

    std::filesystem::path secondary_matching_results_file_path = matching_results_file_path;
    secondary_matching_results_file_path.append(args.second_matching_results_file_name);

    matching_results_file_path.append(name_of_matching_results_file);

    std::filesystem::path feature_vectors_file_path = get_data_folder_path();
    feature_vectors_file_path.append("feature_vectors");

    std::filesystem::path secondary_feature_vectors_file_path = feature_vectors_file_path;
    secondary_feature_vectors_file_path.append(args.second_feature_vector_file_name);

    feature_vectors_file_path.append(name_of_feature_vectors_file);

    int elem_size = 0;
    int individual_image_target_size = MATCHING_MATRIX_INDIVIDUAL_SQUARE_SIZE;

    std::vector<Image_Features_Pair> all_feature_vectors;

    std::vector<cv::Mat*> square_organoid_images;
    std::vector<cv::Mat*> sorted_square_organoid_images;

    std::vector<cv::Point2i> image_centers;
    std::vector<cv::Point2i> sorted_image_centers;

    std::vector<cv::Point2i> offsets_from_squaring;
    std::vector<cv::Point2i> sorted_offsets_from_squaring;

    std::vector<std::filesystem::path> image_paths;
    std::vector<std::filesystem::path> sorted_image_paths;

    std::vector<int> image_number_index;

    std::string name_of_writen_matching_result_file = "";
    float value_of_used_cluster_threshold = -1.0f;

    All_Cost_Parameters cost_params = args.all_model_parameters;

    std::vector<Matching_Result> all_matching_results;

    init_mpi(0,NULL);

    int process_id = get_mpi_process_rank();
    int total_num_processes = get_num_mpi_processes();

    std::cout << "Process: " << process_id << " of " << total_num_processes << std::endl;

    int img_number_offset = 0;

    if(args.skip_visualization && args.read_feature_vector){
        read_image_numbers_and_feature_vectors_from_file(feature_vectors_file_path,all_feature_vectors,1);

        if(args.use_secondary_image_set){
            read_image_numbers_and_feature_vectors_from_file(secondary_feature_vectors_file_path,all_feature_vectors,2);
        }
        /*
        std::vector<int> image_numbers_in_feature_file;

        read_image_numbers_from_feature_vector_file(feature_vectors_file_path,image_numbers_in_feature_file);

        for(int i = 0; i < image_numbers_in_feature_file.size(); i++){

            int current_image_number = image_numbers_in_feature_file[i];

            Image_Features_Pair new_ifp;
            new_ifp.image_path = "";
            new_ifp.image_number = current_image_number;

            new_ifp.features = new std::vector<Feature_Point_Data>;
            new_ifp.candidates = new std::vector<Feature_Point_Data>;

            read_feature_vector_from_file(feature_vectors_file_path,current_image_number,*(new_ifp.features),*(new_ifp.candidates));

            all_feature_vectors.push_back(new_ifp);
        }
        */

    }else{

        bool image_folder_exits = std::filesystem::is_directory(image_folder);

        if(!image_folder_exits){
            std::cerr << "ERROR: folder: " << image_folder << " does NOT EXISTS!" << std::endl;
            return;
        }

        int largest_img_number_in_set_1 = 0;

        largest_img_number_in_set_1 = extract_all_images_from_folder(image_folder,args,feature_vectors_file_path,square_organoid_images,all_feature_vectors,image_number_index,image_centers,offsets_from_squaring,image_paths,1);

        if(args.write_feature_vector && process_id == 0){
            std::string file_name_for_feature_vectors = "feature_vectors_";

            file_name_for_feature_vectors += args.image_set_name;
            file_name_for_feature_vectors += args.current_time_string;
            file_name_for_feature_vectors += ".csv";

            write_all_feature_vectors_to_file(file_name_for_feature_vectors,all_feature_vectors,args);
        
        }

        if(args.use_secondary_image_set){

            std::filesystem::path secondary_image_set_path = image_folder.parent_path();
            secondary_image_set_path.append(args.second_image_set_name);


            img_number_offset = largest_img_number_in_set_1;

            extract_all_images_from_folder(secondary_image_set_path,args,secondary_feature_vectors_file_path,square_organoid_images,all_feature_vectors,image_number_index,image_centers,offsets_from_squaring,image_paths,2,img_number_offset);

                if(args.write_feature_vector && process_id == 0){
                    std::string file_name_for_combined_feature_vectors = "feature_vectors_";

                    file_name_for_combined_feature_vectors += args.image_set_name;
                    file_name_for_combined_feature_vectors += "_";
                    file_name_for_combined_feature_vectors += args.second_image_set_name; 
                    file_name_for_combined_feature_vectors += args.current_time_string;
                    file_name_for_combined_feature_vectors += ".csv";

                    write_all_feature_vectors_to_file(file_name_for_combined_feature_vectors,all_feature_vectors,args);         
                }
        }
    }
    

    //std::cout << "Finished feature extraction" << std::endl;

    sort_image_feature_pair_vector_by_image_number(&all_feature_vectors);

    Assignment_Visualization_Data viz_data;
    viz_data.centers = nullptr;
    viz_data.images = nullptr;
    viz_data.offsets = nullptr;

    if(!args.skip_visualization){
        sorted_square_organoid_images.reserve(square_organoid_images.size());

        //sort the other vectors based on the new order of the all_feature_vector and the original order which is stored in the image_number_index vector
        for(int i = 0; i < all_feature_vectors.size();i++){
            Image_Features_Pair current_ifp = all_feature_vectors[i];

            int index_of_image = 0; 

            for(int j = 0; j < image_number_index.size();j++){
                if(image_number_index[j] == current_ifp.image_number){
                    index_of_image = j;
                    break;
                }

            }

            sorted_square_organoid_images.push_back(square_organoid_images[index_of_image]);
            sorted_image_centers.push_back(image_centers[index_of_image]);
            //std::cout << "sorted_image_center: " << sorted_image_centers[i] << std::endl;
            sorted_offsets_from_squaring.push_back(offsets_from_squaring[index_of_image]);
            sorted_image_paths.push_back(image_paths[index_of_image]);
        }

        viz_data.centers = &sorted_image_centers;
        viz_data.images = &sorted_square_organoid_images;
        viz_data.offsets = &sorted_offsets_from_squaring;
        viz_data.image_paths = &sorted_image_paths;
        
        for(int i = 0; i < sorted_square_organoid_images.size();i++){

            cv::Size rescaled_image_size(CLUSTERING_VIEW_INDIVIDUAL_SQUARE_SIZE,CLUSTERING_VIEW_INDIVIDUAL_SQUARE_SIZE);

            cv::resize(*(sorted_square_organoid_images[i]),*(sorted_square_organoid_images[i]),rescaled_image_size);
        }

        //print_square_organoid_images(sorted_square_organoid_images);
    }

    //char* empty = " ";

    Additional_Viz_Data_Input viz_input;

    viz_input.all_feature_vectors = &all_feature_vectors;
    viz_input.sorted_image_centers = &sorted_image_centers;
    viz_input.sorted_offsets = &sorted_offsets_from_squaring;
    viz_input.sorted_square_organoid_images = &sorted_square_organoid_images;
    viz_input.sorted_image_paths = &sorted_image_paths;

    if(!args.skip_matching){

        if(args.read_matching_results){

        All_Cost_Parameters cost_params_from_file{  COST_DOUBLE_ASSIGNMENT,
                                                    COST_MISMATCHING_CHANNELS,
                                                    DEFAULT_COLOR_COST_OFFSET,
                                                    DEFAULT_DIST_COST_OFFSET,
                                                    DEFAULT_ANGLE_COST_OFFSET,
                                                    DEFAULT_COLOR_TO_DIST_WEIGHT,
                                                    DEFAULT_UNARY_TO_QUADR_WEIGHT};
        All_Feature_Point_Selection_Parameters feature_selection_params;

        
        //read_matching_results_from_file("matching_results_8_2_13_56.csv",&all_matching_results,cost_params_from_file);

        bool read_pairwise_comparison = false;

        if(!read_pairwise_comparison){
            if(args.use_secondary_image_set){
                read_matching_results_from_file(secondary_matching_results_file_path,&all_matching_results,cost_params_from_file,feature_selection_params);
            }else{
                read_matching_results_from_file(matching_results_file_path,&all_matching_results,cost_params_from_file,feature_selection_params);
            }

            set_image_set_numbers_in_matching_results(all_matching_results,image_folder.parent_path(),args,img_number_offset);
        }else{

            std::filesystem::path path_to_comparison_file = get_data_folder_path();

            path_to_comparison_file.append("clustering_results");

            path_to_comparison_file.append("pairwise_results");

            path_to_comparison_file.append(PAIRWISE_COMPARISON_FILENAME);

            read_pairwise_comparison_as_matching_results(path_to_comparison_file,all_matching_results);

        }

        //read_matching_results_from_file("matching_results_23_2_10_44.csv",&all_matching_results,cost_params_from_file);

        cost_params = cost_params_from_file;



        //print_cost_parameters(&cost_params_from_file);


            if(!args.skip_visualization){

                //std::cout << all_matching_results.size() << std::endl;

                for(int i = 0; i < all_matching_results.size();i++){

                    //std::cout << i << std::endl;
                    int index_of_image_1 = 0;
                    int index_of_image_2 = 0;  

                    for(int j = 0; j < all_feature_vectors.size();j++){
                        Image_Features_Pair current_ifp = all_feature_vectors[j];
                        

                        if(current_ifp.image_number == all_matching_results[i].id_1){
                            index_of_image_1 = j;
                        }

                        if(current_ifp.image_number == all_matching_results[i].id_2){
                            index_of_image_2 = j;
                        }
                    }

                    //all_matching_results[i].additional_viz_data_id1_to_id2 = new Matching_Result_Additional_Viz_Data;
                    //all_matching_results[i].additional_viz_data_id2_to_id1 = new Matching_Result_Additional_Viz_Data;

                    all_matching_results[i].additional_viz_data_id1_to_id2->center_feature_image = sorted_image_centers[index_of_image_1];
                    all_matching_results[i].additional_viz_data_id1_to_id2->center_candidate_image = sorted_image_centers[index_of_image_2];

                    all_matching_results[i].additional_viz_data_id1_to_id2->offset_feature_image = sorted_offsets_from_squaring[index_of_image_1];
                    all_matching_results[i].additional_viz_data_id1_to_id2->offset_candidate_image = sorted_offsets_from_squaring[index_of_image_2];

                    //all_matching_results[i].additional_viz_data_id1_to_id2->feature_image = sorted_square_organoid_images[index_of_image_1];
                    all_matching_results[i].additional_viz_data_id1_to_id2->feature_image = nullptr;//new cv::Mat;
                    //*all_matching_results[i].additional_viz_data_id1_to_id2->feature_image = sorted_square_organoid_images[index_of_image_1]->clone();

                    all_matching_results[i].additional_viz_data_id1_to_id2->path_to_feature_image = sorted_image_paths[index_of_image_1];
                    all_matching_results[i].additional_viz_data_id1_to_id2->path_to_candidate_image = sorted_image_paths[index_of_image_2];

                    //all_matching_results[i].additional_viz_data_id1_to_id2->candidate_image = sorted_square_organoid_images[index_of_image_2];
                    all_matching_results[i].additional_viz_data_id1_to_id2->candidate_image = nullptr;//new cv::Mat;
                    //*all_matching_results[i].additional_viz_data_id1_to_id2->candidate_image = sorted_square_organoid_images[index_of_image_2]->clone();

                    all_matching_results[i].additional_viz_data_id1_to_id2->features = all_feature_vectors[index_of_image_1].features;
                    all_matching_results[i].additional_viz_data_id1_to_id2->assigned_candidates = all_feature_vectors[index_of_image_2].candidates;


                    all_matching_results[i].additional_viz_data_id2_to_id1->center_feature_image = sorted_image_centers[index_of_image_2];
                    all_matching_results[i].additional_viz_data_id2_to_id1->center_candidate_image = sorted_image_centers[index_of_image_1];

                    all_matching_results[i].additional_viz_data_id2_to_id1->offset_feature_image = sorted_offsets_from_squaring[index_of_image_2];
                    all_matching_results[i].additional_viz_data_id2_to_id1->offset_candidate_image = sorted_offsets_from_squaring[index_of_image_1];

                    //all_matching_results[i].additional_viz_data_id2_to_id1->feature_image = sorted_square_organoid_images[index_of_image_2];
                    all_matching_results[i].additional_viz_data_id2_to_id1->feature_image = nullptr;//new cv::Mat;
                    //*all_matching_results[i].additional_viz_data_id2_to_id1->feature_image = sorted_square_organoid_images[index_of_image_2]->clone();

                    all_matching_results[i].additional_viz_data_id2_to_id1->path_to_feature_image = sorted_image_paths[index_of_image_2];
                    all_matching_results[i].additional_viz_data_id2_to_id1->path_to_candidate_image = sorted_image_paths[index_of_image_1];

                    //all_matching_results[i].additional_viz_data_id2_to_id1->candidate_image = sorted_square_organoid_images[index_of_image_1];
                    all_matching_results[i].additional_viz_data_id2_to_id1->candidate_image = nullptr;//new cv::Mat;
                    //*all_matching_results[i].additional_viz_data_id2_to_id1->candidate_image = sorted_square_organoid_images[index_of_image_1]->clone();

                    all_matching_results[i].additional_viz_data_id2_to_id1->features = all_feature_vectors[index_of_image_2].features;
                    all_matching_results[i].additional_viz_data_id2_to_id1->assigned_candidates = all_feature_vectors[index_of_image_1].candidates;
                }
            }


        }else{


            // begin of learning loop

            //size_t hash = std::hash<All_Cost_Parameters>()(cost_params);



            std::unordered_map<All_Cost_Parameters,double>  parameters_to_metric_map;
            std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>  parameters_to_prev_parameters_map;

            std::queue<All_Cost_Parameters> cost_parameter_queue;
            cost_parameter_queue.push(cost_params);

            All_Cost_Parameters best_cost_parameters = cost_params;

            //processed_parameters_map[cost_params] = true;
            Learning_Metric_Types metric_type = args.lm_type;
            Model_Parameter_Selection_Strategy search_strategy = args.sel_strat;

            std::vector<Cluster> reference_clustering;

            bool reference_clustering_is_valid = true;
            void* additional_search_data = nullptr;

            if(process_id == 0){
                if(search_strategy == SEL_STRAT_NO_SEARCH){
                    std::cout << "Begin matching calculations" << std::endl;
                }else{
                    read_clustering_from_csv_file(args.reference_clustering_file_name, reference_clustering);
                    
                    additional_search_data = get_initial_search_data_struct_by_search_strat(search_strategy,args);

                    if(!check_if_all_image_numbers_are_contained_in_clustering(reference_clustering,all_feature_vectors)){
                        reference_clustering_is_valid = false;
                    }else{
                        std::cout << "read valid reference clustering: " << args.reference_clustering_file_name << std::endl;
                    }

                    std::cout << "Begin learning loop" << std::endl;
                    std::cout << "Metric used: " << get_string_from_metric_type(metric_type) << std::endl;
                    std::cout << "Search Strategy used: " << get_string_from_search_strategy(search_strategy) << std::endl;
                    std::cout << "Initial search step size: " << args.initial_search_step_size << std::endl;
                }
            }

            double metric_of_best_matching_results = get_initial_value_by_metric_type(metric_type);
            double threshold_for_best_metric = 0.0;

            double time_limit = args.runtime_limit;

            int current_search_order_index = 0;

            float search_step_size = args.initial_search_step_size;

            int iteration_max = 3000;
            int iteration_counter = 0;

            double total_learing_runtime = 0.0;
            double last_step_runtime = 0.0;

            All_Cost_Parameters current_cost_parameters;

            std::ofstream log_file;


            std::chrono::steady_clock::time_point matching_startup_end = std::chrono::steady_clock::now();

            std::cout << "Matching_startup duration:::::::::::: " << std::chrono::duration_cast<std::chrono::microseconds>(matching_startup_end - matching_startup_begin).count() << std::endl; 



            std::chrono::steady_clock::time_point loop_begin = std::chrono::steady_clock::now();
            std::chrono::high_resolution_clock::time_point high_res_loop_begin = std::chrono::high_resolution_clock::now();

            while(!check_loop_termination_condition_and_distribute_cost_parameters_from_queue(process_id,cost_parameter_queue,&current_cost_parameters,iteration_counter,iteration_max,total_learing_runtime,last_step_runtime,args.runtime_limit, reference_clustering_is_valid,loop_begin)){
                
                
                if(process_id == 0 && search_strategy != SEL_STRAT_NO_SEARCH){
                    std::cout << std::endl;
                    std::cout << " starting learing loop iteration: " << iteration_counter << std::endl;
                    print_model_parameters(current_cost_parameters,true); 
                }

                iteration_counter++;

                std::vector<Matching_Result> local_matching_results;
                std::vector<Op_Numbers_And_Runtime> execution_times;

                std::chrono::steady_clock::time_point calc_begin = std::chrono::steady_clock::now();

                calculate_matching_matrix(process_id,total_num_processes,current_cost_parameters,viz_data,all_feature_vectors,local_matching_results,execution_times,&viz_input,args,iteration_counter);

                std::chrono::steady_clock::time_point calc_end = std::chrono::steady_clock::now();
                //std::cout << "p_id: " << process_id << " finished mm calc: " << iteration_counter << std::endl;

                if(process_id == 0){
                    bool new_metric_is_best = false;
                    Metric_and_Threshold cur_metric_and_thresh = add_new_cost_parameters_to_queue(cost_parameter_queue,parameters_to_metric_map,parameters_to_prev_parameters_map,current_cost_parameters,local_matching_results,&all_matching_results,&metric_of_best_matching_results,&threshold_for_best_metric,&best_cost_parameters,search_strategy,metric_type,additional_search_data,new_metric_is_best,reference_clustering,*(args.search_order),current_search_order_index, search_step_size);

                    if(!args.read_matching_results && args.write_matching_results && process_id == 0){
                        output_results(log_file, search_strategy,metric_type,&local_matching_results,current_cost_parameters, all_feature_vectors,name_of_feature_vectors_file,args.read_feature_vector, args, new_metric_is_best, name_of_writen_matching_result_file,cur_metric_and_thresh);
                    }
                }
                
                //std::cout << "p_id: " << process_id << " output results: " << iteration_counter << std::endl;
                //all_matching_results = local_matching_results;

                for(int i = 0; i < local_matching_results.size();i++){
                    Matching_Result current_matching_result = local_matching_results[i];
                    delete(current_matching_result.assignment);
                   
                }

                std::chrono::steady_clock::time_point loop_end = std::chrono::steady_clock::now();
                std::chrono::high_resolution_clock::time_point high_res_loop_end = std::chrono::high_resolution_clock::now();

                double time_to_calc_mm = std::chrono::duration_cast<std::chrono::microseconds>(calc_end - calc_begin).count();
                double time_to_calc_loop_iteration = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_begin).count();
                double time_to_calc_loop_iteration_high_res = std::chrono::duration_cast<std::chrono::microseconds>(high_res_loop_end - high_res_loop_begin).count();

                last_step_runtime = time_to_calc_loop_iteration;
                total_learing_runtime += last_step_runtime;

                loop_begin = std::chrono::steady_clock::now();
                high_res_loop_begin = std::chrono::high_resolution_clock::now();


                if(process_id == 0){
                    std::cout << "mm_time: " << time_to_calc_mm << " loop_time: " << last_step_runtime << " total: " << total_learing_runtime << " limit: " << args.runtime_limit << std::endl;
                    std::cout << "high_res_time: " << time_to_calc_loop_iteration_high_res << std::endl;
                }

            }

            if(search_strategy != SEL_STRAT_NO_SEARCH){
                std::cout << "process_id: " << process_id << " finished learning_queue " << total_learing_runtime << std::endl;
            }

            if(additional_search_data != nullptr){
                free(additional_search_data);
            }

            //std::cout << "best parameters: " << std::endl;
            //print_model_parameters(best_cost_parameters,true);
            //std::cout << "best threshold:" << threshold_for_best_metric << std::endl;
            viz_threshold_slider = viz_threshold_slider_max * threshold_for_best_metric;
            cost_params = best_cost_parameters;

            Metric_and_Threshold best_metric_and_threshold{metric_of_best_matching_results,threshold_for_best_metric};
            finalize_output(log_file,best_cost_parameters,best_metric_and_threshold);

            for(int i = 0; i < reference_clustering.size();i++){
                delete(reference_clustering[i].members);
            }

        }

        std::vector<Cluster> all_clusters_optimal;

        std::vector<Cluster> all_clusters_seconds_set;

        std::vector<Cluster> clusters_with_representatives;

        std::vector<int> image_order_in_matrix_visualization;

        bool use_cluster_image_order = false;


        uint32_t input_flags = 0;
        uint32_t_flip_single_bit(&(input_flags),MATCHING_MATRIX_SYMMETRIZE_BIT_INDEX);

        Matching_Matrix_Input_Struct mm_input_optimal{&sorted_square_organoid_images,&all_matching_results,&all_feature_vectors,individual_image_target_size,MATCHING_VIZ_QUADRATIC_OPTIMAL,input_flags,&image_order_in_matrix_visualization,use_cluster_image_order,cost_params};

        Matching_Matrix_Input_Struct mm_input{&sorted_square_organoid_images,&all_matching_results,&all_feature_vectors,individual_image_target_size,MATCHING_VIZ_QUADRATIC,input_flags,&image_order_in_matrix_visualization,use_cluster_image_order,cost_params};

        Matching_Matrix_Input_Struct mm_input_lin_feature{&sorted_square_organoid_images,&all_matching_results,&all_feature_vectors,individual_image_target_size,MATCHING_VIZ_LIN_FEATURE,input_flags,&image_order_in_matrix_visualization,use_cluster_image_order,cost_params};
        
        Matching_Matrix_Input_Struct mm_input_lin_candidate{&sorted_square_organoid_images,&all_matching_results,&all_feature_vectors,individual_image_target_size,MATCHING_VIZ_LIN_CANDIDATE,input_flags,&image_order_in_matrix_visualization,use_cluster_image_order,cost_params};
        //show_matching_matrix(&square_organoid_images,&all_matching_results,&all_feature_vectors,individual_image_target_size);
        
        if(process_id == 0 && !args.skip_visualization){

            cv::namedWindow(MATCHING_MATRIX_WINDOW_NAME, cv::WINDOW_KEEPRATIO );
            char TrackbarName[50] = "Threshold";
            //sprintf( TrackbarName, "Threshold / %d", viz_threshold_slider_max );
            cv::createTrackbar( TrackbarName, MATCHING_MATRIX_WINDOW_NAME, &viz_threshold_slider, viz_threshold_slider_max, on_trackbar,&mm_input );
            on_trackbar( viz_threshold_slider, &mm_input);
        }

        /*

        cv::namedWindow("Linear Feature Metric Threshold", cv::WINDOW_GUI_NORMAL); // Create Window
        sprintf( TrackbarName, "Threshold / %d", viz_threshold_slider_max );
        cv::createTrackbar( TrackbarName, "Linear Feature Metric Threshold", &viz_threshold_slider, viz_threshold_slider_max, on_trackbar,&mm_input_lin_feature );
        on_trackbar( viz_threshold_slider, &mm_input_lin_feature);

        cv::namedWindow("Linear Candidate Metric Threshold", cv::WINDOW_GUI_NORMAL); // Create Window
        sprintf( TrackbarName, "Threshold / %d", viz_threshold_slider_max );
        cv::createTrackbar( TrackbarName, "Linear Candidate Metric Threshold", &viz_threshold_slider, viz_threshold_slider_max, on_trackbar,&mm_input_lin_candidate );
        on_trackbar( viz_threshold_slider, &mm_input_lin_candidate);

        if (SOLVE_TO_OPTIMALITY_USING_GUROBI && process_id == 0 && !args.skip_visualization) {

            cv::namedWindow("Optimal Quadratic Metric Threshold", cv::WINDOW_GUI_NORMAL); // Create Window
            char TrackbarName[50] = "Threshold Optimal";
            cv::createTrackbar( TrackbarName, "Optimal Quadratic Metric Threshold", &viz_threshold_slider, viz_threshold_slider_max, on_trackbar,&mm_input_optimal );
            on_trackbar( viz_threshold_slider, &mm_input_optimal);
        }
        */

        std::vector<Single_Cluster_Layout>* cluster_layouts_for_visualization = nullptr;

        std::vector<Single_Cluster_Layout>* secondary_cluster_layouts_for_visualization = nullptr; 

        std::vector<Cluster_Representative_Pair> selected_cluster_representatives;

        std::vector<Matching_Result> extracted_matching_results;

        std::vector<Matching_Result> second_extracted_matching_results; 

        if(process_id == 0 && !args.skip_visualization){

            bool use_optimized_clustering = true;
            bool use_gurubi_for_clustering = CALCULATE_CLUSTERING_USING_GUROBI;
            bool use_lazy_constraint = true;
            
            final_visualization:

            print_key_press_prompt();

            int keycode = cv::waitKey(0) & 0xEFFFFF;

            //std::cout << keycode << std::endl;

            if(keycode != KEYCODE_ESC){

                if(keycode == KEYCODE_E){
                    std::vector<Matching_Result> merged_matching_results;
                    merged_matching_results.clear();

                    merge_matching_results(all_matching_results, merged_matching_results,0.0);

                    //std::cout << all_matching_results.size() << " " << merged_matching_results.size() << std::endl;     

                    //double bce = calculate_binary_cross_entropy(merged_matching_results,true);
                }

                if(keycode == KEYCODE_H){
                    std::vector<Cluster> reference_clustering;

                    read_clustering_from_csv_file(args.reference_clustering_file_name, reference_clustering);

                    if(check_if_all_image_numbers_are_contained_in_clustering(reference_clustering,all_feature_vectors)){

                        std::cout << "read valid reference clustering: " << args.reference_clustering_file_name << std::endl;

                        double best_threshold = 0;

                        extract_best_threshold_and_metric_value_from_confusion_matrix(all_matching_results,args.lm_type,&best_threshold,reference_clustering);

                        viz_threshold_slider = best_threshold * viz_threshold_slider_max;

                        cv::setTrackbarPos("Threshold",MATCHING_MATRIX_WINDOW_NAME,viz_threshold_slider);
                        //on_trackbar( viz_threshold_slider, &mm_input);
                        //show_matching_matrix(mm_input.square_organoid_images,mm_input.all_matching_results,mm_input.all_feature_vectors,mm_input.individual_image_size,viz_threshold,mm_input.viz_type,mm_input.flags,&image_order_in_matrix_visualization,use_cluster_image_order, cost_params);
                        
                    }else{
                        std::cout << "clustering file is not a valid for the current image set" << std::endl;
                    }

                }

                if(keycode == KEYCODE_R){

                    std::filesystem::path path_to_comparison_file = get_data_folder_path();

                    path_to_comparison_file.append("clustering_results");

                    path_to_comparison_file.append("pairwise_results");

                    path_to_comparison_file.append(PAIRWISE_COMPARISON_FILENAME);

                    std::vector<Matching_Result> pairwise_truth;
                    read_pairwise_comparison_as_matching_results(path_to_comparison_file,pairwise_truth);

                    std::vector<Cluster> reference_clustering;

                    read_clustering_from_csv_file(args.reference_clustering_file_name, reference_clustering);

                    std::vector<Matching_Result> merged_pairwise_truth;
                    merge_matching_results(pairwise_truth,merged_pairwise_truth,0.0);


                    std::vector<TPR_FPR_Tuple> all_tpr_fpr_tuples;
                    all_tpr_fpr_tuples.clear();



                    std::filesystem::path output_file_path = get_data_folder_path();
                    std::filesystem::path clustered_output_file_path = get_data_folder_path();
                    output_file_path.append("measurements");
                    clustered_output_file_path.append("measurements");

                    std::string clustering_file_name = args.reference_clustering_file_name;
                    
                    std::string output_file_name = "results_";

                    if(USE_HISTOGRAM_COMPARISON_FOR_MATCHING){
                        output_file_name += "histo_";
                    }

                    output_file_name += clustering_file_name;
                    append_current_date_and_time_to_string(output_file_name);
                    output_file_name += "_joins_and_cuts.csv";

                    std::string clustered_output_file_name = "results_";

                    double learned_offset = 0.64;

                    if(USE_HISTOGRAM_COMPARISON_FOR_MATCHING){
                        clustered_output_file_name += "histo_";
                        learned_offset = 0.81;
                    }

                    clustered_output_file_name += clustering_file_name;
                    append_current_date_and_time_to_string(clustered_output_file_name);
                    clustered_output_file_name += "_clustered.csv";


                    output_file_path.append(output_file_name);
                    clustered_output_file_path.append(clustered_output_file_name);

                    std::ofstream output_file;
                    std::ofstream clustering_output_file;

                    output_file.open(output_file_path);
                    clustering_output_file.open(clustered_output_file_path);

                    output_file << args.feature_vector_file_name << std::endl;
                    output_file << args.matching_results_file_name << std::endl;
                    output_file << args.reference_clustering_file_name << std::endl;
                    output_file << std::endl;
                    output_file << "threshold,precision_cuts,recall_cuts,precision_joins,recall_joins,F1,F1_joins,F1_cuts" << std::endl;

                    clustering_output_file << args.feature_vector_file_name << std::endl;
                    clustering_output_file << args.matching_results_file_name << std::endl;
                    clustering_output_file << args.reference_clustering_file_name << std::endl;
                    clustering_output_file << std::endl;
                    clustering_output_file << "threshold,precision_cuts,recall_cuts,precision_joins,recall_joins,RI,VI,VI_false_cuts,VI_false_joins,runtime" << std::endl;

                    std::vector<TPR_FPR_Tuple> tpr_fpr_tuples_using_clustering;

                    bool normalize_intervall = false;


                    int loop_upper = 100 + (100 * learned_offset);

                    if(normalize_intervall){
                        loop_upper = 100 + (100 * (learned_offset * 2.0 - 1.0));
                    }

                    for(int thresh = -100; thresh <= loop_upper; thresh++){                        

                        double current_threshold = (double)thresh/100;

                        
                        std::vector<Matching_Result> merged_matching_results;
                        merged_matching_results.clear();


                        merge_matching_results(all_matching_results, merged_matching_results,current_threshold,normalize_intervall);


                        Confusion_Matrix confusion_matrix_for_current_threshold;

                        calculate_confusion_matrix(confusion_matrix_for_current_threshold,merged_matching_results,current_threshold,reference_clustering);

                        TPR_FPR_Tuple current_tpr_fpr_tuple = get_tpr_fpr_tuple_from_confusion_matrix(confusion_matrix_for_current_threshold, current_threshold);
                        all_tpr_fpr_tuples.push_back(current_tpr_fpr_tuple);

                        std::vector<Cluster> clustering_of_current_threshold;


                        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

                        //calculate_clustering_using_gurobi_with_lazy_constraints(merged_matching_results_with_log_odds,all_feature_vectors,clustering_of_current_threshold,current_threshold);
                        calculate_clustering_using_gurobi_with_lazy_constraints(merged_matching_results,all_feature_vectors,clustering_of_current_threshold,current_threshold);

                        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                        std::cout << "Clustering for offset: "<< current_threshold << " took: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " seconds to calculate" << std::endl;
                        double clustering_time = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
                        //write_clusters_to_csv_file(&clustering_of_current_threshold,"",0.0);


                        Confusion_Matrix confusion_matrix_using_clustering;
                        calculate_confusion_matrix_from_clustering(confusion_matrix_using_clustering,reference_clustering,merged_matching_results,current_threshold,clustering_of_current_threshold);

                        //calculate_confusion_matrix_of_clustering_to_pairwise_truth(confusion_matrix_using_clustering,merged_pairwise_truth,current_threshold,clustering_of_current_threshold);

                        TPR_FPR_Tuple current_tpr_fpr_tuple_for_clustering = get_tpr_fpr_tuple_from_confusion_matrix(confusion_matrix_using_clustering, current_threshold);
                        //tpr_fpr_tuples_using_clustering.push_back(current_tpr_fpr_tuple_for_clustering);

                        clustering_output_file << current_threshold << ",";
                        calculate_variation_of_information_for_clusterings(reference_clustering,clustering_of_current_threshold,clustering_output_file,clustering_time);

                        output_file << current_threshold << "," << current_tpr_fpr_tuple.prec_cuts << "," << current_tpr_fpr_tuple.recall_cuts << "," << current_tpr_fpr_tuple.prec_joins << "," << current_tpr_fpr_tuple.recall_joins << "," << current_tpr_fpr_tuple.f1_score << "," << current_tpr_fpr_tuple.f1_joins << "," << current_tpr_fpr_tuple.f1_cuts << std::endl;

                    }

                    display_roc_curves(all_tpr_fpr_tuples,"Matchings");
                    
                    write_ROC_to_csv_file("roc.csv",all_tpr_fpr_tuples);

                    output_file.close();
                    clustering_output_file.close();
                }

                if(keycode == KEYCODE_J){
                    use_lazy_constraint = !use_lazy_constraint;
                    std::cout << "switched use_lazy_constraint to: " << use_lazy_constraint << std::endl;
                }

                if(keycode == KEYCODE_K){
                    use_optimized_clustering = !use_optimized_clustering;
                    std::cout << "switched use_optimized_clustering to: " << use_optimized_clustering << std::endl;
                }

                if(keycode == KEYCODE_L){
                    use_gurubi_for_clustering = !use_gurubi_for_clustering;
                    std::cout << "switched use_gurubi_for_clustering to: " << use_gurubi_for_clustering << std::endl;
                }

                if(keycode == KEYCODE_W){
                    write_clusters_to_csv_file(&all_clusters_optimal,name_of_writen_matching_result_file,value_of_used_cluster_threshold);
                }

                if(keycode == KEYCODE_V){
                    if(mm_input.image_order->size() > 0){
                        use_cluster_image_order = !use_cluster_image_order;
                        mm_input.use_image_order = use_cluster_image_order;

                        //std::cout << use_cluster_image_order << " " << mm_input.use_image_order << std::endl;
                        update_selected_cells_after_image_order_switch(use_cluster_image_order);
                        
                        show_matching_matrix(mm_input.square_organoid_images,mm_input.all_matching_results,mm_input.all_feature_vectors,mm_input.individual_image_size,viz_threshold,mm_input.viz_type,mm_input.flags,&image_order_in_matrix_visualization,use_cluster_image_order, cost_params);

                        if(cluster_layouts_for_visualization != nullptr){
                            show_clustering(&sorted_square_organoid_images,&all_clusters_optimal,&all_matching_results,&all_feature_vectors,CLUSTERING_VIEW_INDIVIDUAL_SQUARE_SIZE,cluster_layouts_for_visualization,cost_params);
                        }
                    }
                }
                
                if(keycode == KEYCODE_C){

                    double clustering_threshold = (double)viz_threshold_slider/viz_threshold_slider_max;

                    std::vector<Matching_Result> merged_matching_results;
                    merged_matching_results.clear();

                    merge_matching_results(all_matching_results, merged_matching_results,clustering_threshold);

                    
                    extracted_matching_results.clear();

                    std::vector<int> image_sets_to_extract;
                    image_sets_to_extract.clear();
                    image_sets_to_extract.push_back(1);

                    extract_matching_results_by_image_set_numbers(merged_matching_results,extracted_matching_results,image_sets_to_extract,BOTH_MATCH);

                    all_clusters_optimal.clear();

                    value_of_used_cluster_threshold = clustering_threshold;
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

                    if(!use_gurubi_for_clustering){
                        //calculate_clustering(merged_matching_results,all_feature_vectors,all_clusters_optimal,use_optimized_clustering);
                        //calculate_clustering(extracted_matching_results,all_feature_vectors,all_clusters_optimal,use_optimized_clustering);
                        calculate_clustering_using_graph_implementation(extracted_matching_results,all_feature_vectors,all_clusters_optimal);
                    }else{
                        
                        if(use_lazy_constraint){
                            calculate_clustering_using_gurobi_with_lazy_constraints(merged_matching_results,all_feature_vectors,all_clusters_optimal,clustering_threshold);
                        }else{
                            calculate_clustering_using_gurobi(merged_matching_results,all_feature_vectors,all_clusters_optimal,clustering_threshold);
                        }
                        
                    }

                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    std::cout << "Clustering took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [µs] to calculate" << std::endl;

                    reset_clustering_visualization_selection();
                    show_clustering(&sorted_square_organoid_images,&all_clusters_optimal,&extracted_matching_results,&all_feature_vectors,CLUSTERING_VIEW_INDIVIDUAL_SQUARE_SIZE,cluster_layouts_for_visualization, cost_params);

                    adapt_image_order_to_clusters(image_order_in_matrix_visualization,all_clusters_optimal,all_feature_vectors);

                    if(use_cluster_image_order){
                        show_matching_matrix(mm_input.square_organoid_images,mm_input.all_matching_results,mm_input.all_feature_vectors,mm_input.individual_image_size,viz_threshold,mm_input.viz_type,mm_input.flags,&image_order_in_matrix_visualization,use_cluster_image_order, cost_params);
                    }

                    //write_clusters_to_csv_file(&all_clusters_optimal,name_of_writen_matching_result_file,value_of_used_cluster_threshold);
                }

                if(keycode == KEYCODE_F){

                    double clustering_threshold = (double)viz_threshold_slider/viz_threshold_slider_max;

                    std::vector<Matching_Result> merged_matching_results;
                    merged_matching_results.clear();

                    merge_matching_results(all_matching_results, merged_matching_results,clustering_threshold);

                    extracted_matching_results.clear();

                    std::vector<int> image_sets_to_extract;
                    image_sets_to_extract.clear();
                    image_sets_to_extract.push_back(1);

                    extract_matching_results_by_image_set_numbers(merged_matching_results,extracted_matching_results,image_sets_to_extract,BOTH_MATCH);


                    second_extracted_matching_results.clear();

                    std::vector<int> second_results_to_extract;
                    second_results_to_extract.clear();
                    second_results_to_extract.push_back(2);

                    extract_matching_results_by_image_set_numbers(merged_matching_results,second_extracted_matching_results,second_results_to_extract,BOTH_MATCH);

                    all_clusters_optimal.clear();
                    clusters_with_representatives.clear();
                    all_clusters_seconds_set.clear();

                    value_of_used_cluster_threshold = clustering_threshold;
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

                    if(!use_gurubi_for_clustering){
                        //calculate_clustering(merged_matching_results,all_feature_vectors,all_clusters_optimal,use_optimized_clustering);
                        calculate_clustering(extracted_matching_results,all_feature_vectors,all_clusters_optimal,use_optimized_clustering);
                        calculate_clustering(second_extracted_matching_results,all_feature_vectors,all_clusters_seconds_set,use_optimized_clustering);
                    }else{

                        if(use_lazy_constraint){
                            calculate_clustering_using_gurobi_with_lazy_constraints(extracted_matching_results,all_feature_vectors,all_clusters_optimal,clustering_threshold);
                            calculate_clustering_using_gurobi_with_lazy_constraints(second_extracted_matching_results,all_feature_vectors,all_clusters_seconds_set,clustering_threshold);
                        }else{
                            calculate_clustering_using_gurobi(extracted_matching_results,all_feature_vectors,all_clusters_optimal,clustering_threshold);
                            calculate_clustering_using_gurobi(second_extracted_matching_results,all_feature_vectors,all_clusters_seconds_set,clustering_threshold);
                        }
                    }

                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    std::cout << "Clustering took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [µs] to calculate" << std::endl;


                    std::vector<Matching_Result> matching_results_using_cluster_representatives;
                    

                    select_cluster_representatives_and_update_matching_results(merged_matching_results,all_clusters_optimal,matching_results_using_cluster_representatives, selected_cluster_representatives);

                    calculate_clustering(matching_results_using_cluster_representatives,all_feature_vectors,clusters_with_representatives,use_optimized_clustering);

                    reset_clustering_visualization_selection();
                    show_clustering(&sorted_square_organoid_images,&all_clusters_optimal,&extracted_matching_results,&all_feature_vectors,CLUSTERING_VIEW_INDIVIDUAL_SQUARE_SIZE,cluster_layouts_for_visualization, cost_params,PRIMARY_CLUSTERING_WINDOW_IMG_ID , &selected_cluster_representatives);
                    show_clustering(&sorted_square_organoid_images,&all_clusters_seconds_set,&second_extracted_matching_results,&all_feature_vectors,CLUSTERING_VIEW_INDIVIDUAL_SQUARE_SIZE,cluster_layouts_for_visualization, cost_params,SECONDARY_CLUSTERING_WINDOW_IMG_ID , &selected_cluster_representatives);
                    show_clustering(&sorted_square_organoid_images,&clusters_with_representatives,&extracted_matching_results,&all_feature_vectors,CLUSTERING_VIEW_INDIVIDUAL_SQUARE_SIZE,secondary_cluster_layouts_for_visualization, cost_params, COMBINED_CLUSTERING_WINDOW_IMG_ID, &selected_cluster_representatives);

                    //std::cout << "finished show clustering" << std::endl;

                    adapt_image_order_to_clusters(image_order_in_matrix_visualization,all_clusters_optimal,all_feature_vectors);

                    //std::cout << "finished adapt image order" << std::endl;

                    if(use_cluster_image_order){
                        show_matching_matrix(mm_input.square_organoid_images,mm_input.all_matching_results,mm_input.all_feature_vectors,mm_input.individual_image_size,viz_threshold,mm_input.viz_type,mm_input.flags,&image_order_in_matrix_visualization,use_cluster_image_order, cost_params);
                    }
                }

                if(keycode == KEYCODE_P){
                    uint32_t_set_single_bit_to_one(&(mm_input.flags),MATCHING_MATRIX_WRITE_TO_FILE_BIT_INDEX);
                    show_matching_matrix(mm_input.square_organoid_images,mm_input.all_matching_results,mm_input.all_feature_vectors,mm_input.individual_image_size,viz_threshold,mm_input.viz_type,mm_input.flags,&image_order_in_matrix_visualization,use_cluster_image_order, cost_params);
                }

                if(keycode == KEYCODE_B){
                    uint32_t_flip_single_bit(&(mm_input.flags),MATCHING_MATRIX_SYMMETRIZE_BIT_INDEX);
                    show_matching_matrix(mm_input.square_organoid_images,mm_input.all_matching_results,mm_input.all_feature_vectors,mm_input.individual_image_size,viz_threshold,mm_input.viz_type,mm_input.flags,&image_order_in_matrix_visualization,use_cluster_image_order, cost_params);
                }

                if(keycode == KEYCODE_N){
                    uint32_t_flip_single_bit(&(mm_input.flags),MATCHING_MATRIX_COLORGRADIENT_BIT_INDEX);
                    show_matching_matrix(mm_input.square_organoid_images,mm_input.all_matching_results,mm_input.all_feature_vectors,mm_input.individual_image_size,viz_threshold,mm_input.viz_type,mm_input.flags,&image_order_in_matrix_visualization,use_cluster_image_order, cost_params);
                }

                if(keycode == KEYCODE_M){
                    uint32_t_flip_single_bit(&(mm_input.flags),MATCHING_MATRIX_NUMBER_DISPLAY_BIT_INDEX);
                    show_matching_matrix(mm_input.square_organoid_images,mm_input.all_matching_results,mm_input.all_feature_vectors,mm_input.individual_image_size,viz_threshold,mm_input.viz_type,mm_input.flags,&image_order_in_matrix_visualization,use_cluster_image_order, cost_params);
                }

                goto final_visualization;
            }
        }else if(!SKIP_CLUSTERING){
            // only calculate clustering with fixed settings and threshold

            std::cout << "begin of clustering calculation "<< std::endl;

            bool use_gurubi_for_clustering = false;
            bool use_lazy_constraint = true;
            bool use_optimized_clustering = true;

            double clustering_threshold = 0.64;

            std::vector<Matching_Result> merged_matching_results;
            merged_matching_results.clear();

            std::cout << all_feature_vectors.size() << std::endl;

            merge_matching_results_from_ordered_results(all_matching_results, merged_matching_results,clustering_threshold,all_feature_vectors.size());

            
            extracted_matching_results.clear();

            all_clusters_optimal.clear();

            value_of_used_cluster_threshold = clustering_threshold;

            calculate_clustering_using_graph_implementation(merged_matching_results,all_feature_vectors,all_clusters_optimal);

            write_clusters_to_csv_file(&all_clusters_optimal,name_of_writen_matching_result_file,value_of_used_cluster_threshold);
        }

        if(cluster_layouts_for_visualization != nullptr){
            delete cluster_layouts_for_visualization;
        }

    }


    for(int i = 0; i < all_feature_vectors.size();i++){
        Image_Features_Pair current_ifp = all_feature_vectors[i];
        delete(current_ifp.features);
        delete(current_ifp.candidates);
    }

    for(int i = 0; i < square_organoid_images.size();i++){
        cv::Mat* current_img = square_organoid_images[i];
        delete(current_img);
    }

    for(int i = 0; i < all_matching_results.size();i++){
        Matching_Result current_matching_result = all_matching_results[i];

        if(current_matching_result.additional_viz_data_id1_to_id2 != nullptr){
            //delete(current_matching_result.additional_viz_data_id1_to_id2->assignment);
            if(current_matching_result.additional_viz_data_id1_to_id2->feature_image != nullptr){
                delete(current_matching_result.additional_viz_data_id1_to_id2->feature_image);
            }

            if(current_matching_result.additional_viz_data_id1_to_id2->candidate_image != nullptr){
                delete(current_matching_result.additional_viz_data_id1_to_id2->candidate_image);
            }
        }

        if(current_matching_result.additional_viz_data_id2_to_id1 != nullptr){
            //delete(current_matching_result.additional_viz_data_id2_to_id1->assignment);
            if(current_matching_result.additional_viz_data_id2_to_id1->feature_image != nullptr){
                delete(current_matching_result.additional_viz_data_id2_to_id1->feature_image);
            }

            if(current_matching_result.additional_viz_data_id2_to_id1->candidate_image != nullptr){
                delete(current_matching_result.additional_viz_data_id2_to_id1->candidate_image);
            }
        }

        delete(current_matching_result.additional_viz_data_id1_to_id2);
        delete(current_matching_result.additional_viz_data_id2_to_id1);

        //std::cout << "deleting assignment: " << current_matching_result.assignment << std::endl;

        delete(current_matching_result.assignment);
        
    }



    finalize_and_quit_mpi();

}

void reset_line_search_data_struct(Line_Search_Data_Struct* data_struct){

    data_struct->current_search_dim = SD_NOT_SET;
    data_struct->current_search_dir = 0;
    data_struct->dir_is_set = false;

    All_Cost_Parameters empty_parameters{0,0,0,0,0,0,0};

    data_struct->neg_search_dir_candidate = empty_parameters;
    data_struct->pos_search_dir_candidate = empty_parameters;
    data_struct->start_parameters = empty_parameters;
}

void set_new_start_and_search_dim_in_line_search_data_struct(Line_Search_Data_Struct* data_struct, All_Cost_Parameters& new_start, Search_Dim prev_search_dim, std::vector<Search_Dim>& search_order, int& current_search_order_index){
    data_struct->start_parameters = new_start;

    //std::cout << "new dir" << std::endl;

    data_struct->current_search_dim = get_next_dim_in_search_order(search_order, current_search_order_index);

    /*
    switch (prev_search_dim)
    {
    case SD_UNARY_TO_QUADR_WEIGHT:
        data_struct->current_search_dim = SD_COLOR_TO_DIST_WEIGHT;
        break;
    case SD_COLOR_TO_DIST_WEIGHT:
        data_struct->current_search_dim = SD_COLOR_OFFSET;
        break;
    case SD_COLOR_OFFSET:
        data_struct->current_search_dim = SD_DIST_OFFSET;
        break;
    case SD_DIST_OFFSET:
        data_struct->current_search_dim = SD_ANGLE_OFFSET;
        break;
    case SD_ANGLE_OFFSET:
        data_struct->current_search_dim = SD_UNARY_TO_QUADR_WEIGHT;
        break;

    default:
        data_struct->current_search_dim = SD_NOT_SET;
        break;
    }
    */
}

bool is_metric_a_better_then_b(double metric_a, double metric_b, Learning_Metric_Types metric_type){

    switch (metric_type)
    {
    case LM_BCE:
        return metric_a < metric_b;
        break;
    case LM_WBCE:
        return metric_a < metric_b;
        break;    
    case LM_FN_FP_NUM:
        return metric_a < metric_b;
        break;
    case LM_ACC:
        return metric_a > metric_b;
        break;
    case LM_F1_SCORE:
        return metric_a > metric_b;
        break;
    case LM_TPR_TNR_AVG:
        return metric_a > metric_b;
        break;
    case LM_MCC:
        return metric_a > metric_b;
        break;
    default:
        std::cout << "Unknown metric type in is_metric_a_better_then_b" << std::endl;
        return false;
        break;
    }

}

bool check_if_parameters_are_present_in_map(const All_Cost_Parameters& all_cost_parameters, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map){

    if(parameters_to_metric_map.find(all_cost_parameters) == parameters_to_metric_map.end()){
        return false;
    }else{
        return true;
    }

}

void add_step_and_bound_model_parameter(double& parameter, double step, double lower_bound, double upper_bound){

    parameter += step;
    if(parameter < lower_bound){
        parameter = lower_bound;
    }

    if(parameter > upper_bound){
        parameter = upper_bound;
    }

}

All_Cost_Parameters get_new_cost_parameters(All_Cost_Parameters old_cost_parameters, All_Cost_Parameters cost_steps){

    All_Cost_Parameters new_cost_parameters = old_cost_parameters;

    add_step_and_bound_model_parameter(new_cost_parameters.color_offset,cost_steps.color_offset,MODEL_PARAMETER_OFFSET_LOWER_BOUND,MODEL_PARAMETER_OFFSET_UPPER_BOUND);

    add_step_and_bound_model_parameter(new_cost_parameters.dist_offset,cost_steps.dist_offset,MODEL_PARAMETER_OFFSET_LOWER_BOUND,MODEL_PARAMETER_OFFSET_UPPER_BOUND);

    add_step_and_bound_model_parameter(new_cost_parameters.angle_offset,cost_steps.angle_offset,MODEL_PARAMETER_OFFSET_LOWER_BOUND,MODEL_PARAMETER_OFFSET_UPPER_BOUND);

    add_step_and_bound_model_parameter(new_cost_parameters.color_to_dist_weight,cost_steps.color_to_dist_weight,MODEL_PARAMETER_WEIGHT_LOWER_BOUND,MODEL_PARAMETER_WEIGHT_UPPER_BOUND);

    add_step_and_bound_model_parameter(new_cost_parameters.unary_to_to_quadr_weight,cost_steps.unary_to_to_quadr_weight,MODEL_PARAMETER_WEIGHT_LOWER_BOUND,MODEL_PARAMETER_WEIGHT_UPPER_BOUND);

    return new_cost_parameters;
}

All_Cost_Parameters get_new_cost_parameters(All_Cost_Parameters old_cost_parameters, Search_Dim search_dim, double step){

    All_Cost_Parameters new_cost_parameters = old_cost_parameters;

    switch (search_dim)
    {
    case SD_NOT_SET:
        std::cout << "Tried to search along undefinded search dimension!";
        break;
    
    case SD_DIST_OFFSET:
        add_step_and_bound_model_parameter(new_cost_parameters.dist_offset,step,MODEL_PARAMETER_OFFSET_LOWER_BOUND,MODEL_PARAMETER_OFFSET_UPPER_BOUND);
        break;

    case SD_COLOR_OFFSET:
        add_step_and_bound_model_parameter(new_cost_parameters.color_offset,step,MODEL_PARAMETER_OFFSET_LOWER_BOUND,MODEL_PARAMETER_OFFSET_UPPER_BOUND);
        break;

    case SD_ANGLE_OFFSET:
        add_step_and_bound_model_parameter(new_cost_parameters.angle_offset,step,MODEL_PARAMETER_OFFSET_LOWER_BOUND,MODEL_PARAMETER_OFFSET_UPPER_BOUND);
        break;

    case SD_COLOR_TO_DIST_WEIGHT:
        add_step_and_bound_model_parameter(new_cost_parameters.color_to_dist_weight,step,MODEL_PARAMETER_WEIGHT_LOWER_BOUND,MODEL_PARAMETER_WEIGHT_UPPER_BOUND);
        break;

    case SD_UNARY_TO_QUADR_WEIGHT:
        add_step_and_bound_model_parameter(new_cost_parameters.unary_to_to_quadr_weight,step,MODEL_PARAMETER_WEIGHT_LOWER_BOUND,MODEL_PARAMETER_WEIGHT_UPPER_BOUND);
        break;

    default:
        break;
    }

    return new_cost_parameters;
}

bool add_parameters_to_queue_and_map(All_Cost_Parameters new_cost_parameters, All_Cost_Parameters prev_cost_parameter, std::queue<All_Cost_Parameters>& cost_parameter_queue, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map){

    if(!check_if_parameters_are_present_in_map(new_cost_parameters,parameters_to_metric_map)){
        cost_parameter_queue.push(new_cost_parameters);
        parameters_to_prev_parameters_map[new_cost_parameters] = prev_cost_parameter;
        return true;
    }else{
        return false;
    }
}

void choose_next_parameters(std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters& current_parameters, Model_Parameter_Selection_Strategy sel_strat, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, double step_width, void* additional_data, Learning_Metric_Types lm_type, double prev_metric_val , double current_metric_val,std::vector<Search_Dim>& search_order,int& current_search_order_index,All_Cost_Parameters* best_parameters, double metric_of_best_parameters){

    bool was_better_then_prev = is_metric_a_better_then_b(current_metric_val,prev_metric_val,lm_type);

    switch(sel_strat){
        case SEL_STRAT_NO_SEARCH:
            break;

        case SEL_STRAT_SIM_ANN:
            choose_next_parameter_using_simulated_annealing(cost_parameter_queue,current_parameters,parameters_to_metric_map,parameters_to_prev_parameters_map,step_width,additional_data,lm_type,was_better_then_prev,best_parameters,current_metric_val,metric_of_best_parameters);
            break;

        case SEL_STRAT_EXHAUSTIVE_ADJ:
            choose_next_parameter_using_exhaustive_adj_search(cost_parameter_queue,current_parameters,parameters_to_metric_map,parameters_to_prev_parameters_map,step_width,additional_data,lm_type,was_better_then_prev,best_parameters);
            break;

        case SEL_STRAT_LINE_SEARCH:          
            choose_next_parameters_using_line_search(cost_parameter_queue,current_parameters,parameters_to_metric_map,parameters_to_prev_parameters_map,step_width,additional_data,lm_type,was_better_then_prev, search_order, current_search_order_index);
            break;
        default:
            break;
    }

}

void start_line_search_in_next_dim(Line_Search_Data_Struct* data, All_Cost_Parameters start_parameters, double step_width, std::queue<All_Cost_Parameters>& cost_parameter_queue, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, std::vector<Search_Dim>& search_order, int& current_search_order_index){
    Search_Dim current_search_dim = data->current_search_dim;

    reset_line_search_data_struct(data);
    set_new_start_and_search_dim_in_line_search_data_struct(data,start_parameters,current_search_dim,search_order,current_search_order_index);

    add_both_line_search_dirs(data,start_parameters,step_width,cost_parameter_queue,parameters_to_metric_map,parameters_to_prev_parameters_map);
}

void add_both_line_search_dirs(Line_Search_Data_Struct* data, All_Cost_Parameters current_parameters, double step_width, std::queue<All_Cost_Parameters>& cost_parameter_queue, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map){

    All_Cost_Parameters new_params_neg = get_new_cost_parameters(current_parameters,data->current_search_dim,-step_width);
    add_parameters_to_queue_and_map(new_params_neg,current_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);
    data->neg_search_dir_candidate = new_params_neg;

    All_Cost_Parameters new_params_pos = get_new_cost_parameters(current_parameters,data->current_search_dim,step_width);
    add_parameters_to_queue_and_map(new_params_pos,current_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);
    data->pos_search_dir_candidate = new_params_pos;
}

void choose_next_parameter_using_simulated_annealing(std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters& current_parameters, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, double step_width, void* additional_data, Learning_Metric_Types lm_type, bool current_was_better_then_prev, All_Cost_Parameters* best_parameters,double current_metric_value, double metric_of_best_parameters){

    Sim_Annealing_Search_Data_Struct* search_data = (Sim_Annealing_Search_Data_Struct*)additional_data;

    float offset_parameter_rescaled_std_dev = (MODEL_PARAMETER_OFFSET_UPPER_BOUND - MODEL_PARAMETER_OFFSET_LOWER_BOUND) * search_data->std_dev;
    float weight_parameter_rescaled_std_dev = (MODEL_PARAMETER_WEIGHT_UPPER_BOUND - MODEL_PARAMETER_WEIGHT_LOWER_BOUND) * search_data->std_dev;

    All_Cost_Parameters random_parameter_steps;
    random_parameter_steps.cost_double_assignment = 0.0;
    random_parameter_steps.cost_mismatching_channels = 0.0;
    random_parameter_steps.angle_offset = sample_gaussian(0.0,offset_parameter_rescaled_std_dev);
    random_parameter_steps.color_offset = sample_gaussian(0.0,offset_parameter_rescaled_std_dev);
    random_parameter_steps.dist_offset = sample_gaussian(0.0,offset_parameter_rescaled_std_dev);
    random_parameter_steps.color_to_dist_weight = sample_gaussian(0.0,weight_parameter_rescaled_std_dev);
    random_parameter_steps.unary_to_to_quadr_weight = sample_gaussian(0.0,weight_parameter_rescaled_std_dev);

    double probability_to_accept = exp(-((search_data->prev_metric - current_metric_value) / search_data->temperature));
    bool new_step_is_accepted = probability_to_accept > sample_uniform_0_1();

    if(search_data->metric_of_best < current_metric_value){
        search_data->num_time_steps_since_last_best = 0;
        search_data->metric_of_best = current_metric_value;
        search_data->temp_at_last_best = search_data->temperature;
        search_data->last_best_parameters = current_parameters;
    }else{
        search_data->num_time_steps_since_last_best++;

        if(search_data->num_time_steps_since_last_best > search_data->restart_time_step_threshold){

            std::cout << "RESET" << std::endl;

            new_step_is_accepted = false;
            current_was_better_then_prev = false;

            search_data->num_time_steps_since_last_best = 0;
            search_data->prev_accepted_parameters = search_data->last_best_parameters;
            search_data->prev_metric = search_data->metric_of_best;

        }
    }

    
    std::cout << std::endl;
    std::cout << "sim step start" << std::endl;
    std::cout << "best_metric: " << search_data->metric_of_best << std::endl;
    std::cout << "best_params: "; print_model_parameters(search_data->last_best_parameters,true);
    std::cout << "steps since last best: " << search_data->num_time_steps_since_last_best << std::endl; 
    std::cout << "temp: " << search_data->temperature << std::endl;
    std::cout << "std_dev: " << search_data->std_dev << std::endl;
    std::cout << "prev_metric: " << search_data->prev_metric << std::endl;
    std::cout << "prev_params: "; print_model_parameters(search_data->prev_accepted_parameters,true); 
    std::cout << "curr metric: " << current_metric_value << std::endl;
    std::cout << "curr_params:"; print_model_parameters(current_parameters,true); 
    std::cout << "prob to accept: " << probability_to_accept << std::endl;
    std::cout << "was accepted: " << new_step_is_accepted << std::endl; 
    

    if(current_was_better_then_prev || new_step_is_accepted){
        All_Cost_Parameters new_params  = get_new_cost_parameters(current_parameters,random_parameter_steps);
        add_parameters_to_queue_and_map(new_params,current_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        //std::cout << "CURRENT WAS BETTER OR NEW IS ACCEPTED" << std::endl;
        //std::cout << "new params: "; print_model_parameters(new_params,true); 

        search_data->prev_metric = current_metric_value;
        search_data->prev_accepted_parameters = current_parameters;
    }else{

        All_Cost_Parameters new_params  = get_new_cost_parameters(search_data->prev_accepted_parameters,random_parameter_steps);
        add_parameters_to_queue_and_map(new_params,search_data->prev_accepted_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        //std::cout << "NOT ACCEPTED" << std::endl;
        //std::cout << "new params: "; print_model_parameters(new_params,true); 
    }

    std::cout << "sim step end" << std::endl;
    std::cout << std::endl;

    search_data->temperature *= search_data->cooling_rate;
}

void choose_next_parameter_using_exhaustive_adj_search(std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters& current_parameters, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, double step_width, void* additional_data, Learning_Metric_Types lm_type, bool current_was_better_then_prev,All_Cost_Parameters* best_parameters){

    Exhaustive_Adj_Search_Data_Struct* search_data = (Exhaustive_Adj_Search_Data_Struct*)additional_data;
    std::cout << "queue len: " << cost_parameter_queue.size() << std::endl;

    if(current_was_better_then_prev){

        std::cout << "found better" << std::endl;
        search_data->new_best_was_found = true;
    }

    if(cost_parameter_queue.empty() && search_data->new_best_was_found){

        std::cout << "adding new candidates" << std::endl;

        search_data->currently_search_for_best_adj = true;
        search_data->new_best_was_found = false;

        All_Cost_Parameters new_params;

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,step_width,0.0,0.0,0.0,0.0});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,-step_width,0.0,0.0,0.0,0.0});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,0.0,step_width,0.0,0.0,0.0});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,0.0,-step_width,0.0,0.0,0.0});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,0.0,0.0,step_width,0.0,0.0});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,0.0,0.0,-step_width,0.0,0.0});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,0.0,0.0,0.0,step_width,0.0});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,0.0,0.0,0.0,-step_width,0.0});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,0.0,0.0,0.0,0.0,step_width});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

        new_params = get_new_cost_parameters(*best_parameters,All_Cost_Parameters{0.0,0.0,0.0,0.0,0.0,0.0,-step_width});
        add_parameters_to_queue_and_map(new_params,*best_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);
    }

}

void choose_next_parameters_using_line_search(std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters& current_parameters, std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, double step_width, void* additional_data, Learning_Metric_Types lm_type, bool current_was_better_then_prev, std::vector<Search_Dim>& search_order, int& current_search_order_index){
    Line_Search_Data_Struct* data = (Line_Search_Data_Struct*)additional_data;

    if(data->current_search_dim == SD_NOT_SET){
        data->current_search_dim = SD_UNARY_TO_QUADR_WEIGHT;
    }

    if(data->dir_is_set){
        if(current_was_better_then_prev){
            All_Cost_Parameters new_params = get_new_cost_parameters(current_parameters,data->current_search_dim,step_width * data->current_search_dir);

            bool new_params_were_added = add_parameters_to_queue_and_map(new_params,current_parameters,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

            if(!new_params_were_added){
                //std::cout << "new params were not added:" << std::endl;
                //print_model_parameters(new_params,true);

                start_line_search_in_next_dim(data,current_parameters,step_width,cost_parameter_queue,parameters_to_metric_map,parameters_to_prev_parameters_map,search_order,current_search_order_index);
            }
        }else{
            All_Cost_Parameters prev_cost_params = parameters_to_prev_parameters_map.at(current_parameters);
            
            start_line_search_in_next_dim(data,prev_cost_params,step_width,cost_parameter_queue,parameters_to_metric_map,parameters_to_prev_parameters_map,search_order,current_search_order_index);
        }
    }else{
        if(!check_if_parameters_are_present_in_map(data->neg_search_dir_candidate,parameters_to_metric_map) && !check_if_parameters_are_present_in_map(data->pos_search_dir_candidate,parameters_to_metric_map)){

            data->start_parameters = current_parameters;

            add_both_line_search_dirs(data,current_parameters,step_width,cost_parameter_queue,parameters_to_metric_map,parameters_to_prev_parameters_map);
            return;
        }

        if(check_if_parameters_are_present_in_map(data->neg_search_dir_candidate,parameters_to_metric_map) != check_if_parameters_are_present_in_map(data->pos_search_dir_candidate,parameters_to_metric_map)){
            //std::cout << "found only one" << std::endl;
            // both are already pushed to the queue but only one has been processed so far.
            return;
        }

        if(check_if_parameters_are_present_in_map(data->neg_search_dir_candidate,parameters_to_metric_map) == check_if_parameters_are_present_in_map(data->pos_search_dir_candidate,parameters_to_metric_map)){

            double metric_of_neg_dir = parameters_to_metric_map.at(data->neg_search_dir_candidate);
            double metric_of_pos_dir = parameters_to_metric_map.at(data->pos_search_dir_candidate);

            double metric_of_start = parameters_to_metric_map.at(data->start_parameters);

            bool neg_dir_better_then_start = is_metric_a_better_then_b(metric_of_neg_dir, metric_of_start, lm_type);
            bool pos_dir_better_then_start = is_metric_a_better_then_b(metric_of_pos_dir, metric_of_start, lm_type);
            bool neg_dir_better_then_pos = is_metric_a_better_then_b(metric_of_neg_dir, metric_of_pos_dir, lm_type);

            if(neg_dir_better_then_start || pos_dir_better_then_start){
                //std::cout << "pos or neg better" << std::endl;

                All_Cost_Parameters better_candidate;

                data->dir_is_set = true;
                if(neg_dir_better_then_pos){
                    data->current_search_dir = -1;
                    better_candidate = data->neg_search_dir_candidate;
                }else{
                    data->current_search_dir = 1;
                    better_candidate = data->pos_search_dir_candidate;
                }

                All_Cost_Parameters new_params = get_new_cost_parameters(better_candidate,data->current_search_dim,step_width * data->current_search_dir);
                bool new_params_were_added = add_parameters_to_queue_and_map(new_params,better_candidate,cost_parameter_queue,parameters_to_prev_parameters_map,parameters_to_metric_map);

                if(!new_params_were_added){
                    //std::cout << "new params were not added:" << std::endl;
                    //print_model_parameters(new_params,true);

                    start_line_search_in_next_dim(data,better_candidate,step_width,cost_parameter_queue,parameters_to_metric_map,parameters_to_prev_parameters_map,search_order,current_search_order_index);
                }
            }else{
                // no improvement in either pos or neg direction
                //std::cout << "neither are better" << std::endl;
                start_line_search_in_next_dim(data,data->start_parameters,step_width,cost_parameter_queue,parameters_to_metric_map,parameters_to_prev_parameters_map,search_order,current_search_order_index);
            }
            

            return;
        }
    }
}

double evaluate_learning_metric(std::vector<Matching_Result>& all_matching_results, Learning_Metric_Types lm_type, const std::vector<Cluster> &clustering, double* threshold_corresponding_to_best_metric){

    std::vector<Matching_Result> merged_matching_results;
    merged_matching_results.clear();

    merge_matching_results(all_matching_results, merged_matching_results,0.0);
    
    double metric = 10000.0;

    switch(lm_type){

        case LM_BCE:
            extract_best_threshold_and_metric_value_from_confusion_matrix(all_matching_results,LM_F1_SCORE,threshold_corresponding_to_best_metric,clustering);
            metric = calculate_binary_cross_entropy(merged_matching_results,false,clustering);
            break;

        case LM_WBCE:
            extract_best_threshold_and_metric_value_from_confusion_matrix(all_matching_results,LM_F1_SCORE,threshold_corresponding_to_best_metric,clustering);
            metric = calculate_binary_cross_entropy(merged_matching_results,true,clustering);
            break;

        case LM_ACC:
            metric = extract_best_threshold_and_metric_value_from_confusion_matrix(all_matching_results,lm_type,threshold_corresponding_to_best_metric,clustering);
            break;
        case LM_F1_SCORE:
            metric = extract_best_threshold_and_metric_value_from_confusion_matrix(all_matching_results,lm_type,threshold_corresponding_to_best_metric,clustering);
            break;
        case LM_TPR_TNR_AVG:
            metric = extract_best_threshold_and_metric_value_from_confusion_matrix(all_matching_results,lm_type,threshold_corresponding_to_best_metric,clustering);
            break;
        case LM_MCC:
            metric = extract_best_threshold_and_metric_value_from_confusion_matrix(all_matching_results,lm_type,threshold_corresponding_to_best_metric,clustering);
            break;
        default:
            metric = calculate_binary_cross_entropy(merged_matching_results,true,clustering);
            break;

    }

    return metric;

}

float extract_best_threshold_and_metric_value_from_confusion_matrix(std::vector<Matching_Result>& all_matching_results, Learning_Metric_Types lm_type, double* best_threshold, const std::vector<Cluster> &clustering){

    //std::vector<TPR_FPR_Tuple> all_tpr_fpr_tuples;
    //all_tpr_fpr_tuples.clear();

    float local_best_metric_value = 0.0;

    if(best_threshold == nullptr){
        std::cout << "WARNING: best_threshold is NULL in extract_best_threshold_and_metric_value_from_confusion_matrix" << std::endl;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::vector<Matching_Result> merged_matching_results;
    merged_matching_results.clear();


    for(int thresh = 0; thresh <= 100; thresh++){                        

        double current_threshold = (double)thresh/100;

        merge_matching_results(all_matching_results, merged_matching_results,current_threshold);

        Confusion_Matrix confusion_matrix_for_current_threshold;

        calculate_confusion_matrix(confusion_matrix_for_current_threshold,merged_matching_results,current_threshold,clustering);

        TPR_FPR_Tuple current_tpr_fpr_tuple = get_tpr_fpr_tuple_from_confusion_matrix(confusion_matrix_for_current_threshold, current_threshold);

        //std::cout << "pj: " << current_tpr_fpr_tuple.prec_joins << " rj: " << current_tpr_fpr_tuple.recall_joins << " pc: " << current_tpr_fpr_tuple.prec_cuts << " rc: " << current_tpr_fpr_tuple.recall_cuts << " RI: " << current_tpr_fpr_tuple.accuracy << std::endl;

        float metric = get_initial_value_by_metric_type(lm_type);

        switch (lm_type)
        {
        case LM_ACC:
            metric = current_tpr_fpr_tuple.accuracy;
            break;

        case LM_F1_SCORE:
            metric = current_tpr_fpr_tuple.f1_score;
            //std::cout << current_threshold << " : " << metric << std::endl;
            break;

        case LM_TPR_TNR_AVG:
            metric = (current_tpr_fpr_tuple.tpr + current_tpr_fpr_tuple.tnr) * 0.5;
            break;

        case LM_MCC:
            metric = current_tpr_fpr_tuple.mcc;
            break;

        default:
            break;
        }

        if(is_metric_a_better_then_b(metric,local_best_metric_value,lm_type)){
            local_best_metric_value = metric;
            if(best_threshold != nullptr){
                *best_threshold = current_threshold;
            }
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time_to_calc_mm = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    //std::cout << "threshold opt took: " << time_to_calc_mm << std::endl;

    return local_best_metric_value;
    /*
    std::sort(all_tpr_fpr_tuples.begin(),all_tpr_fpr_tuples.end(),TPR_FPR_Tuple_Ordering_Compare);

    for(int i = 0; i < all_tpr_fpr_tuples.size();i++){
        std::cout << all_tpr_fpr_tuples[i].tpr << " " << all_tpr_fpr_tuples[i].fpr << std::endl;
    }
    */
}

Metric_and_Threshold add_new_cost_parameters_to_queue(std::queue<All_Cost_Parameters>& cost_parameter_queue, 
                                        std::unordered_map<All_Cost_Parameters,double>& parameters_to_metric_map, 
                                        std::unordered_map<All_Cost_Parameters,All_Cost_Parameters>& parameters_to_prev_parameters_map, 
                                        All_Cost_Parameters& current_cost_parameters, 
                                        std::vector<Matching_Result>& local_matching_results, 
                                        std::vector<Matching_Result>* best_matching_results, 
                                        double* metric_of_best,
                                        double* threshold_of_best, 
                                        All_Cost_Parameters* best_parameters, 
                                        Model_Parameter_Selection_Strategy search_strat, 
                                        Learning_Metric_Types learning_metric_type, 
                                        void* additional_search_data,
                                        bool& new_metric_is_best,
                                        const std::vector<Cluster> &clustering,
                                        std::vector<Search_Dim>& search_order,
                                        int& current_search_order_index,
                                        float step_width){
                                            

    // evaluate the metric for the current_cost_parameters + all_matching_results

    //print_cost_parameters(current_cost_parameters);
    //print_model_parameters(current_cost_parameters,true);

    double best_threshold_of_current_params = 0.0;

    double learning_metric = evaluate_learning_metric(local_matching_results,learning_metric_type,clustering,&best_threshold_of_current_params);

    Metric_and_Threshold return_val;
    return_val.metric = learning_metric;
    return_val.threshold = best_threshold_of_current_params;

    std::cout << "learning_metric: " << learning_metric << std::endl;
    std::cout << "metric of best: " << *metric_of_best << std::endl;

    new_metric_is_best = false;

    if(is_metric_a_better_then_b(learning_metric,*metric_of_best,learning_metric_type)){
        *metric_of_best = learning_metric;
        *best_parameters = current_cost_parameters;

        new_metric_is_best = true;
        
        if(threshold_of_best != nullptr){
            *threshold_of_best = best_threshold_of_current_params;
            //std::cout << "threshold: " << *threshold_of_best << std::endl; 
        }


        if(best_matching_results->size() == 0){
            // best_matching_results has to be initialized

            for(int i = 0; i < local_matching_results.size();i++){
                Matching_Result cur_loc_mr = local_matching_results[i];

                Matching_Result new_mr_in_best = cur_loc_mr;
                new_mr_in_best.assignment = new std::vector<int>;

                *(new_mr_in_best.assignment) = *(cur_loc_mr.assignment);

                best_matching_results->push_back(new_mr_in_best);
            }
        }

        for(int i = 0; i < local_matching_results.size();i++){
            Matching_Result cur_loc_mr = local_matching_results[i];

            // save the pointer the memory allocated for the best assignments
            std::vector<int>* pointer_to_assignment = (*(best_matching_results))[i].assignment;

            //update the values of the matching result
            (*(best_matching_results))[i] = cur_loc_mr;

            //set the pointer and copy the new assignment into the memory
            (*(best_matching_results))[i].assignment = pointer_to_assignment;
            *((*(best_matching_results))[i].assignment) = *(cur_loc_mr.assignment);
        }
    }

    // add to parameters_to_metric_map


    //std::cout << "before parameters_to_metric_map" << std::endl;
    if(parameters_to_metric_map.find(current_cost_parameters) == parameters_to_metric_map.end()){
        parameters_to_metric_map[current_cost_parameters] = learning_metric;
    }else{
        //std::cout << "Parameter were already present in parameters_to_metric_map" << std::endl;
    }
    
    // get prev_parameters from map, get metric from previous 
    All_Cost_Parameters prev_parameters;
    bool found_prev = false;

    if(parameters_to_prev_parameters_map.find(current_cost_parameters) == parameters_to_prev_parameters_map.end()){
        //std::cout << "no prev_parameters were found in parameters_to_prev_parameters_map" << std::endl;
    }else{
        prev_parameters = parameters_to_prev_parameters_map.at(current_cost_parameters);
        found_prev = true;
    }

    double prev_metric = 10000;

    if(found_prev){
        if(parameters_to_metric_map.find(prev_parameters) == parameters_to_metric_map.end()){
            //std::cout << "prev parameter was not present in parameter to metric map" << std::endl;
        }else{
            prev_metric = parameters_to_metric_map[prev_parameters];
        }   
    }

    // compare if new parameters were an improvement

    choose_next_parameters(cost_parameter_queue,current_cost_parameters,search_strat,parameters_to_metric_map,parameters_to_prev_parameters_map,step_width,additional_search_data,learning_metric_type,prev_metric,learning_metric,search_order,current_search_order_index,best_parameters,*metric_of_best);

    return return_val;
}

void calculate_matching_matrix(int process_id, int total_num_processes, All_Cost_Parameters& cost_params, Assignment_Visualization_Data& viz_data, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Matching_Result>& all_matching_results, std::vector<Op_Numbers_And_Runtime>& execution_times, Additional_Viz_Data_Input* viz_input, const Input_Arguments& args, int iteration_counter){

    std::priority_queue<Matching_Calculation_Task,std::vector<Matching_Calculation_Task>,Matching_Calculation_Task_Compare> task_queue;
            std::vector<Task_Vector> task_vectors;

            Task_Vector active_task_vector;
            active_task_vector.tasks = new std::vector<Matching_Calculation_Task>;
            active_task_vector.total_runtime_estimation = 0.0;

            if(process_id == 0){
                Assignment_Visualization_Data* viz_data_address = nullptr;

                if(!args.skip_visualization){
                    viz_data_address = &viz_data;
                }

                fill_task_queue_with_all_tasks(task_queue,all_feature_vectors,viz_data_address,&cost_params);

                subdivide_task_queue_into_task_vectors(task_queue,task_vectors,total_num_processes);
            }

            mpi_distribute_task_vectors(task_vectors, active_task_vector, &cost_params);

            if(process_id == 0){
                free_task_vectors(task_vectors);
            }



            Task_Handler* task_handler = init_new_task_handler(active_task_vector);

            // check how many threads are available and set up the datastructures for each thread

            unsigned int num_hardware_threads = std::thread::hardware_concurrency();
            if(!PARALLELIZE_MATCHING_CALCULATION){
                num_hardware_threads = 1;
            }

            std::cout << "process: " << process_id << " has " << num_hardware_threads << " threads available" << std::endl;


            std::vector<std::thread*> additional_threads;
            additional_threads.resize(num_hardware_threads - 1);

            std::vector<std::vector<Matching_Result>*> partial_matching_results;

            std::vector<std::vector<Op_Numbers_And_Runtime>*> partial_execution_times;

            for(int i = 0; i < num_hardware_threads; i++){
                std::vector<Matching_Result>* new_partial_matching_result_vector = new std::vector<Matching_Result>;

                partial_matching_results.push_back(new_partial_matching_result_vector);

                std::vector<Op_Numbers_And_Runtime>* new_partial_execution_time_vector = new std::vector<Op_Numbers_And_Runtime>;

                partial_execution_times.push_back(new_partial_execution_time_vector);
            }



            // create the threads and process all the task vectors

            for(int thread = 0; thread < num_hardware_threads - 1; thread++){
                //additional_threads[thread] = new std::thread(process_task_vector,thread,&(task_vectors[thread]),partial_matching_results[thread],&all_feature_vectors);
                additional_threads[thread] = new std::thread(process_task_queue_from_task_handler,thread,task_handler,partial_matching_results[thread],partial_execution_times[thread],&all_feature_vectors);
            }

            //process_task_vector(num_hardware_threads-1,&(task_vectors[num_hardware_threads - 1]),partial_matching_results[num_hardware_threads - 1],&all_feature_vectors);
            process_task_queue_from_task_handler(num_hardware_threads-1,task_handler,partial_matching_results[num_hardware_threads - 1],partial_execution_times[num_hardware_threads - 1],&all_feature_vectors);

            for(int thread = 0; thread < num_hardware_threads - 1; thread++){
                additional_threads[thread]->join();
            }

            //std::cout << "Finished matching calculations" << std::endl;

            //after joining the threads we gather up all the results and delete the used datastructures

            for(int i = 0; i < num_hardware_threads; i++){

                std::vector<Matching_Result>* current_partial_matching_result_vector = partial_matching_results[i];

                for(int j = 0; j < current_partial_matching_result_vector->size();j++){
                    all_matching_results.push_back((*current_partial_matching_result_vector)[j]);
                }

                delete(partial_matching_results[i]);

                std::vector<Op_Numbers_And_Runtime>* current_partial_execution_times_vector = partial_execution_times[i];

                for(int j = 0; j < current_partial_execution_times_vector->size();j++){
                    execution_times.push_back((*current_partial_execution_times_vector)[j]);
                }

                delete(current_partial_execution_times_vector);

            }

            delete(active_task_vector.tasks);

            delete_task_handler(&task_handler);

            Additional_Viz_Data_Input* viz_input_address = nullptr;

            if(!args.skip_visualization){
                viz_input_address = viz_input;
            }

            mpi_gather_matching_results(all_matching_results,execution_times,viz_input_address);


            if(args.write_execution_time_measurements && process_id == 0){
                std::string execution_times_file_name = "execution_time_measurements_";
                execution_times_file_name += std::to_string(iteration_counter);
                execution_times_file_name += args.current_time_string;//append_current_date_and_time_to_string(op_a_r_file);
                execution_times_file_name += ".csv";

                write_execution_time_measurements_to_file(execution_times_file_name,execution_times,all_feature_vectors);
            }



}

double calculate_runtime_estimation(int first_feature_vector_size, int second_feature_vector_size){

    int num_ops = 0;

    // i assume here that the runtime is dominated by the runtime for the cyclic greedy search
    // in which i for each feature point explore as many instances as their are candidates, i.e. num_features * num_candidates
    // in the cyclic greedy search i do this exploration once for each feature point as the start point, hence i get num_features * num_features * num_candidates

    //if(first_feature_vector_size < second_feature_vector_size){
        num_ops = first_feature_vector_size * first_feature_vector_size * second_feature_vector_size;
    //}else{
        //num_ops = second_feature_vector_size * second_feature_vector_size * first_feature_vector_size;
    //}

    // for now I have calculated RUNTIME_ESTIMATION_VAR_A and RUNTIME_ESTIMATION_VAR_B by writing the num_ops and time/num_ops to a file and running a exponential regression on it
    // the prefactor therefore gives me an estimation for time/num_ops so to get the time I need to multiply the prefactor with num_ops in the end

    double prefactor =  exp(log(num_ops) * RUNTIME_ESTIMATION_VAR_A + RUNTIME_ESTIMATION_VAR_B);

    //std::cout << first_feature_vector_size << " " << second_feature_vector_size << " runtime: " << num_ops * prefactor << std::endl;

    return num_ops * prefactor;

}

void process_task_vector(int thread_id, Task_Vector* task_vector, std::vector<Matching_Result>* partial_matching_results, std::vector<Image_Features_Pair>* all_feature_vectors){

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(int i = 0; i < task_vector->tasks->size();i++){

        Matching_Calculation_Task current_task = (*(task_vector->tasks))[i];

        int id_1 = current_task.id_1;
        int id_2 = current_task.id_2;

        Matching_Result next_matching_result = find_matching_between_two_organoids((*all_feature_vectors)[id_1].features,(*all_feature_vectors)[id_2].candidates,*(current_task.cost_params),SOLVE_TO_OPTIMALITY_USING_GUROBI,current_task.viz_data,current_task.index_features_in_viz_data,current_task.index_candidates_in_viz_data);
        //next_matching_result = find_matching_between_two_organoids((*all_feature_vectors)[id_2].features,(*all_feature_vectors)[id_1].features,SOLVE_TO_OPTIMALITY_USING_GUROBI);

        next_matching_result.id_1 = (*all_feature_vectors)[id_1].image_number;
        next_matching_result.id_2 = (*all_feature_vectors)[id_2].image_number;
        next_matching_result.set_id_1 = (*all_feature_vectors)[id_1].set_number;
        next_matching_result.set_id_2 = (*all_feature_vectors)[id_2].set_number;
        partial_matching_results->push_back(next_matching_result);

        if (PRINT_SOLVER_PROGRESS) {
            std::cout << "thread " << thread_id << "  finished task: " << i << std::endl;
        }

    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "Thread: " << thread_id << "  took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs] to calculate: " << task_vector->tasks->size() << " tasks with expected time: " << task_vector->total_runtime_estimation << std::endl;
}

void process_task_queue_from_task_handler(int thread_id, Task_Handler* task_handler, std::vector<Matching_Result>* partial_matching_results, std::vector<Op_Numbers_And_Runtime>* partial_execution_times, std::vector<Image_Features_Pair>* all_feature_vectors){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //std::cout << "Thread: " << thread_id << " starts to process tasks" << std::endl;

    Memory_Manager_Fixed_Size* mem_manager = nullptr; 

    int i = 0;

    while(!task_handler->task_queue->empty()){

        //std::cout << "before request" << std::endl;
        Thread_Task next_thread_task = request_task_from_task_handler(task_handler);
        //std::cout << "after request" << std::endl;

        Matching_Calculation_Task current_task;
        
        if(next_thread_task.thread_task_data != nullptr){
            current_task = *((Matching_Calculation_Task*)next_thread_task.thread_task_data);
        }else{
            //std::cout << "Current task was nullptr" << std::endl;
            continue;
        }

        int id_1 = current_task.id_1;
        int id_2 = current_task.id_2;

        size_t size_of_single_instance = sizeof(int) * (*all_feature_vectors)[id_1].features->size();
        if(mem_manager == nullptr){
            size_t initial_number_of_elements = (*all_feature_vectors)[id_1].features->size() * (*all_feature_vectors)[id_2].candidates->size();
            //std::cout << "initial size: " << initial_number_of_elements << std::endl;

            mem_manager = init_fixed_size_memory_manager(size_of_single_instance,initial_number_of_elements);
        }else{
            change_single_elem_size_and_reset_allocation(mem_manager,size_of_single_instance);
        }

        //std::cout << "before matching result" << std::endl;
        Matching_Result next_matching_result = find_matching_between_two_organoids((*all_feature_vectors)[id_1].features,(*all_feature_vectors)[id_2].candidates,*(current_task.cost_params),SOLVE_TO_OPTIMALITY_USING_GUROBI,current_task.viz_data,current_task.index_features_in_viz_data,current_task.index_candidates_in_viz_data,partial_execution_times,mem_manager);
        //next_matching_result = find_matching_between_two_organoids((*all_feature_vectors)[id_2].features,(*all_feature_vectors)[id_1].features,SOLVE_TO_OPTIMALITY_USING_GUROBI);
        //std::cout << "after matching result" << std::endl;

        //delete the persistent thread tasks which we have allocated in the init_task_handler() function
        delete((Matching_Calculation_Task*)next_thread_task.thread_task_data);

        next_matching_result.id_1 = (*all_feature_vectors)[id_1].image_number;
        next_matching_result.id_2 = (*all_feature_vectors)[id_2].image_number;
        next_matching_result.set_id_1 = (*all_feature_vectors)[id_1].set_number;
        next_matching_result.set_id_2 = (*all_feature_vectors)[id_2].set_number;
        partial_matching_results->push_back(next_matching_result);

        if (PRINT_SOLVER_PROGRESS) {
            std::cout << "thread " << thread_id << "  finished task: " << i << std::endl;
        }

        i++;

    }

    if(mem_manager != nullptr){
        destroy_fixed_size_memory_manager(&mem_manager);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Thread: " << thread_id << "  took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs] to calculate: " << i << " tasks" << std::endl;

}

void merge_matching_results_and_convert_similarity_to_log_odds(std::vector<Matching_Result>& original_matching_result, std::vector<Matching_Result>& merged_matching_results, double threshold){

    merge_matching_results(original_matching_result,merged_matching_results,0);

    for(int i = 0; i < merged_matching_results.size();i++){

        Matching_Result current_mr = merged_matching_results[i];

        double rescaled_value = 0;
        if(current_mr.rel_quadr_cost <= threshold){
            rescaled_value = current_mr.rel_quadr_cost / threshold * 0.5;
        }else{
            rescaled_value = ((current_mr.rel_quadr_cost - threshold)/(1.0 - threshold)) * 0.5 + 0.5;
        }

        double log_odds = log(rescaled_value/(1.0 - rescaled_value));

        merged_matching_results[i].rel_quadr_cost = log_odds;

    }

}

void merge_matching_results(std::vector<Matching_Result>& original_matching_result, std::vector<Matching_Result>& merged_matching_results, double threshold, bool normalize_intervall){
    //std::cout << "original size: " << original_matching_result.size() << std::endl;

    for(int i = 0; i < original_matching_result.size();i++){

        Matching_Result current_matching_result = original_matching_result[i];

        double cost_of_matching = current_matching_result.rel_quadr_cost;
        if(normalize_intervall){
            cost_of_matching = (cost_of_matching * 2.0) - 1.0;
            threshold = (threshold * 2.0) - 1.0;
        }

        int id_1 = current_matching_result.id_1;
        int id_2 = current_matching_result.id_2;

        bool found_existing_merged_result = false;

        for(int j = 0; j < merged_matching_results.size();j++){

            Matching_Result existing_merged_result = merged_matching_results[j];



            if((existing_merged_result.id_1 == id_1 && existing_merged_result.id_2 == id_2)){
                

                if(merged_matching_results[j].rel_quadr_cost > cost_of_matching - threshold){
                    //std::cout << "before: " << merged_matching_results[j].rel_quadr_cost << std::endl;
                    merged_matching_results[j].rel_quadr_cost = cost_of_matching - threshold;
                    //std::cout << "after: " << merged_matching_results[j].rel_quadr_cost << " target: " << current_matching_result.rel_quadr_cost - threshold << std::endl;
                }

                //std::cout << "collision" << std::endl;
                //merged_matching_results[j].additional_viz_data_id1_to_id2 = current_matching_result.additional_viz_data_id1_to_id2;
                   
                found_existing_merged_result = true;
                break;
            }

            if((existing_merged_result.id_1 == id_2 && existing_merged_result.id_2 == id_1)){
                

                if(merged_matching_results[j].rel_quadr_cost > cost_of_matching - threshold){
                    //std::cout << "before: " << merged_matching_results[j].rel_quadr_cost << std::endl;
                    merged_matching_results[j].rel_quadr_cost = cost_of_matching - threshold;
                    //std::cout << "after: " << merged_matching_results[j].rel_quadr_cost << " target: " << current_matching_result.rel_quadr_cost - threshold << std::endl;
                }

                merged_matching_results[j].additional_viz_data_id2_to_id1 = current_matching_result.additional_viz_data_id1_to_id2;     
                found_existing_merged_result = true;
                break;
            }
        }

        if(!found_existing_merged_result){
            Matching_Result new_merged_result;

            new_merged_result.id_1 = id_1;
            new_merged_result.id_2 = id_2;
            new_merged_result.set_id_1 = current_matching_result.set_id_1;
            new_merged_result.set_id_2 = current_matching_result.set_id_2;
            new_merged_result.assignment = current_matching_result.assignment;
            new_merged_result.rel_quadr_cost = cost_of_matching - threshold;

            new_merged_result.rel_quadr_cost_optimal = 0.0;
            new_merged_result.linear_cost_per_candidate = 0.0;
            new_merged_result.linear_cost_per_feature = 0.0;
            new_merged_result.additional_viz_data_id1_to_id2 = current_matching_result.additional_viz_data_id1_to_id2;

            merged_matching_results.push_back(new_merged_result);

        }

    }

    /*
    for(int i = 0; i < merged_matching_results.size();i++){

        Matching_Result current_matching_result = merged_matching_results[i];

        std::cout << current_matching_result.id_1 << " " << current_matching_result.id_2 << " " << current_matching_result.rel_quadr_cost << std::endl;

    }
    */

    //std::cout << "after size: " << merged_matching_results.size() << std::endl;

}

void merge_matching_results_from_ordered_results(std::vector<Matching_Result>& original_matching_result, std::vector<Matching_Result>& merged_matching_results, double threshold, int num_elems){

    if(original_matching_result.size() == 0){
        std::cout << "size of original_matching_result was 0 in merge_matching_results_from_ordered_results." << std::endl;
        return;
    }

    std::filesystem::path output_file_path = get_data_folder_path();
    output_file_path.append("cost_matrix.txt");

    std::ofstream output_file;

    //output_file.open(output_file_path);

    // << 200 << std::endl;

        for(int k = 0; k < num_elems;k++){
            for(int j = k+1; j < num_elems;j++){

                int index_of_first_mr = k * num_elems - k + j - 1;
                int index_of_second_mr = j * num_elems - j + k;

                Matching_Result first_mr = original_matching_result[index_of_first_mr];
                Matching_Result second_mr = original_matching_result[index_of_second_mr];

                //std::cout << "first: " << first_mr.id_1 << " " << first_mr.id_2 << "  second: " << second_mr.id_1 << " " << second_mr.id_2 << std::endl;

                double merged_cost = first_mr.rel_quadr_cost;
                if(second_mr.rel_quadr_cost < merged_cost){
                    merged_cost = second_mr.rel_quadr_cost;
                }  

                merged_cost -= threshold;

                int interger_cost = floor(merged_cost * 1000);

                //output_file << interger_cost << " ";

                Matching_Result new_merged_result;

                new_merged_result.id_1 = first_mr.id_1;
                new_merged_result.id_2 = first_mr.id_2;
                new_merged_result.set_id_1 = first_mr.set_id_1;
                new_merged_result.set_id_2 = first_mr.set_id_2;
                new_merged_result.assignment = first_mr.assignment;
                new_merged_result.rel_quadr_cost = merged_cost;

                new_merged_result.rel_quadr_cost_optimal = 0.0;
                new_merged_result.linear_cost_per_candidate = 0.0;
                new_merged_result.linear_cost_per_feature = 0.0;
                new_merged_result.additional_viz_data_id1_to_id2 = first_mr.additional_viz_data_id1_to_id2;

                merged_matching_results.push_back(new_merged_result);

            }
            output_file << std::endl;
        }

        std::cout << merged_matching_results.size() << std::endl;

        //output_file.close();
}

void extract_matching_results_by_image_set_numbers(std::vector<Matching_Result>& all_matching_results, std::vector<Matching_Result>& extracted_matching_results,std::vector<int>& set_numbers, Extraction_By_Image_Set_Number_Mode extraction_mode){

    extracted_matching_results.clear();

    for(int i = 0; i < all_matching_results.size();i++){

        Matching_Result current_mr = all_matching_results[i];

        bool first_matches = false;
        bool second_matches = false;

        for(int j = 0; j < set_numbers.size();j++){
            if(set_numbers[j] == current_mr.set_id_1){
                first_matches = true;
            }

            if(set_numbers[j] == current_mr.set_id_2){
                second_matches = true;
            }

        }

        switch(extraction_mode){
            case BOTH_MATCH:
                if(first_matches && second_matches){
                    extracted_matching_results.push_back(current_mr);
                }
                break;
            case ONLY_ONE_MATCHES:
                if((first_matches && !second_matches) || (!first_matches && second_matches)){
                    extracted_matching_results.push_back(current_mr);
                }
                break;
            case ATLEAST_ONE_MATCHES:
                if(first_matches || second_matches){
                    extracted_matching_results.push_back(current_mr);
                }
                break;
            default:
            break;
        }
    }

    //for(int i = 0; i < extracted_matching_results.size();i++){
    //    std::cout << extracted_matching_results[i].id_1 << " " << extracted_matching_results[i].id_2 << std::endl;
    //}
}

void fill_task_queue_with_all_tasks(std::priority_queue<Matching_Calculation_Task,std::vector<Matching_Calculation_Task>,Matching_Calculation_Task_Compare>& task_queue, std::vector<Image_Features_Pair>& all_feature_vectors, Assignment_Visualization_Data* viz_data, All_Cost_Parameters* cost_params){

    for(int i = 0; i < all_feature_vectors.size();i++){
                for(int j = i+1; j < all_feature_vectors.size();j++){
                    if(i == j){
                        continue;
                    }

                    Matching_Calculation_Task new_task;
                    new_task.cost_params = cost_params;
                    new_task.id_1 = i;
                    new_task.id_2 = j;
                    new_task.index_features_in_viz_data = i;
                    new_task.index_candidates_in_viz_data = j;
                    new_task.viz_data = viz_data;

                    new_task.runtime_estimation = calculate_runtime_estimation(all_feature_vectors[i].features->size(),all_feature_vectors[j].candidates->size());

                    task_queue.push(new_task);

                    new_task.cost_params = cost_params;
                    new_task.id_1 = j;
                    new_task.id_2 = i;
                    new_task.index_features_in_viz_data = j;
                    new_task.index_candidates_in_viz_data = i;
                    new_task.viz_data = viz_data;

                    new_task.runtime_estimation = calculate_runtime_estimation(all_feature_vectors[j].features->size(),all_feature_vectors[i].candidates->size());

                    task_queue.push(new_task);
                }
            }
}

void subdivide_task_queue_into_task_vectors(std::priority_queue<Matching_Calculation_Task,std::vector<Matching_Calculation_Task>,Matching_Calculation_Task_Compare>& task_queue, std::vector<Task_Vector>& all_task_vectors, int num_task_vectors_to_subdivide_into){

    for(int i = 0; i < num_task_vectors_to_subdivide_into; i++){

        Task_Vector new_task_vector;
        new_task_vector.total_runtime_estimation = 0;
        new_task_vector.tasks = new std::vector<Matching_Calculation_Task>;

        all_task_vectors.push_back(new_task_vector);
    }

    while(!task_queue.empty()){
        Matching_Calculation_Task current_task = task_queue.top();
        task_queue.pop();

        //sort the task vectors s.t. the on with the lowest accumulated runtime is first
        std::sort(all_task_vectors.begin(),all_task_vectors.end(),Task_Vector_Compare);

        //add the current task to the first task vector
        all_task_vectors[0].tasks->push_back(current_task);
        all_task_vectors[0].total_runtime_estimation += current_task.runtime_estimation;
    }
}

void free_task_vectors(std::vector<Task_Vector>& task_vectors){

    for(int i = 0; i < task_vectors.size();i++){
        delete(task_vectors[i].tasks);
    }
}


void adapt_image_order_to_clusters(std::vector<int>& image_order, std::vector<Cluster>& all_clusters, std::vector<Image_Features_Pair>& all_feature_vectors){

    image_order.clear();

    std::vector<int> all_cluster_members_flat;
    all_cluster_members_flat.clear();

    for(int cluster_id = 0; cluster_id < all_clusters.size();cluster_id++){

        Cluster current_cluster = all_clusters[cluster_id];
        //std::cout << "cluster " << cluster_id << ": ";

        for(int cluster_member_id = 0;cluster_member_id < current_cluster.members->size();cluster_member_id++){
            all_cluster_members_flat.push_back((*(current_cluster.members))[cluster_member_id]);
            //std::cout << (*(current_cluster.members))[cluster_member_id] << " ";
        }
        //std::cout << std::endl;
    }
    //std::cout << "new order: "; 

    //image_order.resize(all_feature_vectors.size());

    //std::cout << all_cluster_members_flat.size() << " " << all_feature_vectors.size() << std::endl;

    if(all_cluster_members_flat.size() != all_feature_vectors.size()){

        std::cout << "different_size" << std::endl;

        for(int i = 0; i < all_feature_vectors.size();i++){
            image_order.push_back(i);
            //std::cout << i << " " << std::endl;
        }
    }else{
        for(int i = 0; i < all_cluster_members_flat.size();i++){
        int image_num = all_cluster_members_flat[i];

        for(int j = 0; j < all_feature_vectors.size();j++){

            if(image_num == all_feature_vectors[j].image_number){
                image_order.push_back(j);
                //std::cout << j << " " << std::endl;
            }
        }
    }

    }


    
    //std::cout << std::endl;
}


void fill_op_number_struct(Op_Numbers_And_Runtime& op_number_struct, int num_features, int num_candidates, Algorithm_Type algo_type, double runtime, double result){

    op_number_struct.num_features = num_features;
    op_number_struct.num_candiates = num_candidates;
    op_number_struct.runtime[algo_type] = runtime;
    op_number_struct.normalized_results[algo_type] = result; 
}


int extract_all_images_from_folder(std::filesystem::path image_folder_path, const Input_Arguments args, std::filesystem::path feature_vector_file_path,
                                                                                std::vector<cv::Mat*>& square_organoid_images, 
                                                                                std::vector<Image_Features_Pair>& all_feature_vectors,
                                                                                std::vector<int>& image_number_index,
                                                                                std::vector<cv::Point>& image_centers,
                                                                                std::vector<cv::Point>& offsets_from_squaring,
                                                                                std::vector<std::filesystem::path>& image_paths,
                                                                                int set_number, int img_number_offset){


    int largest_img_number = 0;

    for (const auto & entry : std::filesystem::directory_iterator(image_folder_path)){
            if(!entry.is_directory()){
                
                std::string filename = entry.path().filename().string();

                //std::cout << filename << std::endl;

                if(check_if_file_is_mask_image(filename)){
                    continue;
                }

                Image_Features_Pair new_ifp;
                new_ifp.image_path = entry.path();
                new_ifp.image_number = get_image_number_from_file_name(filename) + img_number_offset;

                if(new_ifp.image_number > largest_img_number){
                    largest_img_number = new_ifp.image_number;
                }

                new_ifp.set_number = set_number;

                new_ifp.features = new std::vector<Feature_Point_Data>;
                new_ifp.candidates = new std::vector<Feature_Point_Data>;

                cv::Mat* new_organoid_image = new cv::Mat;

                cv::Point2i new_org_img_center = extract_features_from_single_organoid_image(entry,*(new_ifp.features),*(new_ifp.candidates),new_organoid_image,args.read_feature_vector,feature_vector_file_path,img_number_offset);


                //std::cout << new_ifp.image_number << " " << new_ifp.features->size() << " " << new_ifp.candidates->size() << " " << set_number << std::endl;
                //std::cout << "center: " << new_org_img_center << std::endl;

                int max_dim = std::max<int>(new_organoid_image->cols,new_organoid_image->rows);

                cv::Size square_img_size(max_dim,max_dim);

                int elem_size = new_organoid_image->elemSize1();

                //std::cout << "ELEM SIZE: " << elem_size << " " << new_organoid_image->elemSize() << std::endl;

                int square_img_type = CV_16UC3;
                if( elem_size != 2 && elem_size != 1){
                    std::cout << "unsupported elem size: " << elem_size << std::endl;
                }

                if(new_organoid_image->channels() == 4){
                    cv::cvtColor(*new_organoid_image,*new_organoid_image,cv::COLOR_BGRA2BGR);
                }

                cv::Mat new_square_organoid_image = cv::Mat::zeros(square_img_size,square_img_type);

                int col_offset = (max_dim - new_organoid_image->cols) >> 1;
                int row_offset = (max_dim - new_organoid_image->rows) >> 1;

                cv::Point2i new_offset_from_squaring;
                new_offset_from_squaring.x = col_offset;
                new_offset_from_squaring.y = row_offset;
                //std::cout << new_org_img_center << std::endl;

                //std::cout << "OFFSET FROM SQUARING: " << new_offset_from_squaring << std::endl;
                double scaling_factor = (double)(1 << 16) / (double)(1 << 8);
                
                for(int local_row = 0; local_row < new_organoid_image->rows;local_row++){
                    for(int local_col = 0; local_col < new_organoid_image->cols;local_col++){

                        int shifted_row = local_row + row_offset;
                        int shifted_col = local_col + col_offset;


                        if(elem_size == 1){
                            new_square_organoid_image.at<cv::Vec3w>(shifted_row,shifted_col) = cv::Vec3w(scaling_factor,scaling_factor,scaling_factor).mul(new_organoid_image->at<cv::Vec3b>(local_row,local_col));
                            //new_square_organoid_image.at<cv::Vec3b>(shifted_row,shifted_col) = new_organoid_image->at<cv::Vec3b>(local_row,local_col);
                        }else if(elem_size == 2){
                            new_square_organoid_image.at<cv::Vec3w>(shifted_row,shifted_col) = new_organoid_image->at<cv::Vec3w>(local_row,local_col);
                        }

                    }
                }

                //new_organoid_image->release();

                *new_organoid_image = new_square_organoid_image;

                square_organoid_images.push_back(new_organoid_image);
                all_feature_vectors.push_back(new_ifp);
                image_number_index.push_back(new_ifp.image_number);
                image_centers.push_back(new_org_img_center);
                offsets_from_squaring.push_back(new_offset_from_squaring);
                image_paths.push_back(new_ifp.image_path);

                //std::cout << "SIZE:" << new_ifp.features->size() << std::endl;
                //process_single_organoid_image(entry, exp_data_handler);
                //subdivide_into_single_organoid_images(entry);
            }     
        }

    return largest_img_number;
}

void read_image_numbers_and_feature_vectors_from_file(std::filesystem::path file_path, std::vector<Image_Features_Pair>& all_feature_vectors, int set_number){
    std::vector<int> image_numbers_in_feature_file;

    read_image_numbers_from_feature_vector_file(file_path,image_numbers_in_feature_file);

    for(int i = 0; i < image_numbers_in_feature_file.size(); i++){

        int current_image_number = image_numbers_in_feature_file[i];

        Image_Features_Pair new_ifp;
        new_ifp.image_path = "";
        new_ifp.image_number = current_image_number;
        new_ifp.set_number = set_number;

        new_ifp.features = new std::vector<Feature_Point_Data>;
        new_ifp.candidates = new std::vector<Feature_Point_Data>;

        read_feature_vector_from_file(file_path,current_image_number,*(new_ifp.features),*(new_ifp.candidates));

        all_feature_vectors.push_back(new_ifp);
    }

}

void set_image_set_numbers_in_matching_results(std::vector<Matching_Result>& all_matching_results, std::filesystem::path image_base_folder, const Input_Arguments& input_args, int img_num_offset_set_2){

    std::vector<int> image_numbers_in_set_1;
    image_numbers_in_set_1.clear();

    std::vector<int> image_numbers_in_set_2;
    image_numbers_in_set_2.clear();

    std::filesystem::path img_set_1_path = image_base_folder;
    img_set_1_path.append(input_args.image_set_name);

    read_image_numbers_from_image_set_folder(img_set_1_path,image_numbers_in_set_1);

    if(input_args.use_secondary_image_set){
        std::filesystem::path img_set_2_path = image_base_folder;
        img_set_2_path.append(input_args.second_image_set_name);
        read_image_numbers_from_image_set_folder(img_set_2_path,image_numbers_in_set_2);
    }

    for(int i = 0; i < all_matching_results.size();i++){
        Matching_Result current_mr = all_matching_results[i];

        if(check_if_image_number_is_contained_in_vector(image_numbers_in_set_1,current_mr.id_1)){
            all_matching_results[i].set_id_1 = 1;

        }else{
            if(check_if_image_number_is_contained_in_vector(image_numbers_in_set_2,current_mr.id_1 - img_num_offset_set_2)){
                all_matching_results[i].set_id_1 = 2;
            }
        }

        if(check_if_image_number_is_contained_in_vector(image_numbers_in_set_1,current_mr.id_2)){
            all_matching_results[i].set_id_2 = 1;
        }else{
            if(check_if_image_number_is_contained_in_vector(image_numbers_in_set_2,current_mr.id_2 - img_num_offset_set_2)){
                all_matching_results[i].set_id_2 = 2;
            }
        }

        //std::cout << all_matching_results[i].id_1 << " " << all_matching_results[i].id_2 << " " << all_matching_results[i].set_id_1 << " " << all_matching_results[i].set_id_2 << std::endl;
    }

}

void select_cluster_representatives_and_update_matching_results(std::vector<Matching_Result>& all_matching_results, std::vector<Cluster>& all_current_clusters, std::vector<Matching_Result>& new_matching_results, std::vector<Cluster_Representative_Pair>& selected_cluster_representatives){

    /*
    for(int k = 0; k < all_current_clusters.size(); k++){
        Cluster current_cluster = all_current_clusters[k];

        for(int l = 0; l < current_cluster.members->size();l++){
            std::cout << current_cluster.members->at(l) << " ";
        }c
        std::cout << std::endl;
    }

    for(int i = 0; i < all_matching_results.size();i++){
        std::cout << all_matching_results[i].id_1 << " " << all_matching_results[i].id_2 << " " << all_matching_results[i].rel_quadr_cost << " " << std::endl;
    }
    */

    selected_cluster_representatives.clear();

    find_organoid_cluster_representative(all_matching_results,all_current_clusters,selected_cluster_representatives);

    new_matching_results.clear();

    for(int i = 0; i < selected_cluster_representatives.size();i++){

        int current_rep_img_num = selected_cluster_representatives[i].representative_img_number;

        for(int j = i + 1; j < selected_cluster_representatives.size();j++){

            int second_rep_img_num = selected_cluster_representatives[j].representative_img_number;

            Matching_Result existing_mr = get_matching_result_by_image_ids(all_matching_results,current_rep_img_num,second_rep_img_num,false);

            existing_mr.rel_quadr_cost = -FLT_MAX;
            existing_mr.rel_quadr_cost_optimal = -FLT_MAX;
            existing_mr.linear_cost_per_candidate = -FLT_MAX;
            existing_mr.linear_cost_per_feature = -FLT_MAX;

            if(existing_mr.id_1 == -1){

                existing_mr.id_1 = current_rep_img_num;
                existing_mr.id_2 = second_rep_img_num;
                existing_mr.set_id_1 = 1;
                existing_mr.set_id_2 = 1;
            }

            new_matching_results.push_back(existing_mr);
        }
    }

    for(int i = 0; i < all_matching_results.size();i++){

        Matching_Result existing_mr = all_matching_results[i];

        if(existing_mr.set_id_1 == 1 && existing_mr.set_id_2 == 1){
            continue;
        }

        if(existing_mr.set_id_1 == 2 && existing_mr.set_id_2 == 2){

            new_matching_results.push_back(existing_mr);
            continue;
        }

        Cluster_Representative_Pair current_crp;

        int img_num_from_second_elem = -1;
        int elem_to_find_rep_for = -1;

        if(existing_mr.set_id_1 == 1){
            elem_to_find_rep_for = existing_mr.id_1;
            img_num_from_second_elem = existing_mr.id_2;
        }else if(existing_mr.set_id_2 == 1){
            elem_to_find_rep_for = existing_mr.id_2;
            img_num_from_second_elem = existing_mr.id_1;
        }

        //current_crp = find_cluster_representative_pair_by_image_number(selected_cluster_representatives,elem_to_find_rep_for,all_current_clusters);

        bool is_cluster_rep = check_if_img_num_is_cluster_representative(selected_cluster_representatives,elem_to_find_rep_for,current_crp);

        if(!is_cluster_rep){
            continue;
        }

        if(current_crp.cluster_index == -1){
            std::cout << "found no cluster representative for element: " << elem_to_find_rep_for << std::endl;
        }


        Cluster corresponding_cluster = all_current_clusters[current_crp.cluster_index];

        float cost_of_second_to_existing_cluster = sum_matching_cost_between_single_element_and_cluster(corresponding_cluster,all_matching_results,img_num_from_second_elem,true);

        existing_mr.rel_quadr_cost = cost_of_second_to_existing_cluster;

        new_matching_results.push_back(existing_mr);
    }

    /*
    std::cout << std::endl;
    std::cout << "new matching results " << std::endl;
    std::cout << std::endl;

    for(int i = 0; i < new_matching_results.size();i++){
        std::cout << new_matching_results[i].id_1 << " " << new_matching_results[i].id_2 << " " << new_matching_results[i].rel_quadr_cost << " " << std::endl;
    }
    */

}

float get_initial_value_by_metric_type(Learning_Metric_Types lm_type){

    switch (lm_type)
    {
    case LM_BCE:
        return 1.0f;
        break;
    case LM_WBCE:
        return 1.0f;
        break;    
    case LM_FN_FP_NUM:
        std::cout << "initial value for LM_FN_FP_NUM is not yet implemented" << std::endl;
        return 0.0f;
        break;
    case LM_ACC:
        return 0.0f;
        break;
    case LM_F1_SCORE:
        return 0.0f;
        break;
    case LM_TPR_TNR_AVG:
        return 0.0f;
        break;
    case LM_MCC:
        return -1.0f;
        break;
    default:
        std::cout << "Unknown metric type in get_initial_value_by_metric_type" << std::endl;
        return false;
        break;
    }
}

bool check_loop_termination_condition_and_distribute_cost_parameters_from_queue(int process_id, std::queue<All_Cost_Parameters>& cost_parameter_queue, All_Cost_Parameters* current_parameters, int& iteration_counter, int max_num_iterations, double& current_runtime, double runtime_of_last_mm_calc, double time_limit, bool reference_clustering_was_valid, std::chrono::steady_clock::time_point& loop_begin){

    bool learning_loop_is_finished = false;

    if(process_id == 0){
        //loop_begin = std::chrono::steady_clock::now();

        learning_loop_is_finished = cost_parameter_queue.empty();

        if(learning_loop_is_finished){
            std::cout << "queue was empty" << std::endl;
        }

        if(!reference_clustering_was_valid){
            std::cout << "Invalid reference cluster. Not all images are contained in the clustering" << std::endl;
            learning_loop_is_finished = true;
        }

        if(iteration_counter > max_num_iterations){
            std::cout << "iteration max reached " << std::endl;
            learning_loop_is_finished = true;
        }

        if(current_runtime + runtime_of_last_mm_calc > time_limit){
            std::cout << "runtime max reached " << std::endl;
            learning_loop_is_finished = true;
        }

        if(!learning_loop_is_finished){
            *current_parameters = cost_parameter_queue.front();
            cost_parameter_queue.pop(); 
        }

        mpi_distribute_cost_parameters(current_parameters,learning_loop_is_finished);

    }else{

        mpi_distribute_cost_parameters(current_parameters,learning_loop_is_finished);
    }

    return learning_loop_is_finished;

}

void finalize_output(std::ofstream& log_file, All_Cost_Parameters& best_parameters, Metric_and_Threshold& best_metric_and_threshold){

    log_file << "\n";

    log_file << "best found result:\n";

    log_file << "metric:" << std::to_string(best_metric_and_threshold.metric) << " , threshold:" << best_metric_and_threshold.threshold << " , " << model_parameters_to_string(best_parameters) << "\n";

    log_file.close();

}

void output_results(std::ofstream& log_file, Model_Parameter_Selection_Strategy search_strat ,Learning_Metric_Types lm_type ,std::vector<Matching_Result>* matching_results, All_Cost_Parameters& cost_params, std::vector<Image_Features_Pair>& all_feature_vectors, std::string feature_vector_file_name, bool features_were_read_from_file, const Input_Arguments args, bool metric_is_new_best, std::string& filename_of_best_result, Metric_and_Threshold current_metric_and_threshold){

    std::filesystem::path output_file_path = get_data_folder_path();

    static bool log_file_and_folder_have_been_created = false;

    output_file_path.append("matching_results");

    if(search_strat == SEL_STRAT_NO_SEARCH){

        std::string file_name_for_matching_results = "matching_results";

        file_name_for_matching_results += args.current_time_string;

        std::string hdf5_file_name = file_name_for_matching_results;

        file_name_for_matching_results += ".csv";
        hdf5_file_name += ".hdf5";

        if(metric_is_new_best){
            filename_of_best_result = file_name_for_matching_results;
        }


        output_file_path.append(file_name_for_matching_results);

        write_matching_results_to_file(output_file_path,matching_results,cost_params, all_feature_vectors,feature_vector_file_name,args.read_feature_vector, args);
        //write_matching_results_as_hdf5(hdf5_file_name,&all_matching_results,cost_params, all_feature_vectors,name_of_feature_vectors_file,args.read_feature_vector, args);
    }else{
        std::string folder_name = std::string("learning_results") += args.current_time_string;

        output_file_path.append(folder_name);

        if(!log_file_and_folder_have_been_created){
            std::filesystem::create_directory(output_file_path);

            std::filesystem::path log_file_path = output_file_path;
            log_file_path.append("learning_log.txt");

            log_file.open(log_file_path);
            log_file << "begin of log file\n";
            std::string time_string = args.current_time_string;
            time_string = time_string.substr(1);
            log_file << time_string << "\n";
            log_file << get_string_from_metric_type(lm_type) << "\n";
            log_file << get_string_from_search_strategy(search_strat) << "\n";
            log_file << "offset_lower_bound: " << MODEL_PARAMETER_OFFSET_LOWER_BOUND << "\n";
            log_file << "offset_upper_bound: " << MODEL_PARAMETER_OFFSET_UPPER_BOUND << "\n";
            log_file << "weight_lower_bound: " << MODEL_PARAMETER_WEIGHT_LOWER_BOUND << "\n";
            log_file << "weight_upper_bound: " << MODEL_PARAMETER_WEIGHT_UPPER_BOUND << "\n";
            log_file << "reference cluster: " << args.reference_clustering_file_name << "\n";

            log_file_and_folder_have_been_created = true; 
        }



        std::string file_name_for_matching_results = "matching_results";

        file_name_for_matching_results += args.current_time_string;
        file_name_for_matching_results += "_";
        file_name_for_matching_results += model_parameters_to_string(cost_params);

        std::string hdf5_file_name = file_name_for_matching_results;

        file_name_for_matching_results += ".csv";
        hdf5_file_name += ".hdf5";

        if(metric_is_new_best){
            filename_of_best_result = file_name_for_matching_results;
        }

        log_file << "metric:" << std::to_string(current_metric_and_threshold.metric) << " , threshold:" << current_metric_and_threshold.threshold << " , " << model_parameters_to_string(cost_params) << "\n";

        output_file_path.append(file_name_for_matching_results);

        write_matching_results_to_file(output_file_path,matching_results,cost_params, all_feature_vectors,feature_vector_file_name,args.read_feature_vector, args);
        //write_matching_results_as_hdf5(hdf5_file_name,&all_matching_results,cost_params, all_feature_vectors,name_of_feature_vectors_file,args.read_feature_vector, args);

        log_file.flush();
    }

}

Search_Dim get_next_dim_in_search_order(std::vector<Search_Dim>& search_order,int& current_search_order_index){

    current_search_order_index++;

    if(current_search_order_index >= search_order.size()){
        current_search_order_index = 0;
    }

    return search_order[current_search_order_index];

}

void* get_initial_search_data_struct_by_search_strat(Model_Parameter_Selection_Strategy search_strat,const Input_Arguments& args){

    void* return_ptr;

    All_Cost_Parameters empty_cost_parameters{0,0,0,0,0,0,0};
    Line_Search_Data_Struct line_search_data{SD_NOT_SET,0,false,empty_cost_parameters,empty_cost_parameters,empty_cost_parameters};

    switch(search_strat){
        case SEL_STRAT_NO_SEARCH:
            return nullptr;
            break;
        
        case SEL_STRAT_SIM_ANN:
            return_ptr = (Sim_Annealing_Search_Data_Struct*)malloc(sizeof(Sim_Annealing_Search_Data_Struct));
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->temperature = args.sim_ann_init_temp;
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->cooling_rate = args.sim_ann_cooling_rate;
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->std_dev = args.sim_ann_std_dev;
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->prev_metric = get_initial_value_by_metric_type(args.lm_type);
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->prev_accepted_parameters = empty_cost_parameters;
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->last_best_parameters = empty_cost_parameters;
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->temp_at_last_best = args.sim_ann_init_temp;
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->num_time_steps_since_last_best = 0;
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->restart_time_step_threshold = args.sim_ann_restart_thresh;
            ((Sim_Annealing_Search_Data_Struct*)return_ptr)->metric_of_best = get_initial_value_by_metric_type(args.lm_type);
            return return_ptr;
            break;

        case SEL_STRAT_EXHAUSTIVE_ADJ:
            return_ptr = (Exhaustive_Adj_Search_Data_Struct*)malloc(sizeof(Exhaustive_Adj_Search_Data_Struct));
            ((Exhaustive_Adj_Search_Data_Struct*)return_ptr)->currently_search_for_best_adj = false;
            ((Exhaustive_Adj_Search_Data_Struct*)return_ptr)->new_best_was_found = true;
            return return_ptr;
            break;

        case SEL_STRAT_LINE_SEARCH:
            return_ptr = (Line_Search_Data_Struct*)malloc(sizeof(Line_Search_Data_Struct));
            *((Line_Search_Data_Struct*)return_ptr) = line_search_data;
            return return_ptr;
            break;

        default:
            return nullptr; 
            break;
    }

}

std::vector<int>* assign_candidates_to_features_using_qap_solver(std::vector<Feature_Point_Data>* features, std::vector<Feature_Point_Data>* candidates, Cost_Rescale_Factors cost_rescale_factors){

    int num_features_points_in_model = features->size();
    int num_candidates_in_model = candidates->size();

    int max_num_points = std::max<int>(num_features_points_in_model,num_candidates_in_model);
    double theoretical_optimal_cost = get_optimal_cost_for_problem_size(max_num_points,cost_rescale_factors);

    std::filesystem::path data_folder_path = get_data_folder_path();
    data_folder_path.append("matching.dd");
    
    std::ofstream model_file;
    model_file.open(data_folder_path);

    //std::cout << "Theoretical optimum: " << theoretical_optimal_cost << std::endl;

    int total_num_variables = num_features_points_in_model * num_candidates_in_model;

    std::vector<int>* solution_instance = new std::vector<int>(num_features_points_in_model,-1);


    double* variable_costs = (double*)malloc(sizeof(double) * total_num_variables);

    int total_num_quadratic_expressions = ((num_features_points_in_model * num_features_points_in_model - num_features_points_in_model) / 2) * (num_candidates_in_model * num_candidates_in_model - num_candidates_in_model);

    std::cout << "total_num_quadr_expr: " << total_num_quadratic_expressions << std::endl;

    double* quadratic_expressions_costs = (double*)malloc(sizeof(double) * total_num_quadratic_expressions);
    int* quadratic_edge_indices = (int*)malloc(sizeof(int) * 2 * total_num_quadratic_expressions);
    //model_file << "p " << num_features_points_in_model << " " << num_candidates_in_model << " " << total_num_variables << " " << num_valid_quadr_terms << std::endl; 

    for(int i = 0; i < num_features_points_in_model; i++){
        for(int j = 0; j < num_candidates_in_model; j++){

            int var_index = i * num_candidates_in_model + j;
            
            variable_costs[var_index] = calculate_cost_for_two_feature_points((*features)[i],(*candidates)[j],cost_rescale_factors);

            //model_file << "a " << var_index << " " << i << " " << j << " " << variable_costs[var_index] << std::endl;
        
        }

    }

    int num_quadr_expr = 0;

    for(int i = 0; i < total_num_variables;i++){
        for(int j = i+1; j < total_num_variables;j++){

            int index_first_feature = i / num_candidates_in_model;
            int index_fist_candidate = i % num_candidates_in_model;

            int index_second_feature = j / num_candidates_in_model;
            int index_second_candidate = j % num_candidates_in_model;

            double new_cost = 0.0;

            if(!(index_first_feature == index_second_feature || index_fist_candidate == index_second_candidate)){
                new_cost = calculate_cost_for_single_feature_point_pair_assignment((*features)[index_first_feature],(*candidates)[index_fist_candidate],(*features)[index_second_feature],(*candidates)[index_second_candidate],cost_rescale_factors);
                model_file << "e " << index_first_feature * num_candidates_in_model + index_fist_candidate << " " << index_second_feature * num_candidates_in_model + index_second_candidate << " " << (float)new_cost << std::endl;
                quadratic_expressions_costs[num_quadr_expr] = new_cost;
                quadratic_edge_indices[num_quadr_expr * 2] = index_first_feature * num_candidates_in_model + index_fist_candidate;
                quadratic_edge_indices[num_quadr_expr * 2 + 1] = index_second_feature * num_candidates_in_model + index_second_candidate;
                num_quadr_expr++;
            }

        }
    }


    python_solve_qap(global_python_handler, solution_instance,num_features_points_in_model,num_candidates_in_model,total_num_variables,total_num_quadratic_expressions,variable_costs,quadratic_expressions_costs,quadratic_edge_indices);

    free(variable_costs);
    free(quadratic_expressions_costs);
    free(quadratic_edge_indices);

    
    model_file.close();

    return solution_instance;
    //return NULL;
    
}