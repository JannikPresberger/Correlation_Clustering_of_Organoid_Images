#pragma once

typedef struct Memory_Element Memory_Element;
typedef struct Memory_Manager_Fixed_Size Memory_Manager_Fixed_Size;

#include <iostream>
#include <string>
#include <filesystem>
#include <unordered_map>
#include "global_data_types.h"

#include "memory_manager.h"

#include "andres/marray-hdf5.hxx"

#define SQRT_2 1.41421356237
#define PI 3.14159265359 

enum Token_Type{
    PLATE_NUM_AND_DATE_TOKEN,
    LIBRARY_TOKEN,
    LIBRARY_NAME_TOKEN,
    WELL_NUMBER_TOKEN,
    Z_POS_AND_CHANNEL_TOKEN
};

template <typename T> int get_sign(T val){
    return(T(0) < val ) - (val < T(0));
}

void print_key_press_prompt();

void uint32_t_set_single_bit_to_zero(uint32_t* word, unsigned int bit_index);
void uint32_t_set_single_bit_to_one(uint32_t* word, unsigned int bit_index);

uint32_t uint32_t_flip_single_bit(uint32_t* word, unsigned int bit_index);

uint32_t uint32_t_query_single_bit(uint32_t* word, unsigned int bit_index);

void print_int_vector(std::vector<int>& vec);

bool check_if_doubles_are_equal(double a, double b);

bool check_if_uint_is_power_of_two(unsigned int n);

size_t get_next_bigger_power_of_two(size_t value);

int get_index_of_element_in_ordered_vector(std::vector<int>& vec, int value);
int get_closest_index_of_element_in_ordered_vector(std::vector<int>& vec, int value);


// this function returns the signed differnece between two angles in degrees. The difference gives us how much we need to rotate from alpha to beta in CCW fashion
double get_signed_difference_between_two_angles_radians(double alpha, double beta);

double get_signed_difference_between_two_angles(double alpha, double beta);

double get_difference_between_two_angles(double alpha, double beta);

cv::Vec2f rotate_vec_2d_by_radians(cv::Vec2f& vector, double alpha);

bool compare_two_instance_vectors(std::vector<int>* instance_1, std::vector<int>* instance_2);
bool compare_two_instance_arrays(int* instance_1, int* instance_2, int instance_size);

void print_instance_vector_as_vector(std::vector<int>* instance_vector);
void print_instance_array_as_vector(int* instance_vector, int instance_size);

void print_instance_vector_as_matrix(std::vector<int>* instance_vector, int instance_size_rows, int instance_size_cols);
void print_instance_array_as_matrix(int* instance_vector, int instance_size_rows, int instance_size_cols);

void print_multiple_instance_vectors_as_matrices(std::vector<std::vector<int>*>& instance_vectors, int instance_size_rows, int instance_size_cols);
void print_multiple_instance_arrays_as_matrices(std::vector<int*>& instance_arrays, int instance_size_rows, int instance_size_cols);

bool check_if_instance_vector_column_is_occupied(std::vector<int>& instance_vector, int col);
bool check_if_instance_array_column_is_occupied(int* instance_array, int instance_size, int col);

void list_all_feasible_problem_instances(int rows, int cols);

void list_all_subinstances_recursively(std::vector<int>* instance_vector,int start_row , int last_row, int total_rows, int total_cols, std::vector<std::vector<int>*>& all_feasible_instances, int* num_created_instances);
void list_all_subinstances_recursively_custom_memory(Memory_Element* instance_vector,int start_row, int last_row, int total_rows, int total_cols, std::vector<Memory_Element>& all_feasible_instances, int* num_created_instances, Memory_Manager_Fixed_Size* mem_manager);

float evaluate_logisitic_function(float x, float steepness, float upper_asymptote, float lower_asymptote, float x_value_at_half_of_y_range);

float evaluate_linear_function(float x, float y_intersect, float x_intersect);

int calculate_binomial_coefficient(int n, int k);

void write_cost_arrays_as_hdf5(andres::Marray<double>& unary_cost_array, andres::Marray<double>& pair_cost_array, std::string filename);

float sample_uniform_0_1();
float sample_gaussian(float mu, float sigma);

void remove_extension(std::string& filename);
void remove_extension_and_channel_number(std::string& filename);

void create_folder_structure_for_organoid_image(std::filesystem::path base_path, std::string folder_name);

void parse_token(const char* token, Token_Type& t_type, Organoid_Image_Header& header);

std::filesystem::path find_python_scripts_folder(std::filesystem::path base_path);

Experiment_Replicate get_experiment_replicate_from_char(char replicate);
Channel_Type get_channel_type_from_int(unsigned int channel_int);

Organoid_Image_Header parse_org_img_filename(std::filesystem::path filepath);

bool check_if_file_is_mask_image(std::string file_name);

int get_image_number_from_file_name(std::string file_name);

void append_current_date_and_time_to_string(std::string& string);

void read_clustering_from_csv_file(std::string filename, std::vector<Cluster>& clustering);
void write_clusters_to_csv_file(std::vector<Cluster>* clustering,std::string corresponding_matching_results_file, float used_clustering_threshold);

void write_execution_time_measurements_to_file(std::string file_name, std::vector<Op_Numbers_And_Runtime>& op_numbers, std::vector<Image_Features_Pair>& all_feature_vectors);

void write_matching_results_to_file(std::filesystem::path output_file_path, std::vector<Matching_Result>* matching_results, All_Cost_Parameters& cost_params, std::vector<Image_Features_Pair>& all_feature_vectors, std::string feature_vector_file_name, bool features_were_read_from_file, const Input_Arguments args);
void write_matching_results_as_hdf5(std::string file_name, std::vector<Matching_Result>* matching_results, All_Cost_Parameters& cost_params, std::vector<Image_Features_Pair>& all_feature_vectors, std::string feature_vector_file_name, bool features_were_read_from_file, const Input_Arguments args);


void write_all_feature_vectors_to_file(std::string file_name, std::vector<Image_Features_Pair>& all_feature_vectors,const Input_Arguments args);

void read_matching_results_from_file(std::filesystem::path file_path, std::vector<Matching_Result>* matching_results, All_Cost_Parameters& cost_params, All_Feature_Point_Selection_Parameters& feature_point_selection_params);

bool read_feature_vector_from_file(std::filesystem::path file_path, int image_number, std::vector<Feature_Point_Data>& output_feature_points, std::vector<Feature_Point_Data>& output_candidate_points);

void read_selection_parameters_from_file(std::filesystem::path file_path, All_Feature_Point_Selection_Parameters& feature_point_selection_params);

void print_cost_parameters(All_Cost_Parameters* cost_params);

void pixel_pos_from_center_length_and_angle(int& col, int &row, int center_col, int center_row, double length, double angle, double pixels_per_unit_length);

std::filesystem::path get_data_folder_path();

std::filesystem::path get_image_folder_path();

void fill_all_feature_selection_parameter_struct_with_currently_defined_vars(All_Feature_Point_Selection_Parameters& parameters, std::string feature_vector_file_name);

bool compare_feature_selection_parameters(All_Feature_Point_Selection_Parameters& params_1,All_Feature_Point_Selection_Parameters& params_2);

void read_input_argument(Input_Arguments& input_arguments, int argc, char** argv, All_Cost_Parameters initial_cost_params);

bool read_image_numbers_from_feature_vector_file(Input_Arguments& args, std::vector<int>& image_numbers);

bool read_image_numbers_from_feature_vector_file(std::filesystem::path file_path, std::vector<int>& image_numbers);

void read_image_numbers_from_image_set_folder(std::filesystem::path image_set_path, std::vector<int>& image_numbers);

bool check_if_image_number_is_contained_in_vector(const std::vector<int>& image_number_vector, int image_number);
bool check_if_image_number_is_contained_in_clustering(const std::vector<Cluster>& clustering, int image_number);
bool check_if_all_image_numbers_are_contained_in_clustering(const std::vector<Cluster>& clustering, const std::vector<Image_Features_Pair>& all_feature_vectors);

bool check_if_file_exists(std::filesystem::path file_path);

void merge_images_from_multiple_folders_into_single_folder(std::filesystem::path base_path);

void check_if_image_and_mask_is_present_for_all_images(std::filesystem::path path);

void print_cost_parameters(const All_Cost_Parameters& cost_params);

Matching_Result get_matching_result_by_image_ids(std::vector<Matching_Result>& all_matching_results, int id_1, int id_2, bool exact_match_only);

float sum_matching_cost_between_single_element_and_cluster(Cluster& cluster, std::vector<Matching_Result>& all_matching_results, int img_num_of_single_elem, bool normalize_by_cluster_size);

Cluster_Representative_Pair find_cluster_representative_pair_by_image_number(std::vector<Cluster_Representative_Pair>& all_crp, int target_img_num, std::vector<Cluster>& all_clusters);

bool check_if_img_num_is_cluster_representative(std::vector<Cluster_Representative_Pair>& all_crp, int img_num, Cluster_Representative_Pair& output_crp_of_img_num);

std::string get_clustering_window_name_by_window_num(int window_num);

void calculate_variation_of_information_for_clusterings(const std::vector<Cluster>& reference_clustering,const std::vector<Cluster> &clustering, std::ofstream& output_file, double clustering_runtime);

void calculate_confusion_matrix_from_clustering(Confusion_Matrix& confusion_matrix, const std::vector<Cluster>& reference_clustering, std::vector<Matching_Result>& merged_matching_results, double threshold, const std::vector<Cluster> &clustering);

void calculate_confusion_matrix(Confusion_Matrix& confusion_matrix, std::vector<Matching_Result>& merged_matching_results, double threshold, const std::vector<Cluster> &clustering);

void calculate_confusion_matrix_with_pairwise_truth(Confusion_Matrix& confusion_matrix, std::vector<Matching_Result>& merged_matching_results, double threshold, const std::vector<Matching_Result>& truth_matching_results);

void calculate_confusion_matrix_of_clustering_to_pairwise_truth(Confusion_Matrix& confusion_matrix, std::vector<Matching_Result>& merged_matching_results, double threshold, const std::vector<Cluster> &clustering);

TPR_FPR_Tuple get_tpr_fpr_tuple_from_confusion_matrix(const Confusion_Matrix& confusion_matrix, float used_threshold);

void print_confusion_matrix(const Confusion_Matrix& confusion_matrix);

void write_ROC_to_csv_file(std::string file_name, std::vector<TPR_FPR_Tuple>& roc);

double calculate_binary_cross_entropy(std::vector<Matching_Result>& all_merged_matching_results, bool normalize_by_num_occurences, const std::vector<Cluster> &clustering);

bool read_model_parameters_from_file(std::filesystem::path model_param_file, All_Cost_Parameters& all_read_model_params);

void write_model_parameters_to_file(std::string filename, All_Cost_Parameters& all_cost_parameters);

void print_model_parameters(All_Cost_Parameters& cost_params, bool short_print);

std::string model_parameters_to_string(All_Cost_Parameters& cost_params);

std::string get_string_from_metric_type(Learning_Metric_Types metric_type);

std::string get_string_from_search_strategy(Model_Parameter_Selection_Strategy sel_strat);

Learning_Metric_Types get_metric_type_from_string(std::string metric_type_string);

Model_Parameter_Selection_Strategy get_search_strategy_from_string(std::string sel_strat_string);

int get_cluster_index_from_image_number(const std::vector<Cluster> &clustering, int image_number);

void read_learning_task(std::string learning_task_file_name, Input_Arguments& input_args);

void read_pairwise_comparison_as_matching_results(std::filesystem::path file_path, std::vector<Matching_Result>& all_matching_results);