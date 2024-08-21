#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <chrono>

enum Learning_Metric_Types{
    LM_BCE,
    LM_WBCE,
    LM_FN_FP_NUM,
    LM_ACC,
    LM_F1_SCORE,
    LM_TPR_TNR_AVG,
    LM_MCC
};

#define LM_BCE_STRING "LM_BCE"
#define LM_WBCE_STRING "LM_WBCE"
#define LM_FN_FP_NUM_STRING "LM_FN_FP_NUM"
#define LM_ACC_STRING "LM_ACC"
#define LM_F1_SCORE_STRING "LM_F1_SCORE"
#define LM_TPR_TNR_AVG_STRING "LM_TPR_TNR_AVG"
#define LM_MCC_STRING "LM_MCC"

enum Model_Parameter_Selection_Strategy{
    SEL_STRAT_NO_SEARCH,
    SEL_STRAT_EXHAUSTIVE_ADJ,
    SEL_STRAT_LINE_SEARCH,
    SEL_STRAT_SIM_ANN
};

#define SEL_STRAT_NO_SEARCH_STRING "SEL_STRAT_NO_SEARCH"
#define SEL_STRAT_EXHAUSTIVE_ADJ_STRING "SEL_STRAT_EXHAUSTIVE_ADJ"
#define SEL_STRAT_LINE_SEARCH_STRING "SEL_STRAT_LINE_SEARCH"
#define SEL_STRAT_SIM_ANN_STRING "SEL_STRAT_SIM_ANN"


#define USE_CUSTOM_MEMORY_HANDLER true

#define SEGMENT_ORGANOID_IMAGES false

#define SKIP_FINAL_VISUALIZATION false
#define SKIP_MATCHING_CALCULATION false
#define SKIP_CLUSTERING true

#define READ_FEATURE_VECTOR_FROM_FILE true
#define READ_MATCHING_RESULTS_FROM_FILE true

#define WRITE_FEATURE_VECTOR_TO_FILE false
#define WRITE_MATCHING_RESULTS_TO_FILE false


#define IMAGE_SET_NAME "organoid_dataset_100_split_2"//"organoid_dataset_with_unseen_classes"//"organoid_dataset_unseen_classes"//"organoid_dataset_100_split_1"//"combined_1000"//
#define FEATURE_VECTORE_FILE_NAME "feature_vectors_organoid_dataset_100_split_2_15_1_14_23.csv"//"feature_vectors_organoid_dataset_with_unseen_classes_6_2_12_59.csv"//"feature_vectors_organoid_dataset_unseen_classes_6_2_17_34.csv"//"feature_vectors_combined_1000_23_2_12_19.csv"//
#define MATCHING_RESULTS_FILE_NAME  "matching_results_organoid_dataset_100_split_2.csv"//"matching_results_dataset_with_unseen_classes.csv"//"matching_results_organoid_dataset_unseen_classes.csv"//"matching_results_combined_1000.csv"//

#define PAIRWISE_COMPARISON_FILENAME ""

#define IMAGE_SET_NAME_2 ""
#define FEATURE_VECTORE_FILE_NAME_2 ""
#define MATCHING_RESULTS_FILE_NAME_2 ""

#define USE_SECONDARY_IMAGE_SET false

#define READ_MINIMAL_MATCHING_RESULTS true

#define READ_MODEL_PARAMETERS_FROM_FILE true
#define MODEL_PARAMETER_FILE_NAME "model_parameters.txt"

#define DEFAULT_LEARNING_TASK_FILE "learning_task.txt"
#define DEFAULT_READ_LEARNING_TASK false

#define RUNTIME_LIMIT "00:03:00"




#define DEFAULT_SELECTION_STRATEGY SEL_STRAT_NO_SEARCH
#define DEFAULT_LEARNING_METRIC LM_F1_SCORE

#define DEFAULT_STEP_WIDTH 0.1

#define DEFAULT_STD_DEV_SIM_ANN 0.05
#define DEFAULT_TEMP_SIM_ANN 1.0
#define DEFAULT_COOL_RATE_SIM_ANN 0.98
#define DEFAULT_RESTART_THRESHOLD_SIM_ANN 10

#define DEFAULT_REFERENCE_CLUSTERING "clustering_100_split_2_optimal.csv"//"clustering_with_unseen_classes_optimal.csv"//"clustering_only_unseen_classes_optimal.csv"//"clustering_100_split_1_optimal.csv"//

#define USE_HISTOGRAM_COMPARISON_FOR_MATCHING false

#define PRINT_TO_LOGS false

#define WRITE_EXECUTION_TIME_MEASUREMENTS_TO_FILE false

#define COST_DOUBLE_ASSIGNMENT 10000.0
#define COST_MISMATCHING_CHANNELS 10000.0

#define RED_CHANNEL_FEATURE_ACCEPTANCE_THRESHOLD 0.5//
#define RED_CHANNEL_CANDIDATE_ACCEPTANCE_THRESHOLD 0.4//

#define RED_CHANNEL_DISTANCE_WEIGHT 1.5
#define RED_CHANNEL_PRESIENCE_WEIGHT 2.0
#define RED_CHANNEL_REL_PEAK_VAL_WEIGHT 2.0
#define RED_CHANNEL_NUMBER_POINTS_WEIGHT 0.0
#define RED_CHANNEL_GRADIENT_WEIGHT 0.0

#define RED_CHANNEL_PRESIENCE_CUTOFF 2.0
#define RED_CHANNEL_PEAK_VALUE_CUTOFF 4.0

#define RED_CHANNEL_LOCAL_SEARCH_DIM_AS_PERCENTAGE_OF_IMAGE_SIZE 0.05
#define RED_CHANNEL_MINIMUM_SEPARATION_AS_PERCENTAGE_OF_IMAGE_SIZE 0.02
#define RED_CHANNEL_SEPERATION_THRESHOLD 0.05

#define RED_CHANNEL_MINIMUM_PEAK_VALUE 0.1

#define NUCLEUS_OVERLAP_THRESHOLD 0.5

#define SINGLE_ORGANOID_OUTPUT_COLOR_RESCALE_FACTOR 3.0f

#define MATCHING_MATRIX_INDIVIDUAL_SQUARE_SIZE 8
#define CLUSTERING_VIEW_INDIVIDUAL_SQUARE_SIZE 64

#define CALCULATE_CLUSTERING_USING_GUROBI false


#define USE_GREEDY_ASSIGNMENT_TRANSPOSITIONING false

#define COL_SEARCH_RANGE_IN_CYCLIC_BAB_SEARCH 0.05
#define COL_SEARCH_RANGE_IN_CYCLIC_BAB_SEARCH_MIN_SIZE 10

// i.e 0.1 -> 10% , the neighborhood list will store up to 10% of the total number of feature points
#define ANGLE_RANGE_SEARCH_NEIGHBORHOOD_LIST_SIZE 0.1


#define ANGLE_RANGE_SEARCH_INCREMENT 5

#define ANGLE_RANGE_SEARCH_SIZE 5


#define NUM_ELEMS_MPI_COMM_MATCHING_TASK 3
#define NUM_ELEMS_MPI_COMM_MATCHING_RESULT 10
#define NUM_ELEMS_MPI_COMM_EXECUTION_TIME_MEASUREMENT 3
#define NUM_ELEMS_MPI_COMM_COST_PARAMETERS 6

#define NUM_DIFFERENT_ALGORITHMS 9

#define KEYCODE_B 98
#define KEYCODE_C 99
#define KEYCODE_E 101
#define KEYCODE_F 102
#define KEYCODE_H 104
#define KEYCODE_J 106
#define KEYCODE_K 107
#define KEYCODE_L 108
#define KEYCODE_M 109
#define KEYCODE_N 110
#define KEYCODE_O 111
#define KEYCODE_P 112
#define KEYCODE_R 114
#define KEYCODE_V 118
#define KEYCODE_W 119

#define KEYCODE_ESC 27
#define KEYCODE_SPACE 32

#define MATCHING_MATRIX_WINDOW_NAME "Matching Matrix"

#define NUM_CLUSTERING_WINDOWS 3
#define PRIMARY_CLUSTERING_WINDOW_IMG_ID 0
#define SECONDARY_CLUSTERING_WINDOW_IMG_ID 1
#define COMBINED_CLUSTERING_WINDOW_IMG_ID 2

enum Algorithm_Type{
    BAB_GREEDY_ALGORITHM,
    BAB_GREEDY_CYCLIC_ALGORITHM,
    BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION,
    GREEDY_DISTANCE_MATCHING_ALGORITHM,
    GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD,
    GREEDY_COST_MATCHING_ALGORITHM,
    GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE,
    GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION,
    GUROBI_ALGORITHM
};

enum Channel_Type{
    DAPI_CHANNEL,
    PDX1_GFP_CHANNEL,
    NEUROG3_RFP_CHANNEL,
    Phalloidon_AF647_CHANNEL,
    INVALID_CHANNEL
};

enum Experiment_Replicate{
    EXP_REPLICATE_A,
    EXP_REPLICATE_B,
    EXP_REPLICATE_C,
    EXP_REPLICATE_UNKNOWN
};

enum Search_Dim{
    SD_NOT_SET,
    SD_DIST_OFFSET,
    SD_COLOR_OFFSET,
    SD_ANGLE_OFFSET,
    SD_COLOR_TO_DIST_WEIGHT,
    SD_UNARY_TO_QUADR_WEIGHT,
};

typedef struct All_Cost_Parameters{
    double cost_double_assignment;
    double cost_mismatching_channels;

    double color_offset; // gamma
    double dist_offset; // gamma' 
    double angle_offset; // gamma'
    double color_to_dist_weight; // theta
    double unary_to_to_quadr_weight; 

    //double static_cost_offset;
    //double distance_difference_cost_coefficient;
    //double color_difference_cost_coefficient;
    //double angle_difference_cost_coefficient;
    //double static_pairwise_cost_offset;
}All_Cost_Parameters;

typedef struct Cluster{
    std::vector<int>* members;
    double cost_of_cluster;
}Cluster;

typedef struct Organoid_Image_Header{
    int plate_number;
    unsigned short year_of_experiment;
    unsigned short month_of_experiment;
    unsigned short day_of_experiment;
    unsigned char well_row_char;
    unsigned short well_col_number;
    unsigned short z_position;
    Channel_Type channel;
    Experiment_Replicate replicate;
    std::filesystem::path full_file_path;
}Organoid_Image_Header;

typedef struct Feature_Point_Data{
    Channel_Type channel;
    uint16_t row;
    uint16_t col;
    double local_mean;
    double local_std_dev;
    double local_mean_forground_only;
    double local_std_dev_forground_only;
    uint16_t peak_value;
    double normalize_peak_value;
    uint8_t local_search_dim;
    float relative_distance_center_max_distance;
    float angle;
    float relative_distance_center_boundary;
}Feature_Point_Data;

typedef struct Matching_Result_Additional_Viz_Data{
    //std::vector<int>* assignment;
    std::vector<Feature_Point_Data>* features;
    std::vector<Feature_Point_Data>* assigned_candidates;
    cv::Mat* feature_image;
    cv::Mat* candidate_image;
    std::filesystem::path path_to_feature_image;
    std::filesystem::path path_to_candidate_image;
    cv::Point2i center_feature_image;
    cv::Point2i center_candidate_image;
    cv::Point2i offset_feature_image;
    cv::Point2i offset_candidate_image;
}Matching_Result_Additional_Viz_Data;

typedef struct Assignment_Visualization_Data{
    std::vector<cv::Mat*>* images;
    std::vector<cv::Point2i>* centers;
    std::vector<cv::Point2i>* offsets;
    std::vector<std::filesystem::path>* image_paths;
}Assignment_Visualization_Data;

typedef struct Matching_Calculation_Task{
    int id_1;
    int id_2;
    double runtime_estimation;
    Assignment_Visualization_Data* viz_data; 
    int index_features_in_viz_data;
    int index_candidates_in_viz_data;
    All_Cost_Parameters* cost_params;
}Matching_Calculation_Task;

typedef struct Task_Vector{
    std::vector<Matching_Calculation_Task>* tasks;
    double total_runtime_estimation;
}Task_Vector;

typedef struct MPI_Comm_Matching_Task{
    int id_1;
    int id_2;
    double runtime_estimation;
}MPI_Comm_Matching_Task;

typedef struct MPI_Comm_Cost_Parameters{
    int learning_loop_is_finished;
    double runtime_estimation;
    double color_cost_offset;
    double dist_cost_offset;
    double angle_cost_offset;
    double color_to_dist_weight;
    double unary_to_quadr_weight;
}MPI_Comm_Cost_Parameters;

typedef struct MPI_Comm_Matching_Result{
    double rel_quadr_cost_optimal;
    double rel_quadr_cost;
    double linear_cost_per_feature;
    double linear_cost_per_candidate;
    int id_1;
    int id_2;
    int set_id_1;
    int set_id_2;
    int num_elems_in_assignment;
    int offset_into_assignment_array;
}MPI_Comm_Matching_Result;

typedef struct MPI_Comm_Execution_Time_Measurement{
    int num_features;
    int num_candidates;
    double execution_time;
}MPI_Comm_Execution_Time_Measurement;

typedef struct Matching_Result{
    double rel_quadr_cost_optimal;
    double rel_quadr_cost;
    double linear_cost_per_feature;
    double linear_cost_per_candidate;
    int id_1;
    int id_2;
    int set_id_1;
    int set_id_2;
    std::vector<int>* assignment;
    Matching_Result_Additional_Viz_Data* additional_viz_data_id1_to_id2;
    Matching_Result_Additional_Viz_Data* additional_viz_data_id2_to_id1;
}Matching_Result;

enum Matching_Visualization_Type{
    MATCHING_VIZ_QUADRATIC_OPTIMAL,
    MATCHING_VIZ_QUADRATIC,
    MATCHING_VIZ_LIN_FEATURE,
    MATCHING_VIZ_LIN_CANDIDATE
};

typedef struct Image_Features_Pair{
    std::filesystem::path image_path;
    int image_number;
    int set_number;
    std::vector<Feature_Point_Data>* features;
    std::vector<Feature_Point_Data>* candidates;
}Image_Features_Pair;


typedef struct Op_Numbers_And_Runtime {
    int num_features;
    int num_candiates;
    double total_runtime;
    double runtime[NUM_DIFFERENT_ALGORITHMS];
    double normalized_results[NUM_DIFFERENT_ALGORITHMS];
}Op_Numbers_And_Runtime;

typedef struct All_Feature_Point_Selection_Parameters{

    double red_channel_feature_acceptance_threshold;
    double red_channel_candidate_acceptance_threshold;
    double red_channel_distance_weight;
    double red_channel_presience_weight;
    double red_channel_rel_peak_val_weight;
    double red_channel_number_points_weight;
    double red_channel_presience_cutoff;
    double red_channel_peak_value_cutoff;
    double red_channel_local_search_dim;
    double red_channel_minimum_separation;
    double nucleus_overlap_threshold;
    std::string feature_vector_file;
}All_Feature_Point_Selection_Parameters;

typedef struct Additional_Viz_Data_Input{
    std::vector<Image_Features_Pair>* all_feature_vectors;
    std::vector<cv::Mat*>* sorted_square_organoid_images;
    std::vector<cv::Point2i>* sorted_image_centers;
    std::vector<cv::Point2i>* sorted_offsets;
    std::vector<std::filesystem::path>* sorted_image_paths;
}Additional_Viz_Data_Input;

typedef struct Input_Arguments{
    bool segment_organoids;
    bool skip_visualization;
    bool skip_matching;
    bool read_feature_vector;
    std::string feature_vector_file_name;
    std::string second_feature_vector_file_name;
    bool read_matching_results;
    std::string matching_results_file_name;
    std::string second_matching_results_file_name;
    bool use_secondary_image_set;
    bool write_feature_vector;
    bool write_matching_results;
    std::string image_set_name;
    std::string second_image_set_name;
    bool perform_dry_run;
    bool print_help_info;
    bool print_logs;
    bool write_execution_time_measurements;
    std::string current_time_string;
    All_Cost_Parameters all_model_parameters;
    bool read_model_parameters_from_file;
    std::string model_parameter_file_name;
    double runtime_limit;
    Learning_Metric_Types lm_type;
    Model_Parameter_Selection_Strategy sel_strat;
    std::string reference_clustering_file_name;
    bool read_learning_task;
    std::string learning_task_file_name;
    std::vector<Search_Dim>* search_order; 
    float initial_search_step_size;
    float sim_ann_init_temp;
    float sim_ann_cooling_rate;
    float sim_ann_std_dev;
    int sim_ann_restart_thresh;
}Input_Arguments;

typedef struct Cluster_Representative_Pair{
    int representative_img_number;
    int cluster_index;
}Cluster_Representative_Pair;

typedef struct Confusion_Matrix{
    float true_positives;
    float false_negatives;
    float false_positives;
    float true_negatives;
}Confusion_Matrix;

typedef struct Matching_Result_Label{
    int label;
    int matching_result_id1;
    int matching_result_id2;
}Matching_Result_Label;

typedef struct TPR_FPR_Tuple{
    float tn;
    float tp;
    float fn;
    float fp;

    float tpr;
    float fpr;
    float tnr;
    float ks_score;
    float threshold;
    float accuracy;
    float precision;
    float recall;
    float f1_score;
    float f1_cuts;
    float f1_joins;
    float mcc;

    float prec_cuts;
    float prec_joins;
    float recall_cuts;
    float recall_joins;
}TPR_FPR_Tuple;
