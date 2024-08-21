#include "utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <queue>
#include "math.h"
#include "andres/partition-comparison.hxx"

#define SEGMENT_ORGANOIDS_ARGUMENT_STRING "--segment_organoids"
#define DRY_RUN_ARGUMENT_STRING "--dry_run"
#define HELP_ARGUMENT_STRING "--help"
#define IMAGE_SET_ARGUMENT_STRING "--image_set"
#define FEATURE_VECTOR_FILE_NAME_ARGUMENT_STRING "--features"
#define MATCHING_RESULTS_FILE_NAME_ARGUMENT_STRING "--matchings"
#define SKIP_VIZ_ARGUMENT_STRING "--skip_viz"
#define SKIP_MATCHING_ARGUMENT_STRING "--skip_matching"
#define READ_FEATURES_ARGUMENT_STRING "--read_features"
#define READ_MATCHINGS_ARGUMENT_STRING "--read_matchings"
#define WRITE_FEATURES_ARGUMENT_STRING "--write_features"
#define WRITE_MATCHINGS_ARGUMENT_STRING "--write_matchings"
#define PRINT_LOGS_ARGUMENT_STRING "--print_logs"
#define WRITE_EXECUTION_TIME_MEASURMENTS_ARGUMENT_STRING "--write_exec_times"
#define READ_MODEL_PARAMETER_FILE_ARGUMENT_STRING "--read_parameters"
#define MODEL_PARAMETER_FILE_ARGUMENT_STRING "--parameter_file"
#define RUNTIME_LIMIT_ARGUMENT_STRING "--runtime_limit"
#define LEARNING_METRIC_ARGUMENT_STRING "--learning_metric"
#define LEARNING_SEARCH_STRATEGY_ARGUMENT_STRING "--search_strat"
#define REFERENCE_CLUSTERING_FILE_ARGUMENT_STRING "--ref_clustering"
#define READ_LEARNING_TASK_ARGUMENT_STRING "--read_learning_task"
#define LEARNING_TASK_FILE_ARGUMENT_STRING "--learning_task_file"
#define SEARCH_DIM_ORDER_ARGUMENT_STRING "--search_dim_order"
#define SEARCH_STEP_SIZE_ARGUMENT_STRING "--step_size"
#define SIM_ANN_TEMP_ARGUMENT_STRING "--init_temp"
#define SIM_ANN_COOLING_RATE_STRING "--cooling_rate"
#define SIM_ANN_STD_DEV_ARGUMENT_STRING "--std_dev"
#define SIM_ANN_RESTART_THRES_ARGUMENT_STRING "--restart_thresh"

#define COLOR_COST_OFFSET_ARGUMENT_STRING "--color_cost_offset"
#define DIST_COST_OFFSET_ARGUMENT_STRING "--dist_cost_offset"
#define ANGLE_COST_OFFSET_ARGUMENT_STRING "--angle_cost_offset"
#define COLOR_TO_DISTANCE_WEIGHT_ARGUMENT_STRING "--color_to_dist_weight"
#define UNARY_TO_QUADR_WEIGHT_ARGUMENT_STRING "--unary_to_quadr_weight"

#define COLOR_COST_OFFSET_STRING "color_cost_offset"
#define DIST_COST_OFFSET_STRING "dist_cost_offset"
#define ANGLE_COST_OFFSET_STRING "angle_cost_offset"
#define COLOR_TO_DISTANCE_WEIGHT_STRING "color_to_dist_weight"
#define UNARY_TO_QUADR_WEIGHT_STRING "unary_to_quadr_weight"

#define DEC_NOT_MADE_STRING "DEC_TBM"
#define DEC_SAME_STRING "DEC_SME"
#define DEC_UNSURE_STRING "DEC_UNS"
#define DEC_DIFF_STRING "DEC_DIF"

typedef struct Cluster_Member{
    int member_num;
    int cluster_id;
}Cluster_Member;


typedef struct Image_Number_and_Present_Images{
    int image_number;
    bool image_is_present;
    bool mask_is_present;
}Image_Number_and_Present_Images;


Search_Dim get_search_dim_by_name(const char* search_dim_name);

void parse_search_order_string(std::vector<Search_Dim>* search_order,const char* str);

double hh_mm_ss_time_string_to_double(std::string time);

void print_instance_vector_as_vector(std::vector<int>* instance_vector);

void print_instance_array_as_vector(int* instance_vector,int instance_size);

void print_multiple_instance_vectors_as_vectors(std::vector<std::vector<int>*>& instance_vectors, int instance_size);

void write_single_matching_result_table_to_file(std::ofstream& output_file,std::vector<Matching_Result>* matching_results,std::vector<int>& all_image_numbers,Matching_Visualization_Type output_type);

void write_single_feature_vector_to_file(std::ofstream& output_file, Image_Features_Pair& features);

void write_single_feature_point_to_file(std::ofstream& output_file, Feature_Point_Data& feature_point);

void read_single_feature_point(char* line, std::vector<Feature_Point_Data>& target_vector);

void write_assignment_to_file(std::ofstream& output_file,Matching_Result& matching_result);

void read_single_assignment_from_file(std::vector<Matching_Result>* matching_results);

void write_feature_point_selection_parameters_to_file(std::ofstream& output_file, std::string feature_vector_file_name, bool features_were_read_from_file, const Input_Arguments args );

void handle_input_argument(Input_Arguments& input_args, char* current_arg);

void print_missing_argument_value_error(char* arg_name);

void print_unrecognised_argument_error(char* arg_name);

void print_unrecognised_argument_value_error(char* arg_name, char* arg_value);

void parse_true_false_arg_string(char* arg_name, char* arg_value_string, bool& arg);

void print_help_info();

bool check_file_existence_and_compatibility(Input_Arguments& input_arguments);

bool read_image_numbers_from_image_set(Input_Arguments& args, std::vector<int>& image_numbers);

bool read_image_numbers_from_matching_results_file(Input_Arguments& args, std::vector<int>& image_numbers);

bool compare_image_number_vectors(std::vector<int>& numbers_a, std::vector<int>& numbers_b);

void save_parameter_as_group_in_hdf5_file(hid_t parent_group_id, std::string parameter_name, float parameter_value);

void save_string_as_group_in_hdf5_file(hid_t parent_group_id, std::string string_name, std::string string_to_save);

bool strcmp_ignore_leading_hyphens(const char* str_1, const char* str_2);

void uint32_t_set_single_bit_to_zero(uint32_t* word, unsigned int bit_index){
    uint32_t mask = uint32_t(1) << bit_index;

    *word &= ~mask;

}

void uint32_t_set_single_bit_to_one(uint32_t* word, unsigned int bit_index){
    uint32_t mask = uint32_t(1) << bit_index;

    *word |= mask;
}

uint32_t uint32_t_flip_single_bit(uint32_t* word, unsigned int bit_index){
    uint32_t mask = uint32_t(1) << bit_index;

    *word ^= mask;

    return !(*word & mask);
}

uint32_t uint32_t_query_single_bit(uint32_t* word, unsigned int bit_index){
    uint32_t mask = uint32_t(1) << bit_index;

    return (*word & mask);
}

float sample_uniform_0_1() {

    return (float)rand() / (float)RAND_MAX;
}

bool check_if_doubles_are_equal(double a, double b){
    return fabs(a - b) < DBL_EPSILON;
}

void print_int_vector(std::vector<int>& vec){
    for(int i = 0; i < vec.size();i++){
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}


void clear_directory_queue(std::queue<std::filesystem::path>& q) {
    std::queue<std::filesystem::path> empty_queue;
    std::swap(q, empty_queue);
}

void remove_extension(std::string& filename) {
    size_t lastdot = filename.find_last_of(".");

    if (lastdot == std::string::npos){
        return;
    } 

    filename = filename.substr(0, lastdot); 
}

void remove_mask_suffix(std::string& filename) {
    size_t lastdash = filename.find_last_of("_");

    if (lastdash == std::string::npos){
        return;
    } 

    filename = filename.substr(0, lastdash); 
}

void remove_extension_and_channel_number(std::string& filename) {
    size_t lastdot = filename.find_last_of(".");

    if (lastdot == std::string::npos){
        return;
    } 

    lastdot -= 3;

    filename = filename.substr(0, lastdot); 
}

void print_key_press_prompt(){

    std::cout << std::endl;
    std::cout << "Press Esc to quit" << std::endl;
    std::cout << "Press C to calculate the clustering with the current offset/threshold" << std::endl;
    std::cout << "Press R to calculate Prec/Recall/VI for current matching results" << std::endl;
    std::cout << "Press L to toggle use_gurobi_for_clustering" << std::endl;
    std::cout << "Click on an entry in the Matching Matrix window to inspect individual matchings" << std::endl;
    std::cout << std::endl;
}

void create_folder_structure_for_organoid_image(std::filesystem::path base_path, std::string folder_name){

    std::filesystem::path new_directory_path = base_path.parent_path().parent_path();
    new_directory_path.append("segmented_organoids");
    new_directory_path.append(folder_name);

    std::filesystem::create_directory(new_directory_path);

    std::filesystem::path good_images_folder_path = new_directory_path;
    good_images_folder_path.append("good_images");
    std::filesystem::create_directory(good_images_folder_path);

    std::filesystem::path high_aspect_folder_path = new_directory_path;
    high_aspect_folder_path.append("high_aspect_ratio");
    std::filesystem::create_directory(high_aspect_folder_path);

    std::filesystem::path low_pixel_count_folder_path = new_directory_path;
    low_pixel_count_folder_path.append("low_pixel_count");
    std::filesystem::create_directory(low_pixel_count_folder_path);

}

void parse_token(const char* token, Token_Type& t_type, Organoid_Image_Header& header){
    switch(t_type){
        case PLATE_NUM_AND_DATE_TOKEN:

            char plate_num[4];
            plate_num[0] = token[0];
            plate_num[1] = token[1];
            plate_num[2] = token[2];
            plate_num[3] = 0;

            header.plate_number = atoi(plate_num);

            char date_str[3];

            date_str[0] = token[5];
            date_str[1] = token[6];
            date_str[2] = 0;

            header.year_of_experiment = atoi(date_str);

            date_str[0] = token[7];
            date_str[1] = token[8];

            header.month_of_experiment = atoi(date_str);

            date_str[0] = token[9];
            date_str[1] = token[10];

            header.day_of_experiment = atoi(date_str);

            header.replicate =  get_experiment_replicate_from_char(token[11]);
             
            t_type = LIBRARY_TOKEN;
            break;

        case LIBRARY_TOKEN:
            t_type = LIBRARY_NAME_TOKEN;
            break;

        case LIBRARY_NAME_TOKEN:
            t_type = WELL_NUMBER_TOKEN;
            break;

        case WELL_NUMBER_TOKEN:
            t_type = Z_POS_AND_CHANNEL_TOKEN;

            char well_number_str[3];

            header.well_row_char = token[0];

            well_number_str[0] = token[1];
            well_number_str[1] = token[2];
            well_number_str[2] = 0;

            header.well_col_number = atoi(well_number_str);
            break;

        case Z_POS_AND_CHANNEL_TOKEN:

            unsigned int token_str_len = strlen(token);

            for(int i = 0; i < token_str_len; i++){
                if(token[i] == 'Z'){
                    char z_pos_str[3];
                    z_pos_str[0] = token[i + 1];
                    z_pos_str[1] = token[i + 2];
                    z_pos_str[2] = 0;

                    header.z_position = atoi(z_pos_str);

                    i+=2;
                }

                if(token[i] == 'C'){
                    char channel_num_str[3];
                    channel_num_str[0] = token[i + 1];
                    channel_num_str[1] = token[i + 2];
                    channel_num_str[2] = 0;

                    header.channel = get_channel_type_from_int(atoi(channel_num_str));

                    i+=2;
                }
            }
            break;

    }

}
int get_image_number_from_file_name(std::string filename){

    remove_extension(filename);

    char* first_token;
    char* prev_token;
    char delimiter[] = "_";

    char* filename_cstr = (char*)malloc(filename.length() + 1);
    strcpy(filename_cstr,filename.c_str());

    first_token = strtok(filename_cstr, delimiter);
    //std::cout << first_token << std::endl;
    while (true) {
        prev_token = first_token;
        first_token = strtok(NULL, delimiter);

        if(first_token == NULL){
            break;
        }

    }

    free(filename_cstr);

    return atoi(prev_token);

}

bool check_if_file_is_mask_image(std::string filename){

    char* first_token;
    char delimiter[] = "_.";

    char* filename_cstr = (char*)malloc(filename.length() + 1);
    strcpy(filename_cstr,filename.c_str());

    first_token = strtok(filename_cstr, delimiter);
    //std::cout << first_token << std::endl;

    if(strcmp(first_token,"mask") == 0){
        return true;
    }

    while (true) {

        first_token = strtok(NULL, delimiter);

        if(first_token == NULL){
            break;
        }

        if(strcmp(first_token,"mask") == 0){
        return true;
        }

    }

    free(filename_cstr);

    return false;

}

Organoid_Image_Header parse_org_img_filename(std::filesystem::path filepath){

    //remove_extension(filename);

    std::string filename = filepath.filename().string();


    Organoid_Image_Header new_header;

    new_header.full_file_path = filepath;

    char* first_token;
    char delimiter[] = "_-";

    char* filename_cstr = (char*)malloc(filename.length() + 1);
    strcpy(filename_cstr,filename.c_str());

    Token_Type t_type = PLATE_NUM_AND_DATE_TOKEN;

    first_token = strtok(filename_cstr, delimiter);
    //std::cout << first_token << std::endl;

    parse_token(first_token,t_type,new_header);

    while (true) {

        first_token = strtok(NULL, delimiter);

        //std::cout << first_token << std::endl;

        if(first_token == NULL){
            break;
        }

        parse_token(first_token,t_type,new_header);

    }

    free(filename_cstr);

    return new_header;
}

Experiment_Replicate get_experiment_replicate_from_char(char replicate){
    switch(replicate){
        case 'A':
            return EXP_REPLICATE_A;
        case 'B':
            return EXP_REPLICATE_B;
        case 'C':
            return EXP_REPLICATE_C;
        default:
            std::cout << "unknown replicate char: " << replicate << " in function: get_experiment_replicate_from_char" << std::endl;
            return EXP_REPLICATE_UNKNOWN;

    }


}

Channel_Type get_channel_type_from_int(unsigned int channel_int){

    if(channel_int == 1){
        return DAPI_CHANNEL;
    }

    if(channel_int == 2){
        return PDX1_GFP_CHANNEL;
    }

    if(channel_int == 3){
        return NEUROG3_RFP_CHANNEL;
    }

    if(channel_int == 4){
        return Phalloidon_AF647_CHANNEL;
    }

    return INVALID_CHANNEL;
}

std::filesystem::path find_python_scripts_folder(std::filesystem::path base_path) {

    //std::cout << "in find python scripts folder" << std::endl;

    std::queue<std::filesystem::path> remaining_directories;

    for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
        //std::cout << entry << std::endl;
        if (entry.is_directory()) {

            std::string directory_name = entry.path().filename().string();
            
            //std::cout << "directory: " << entry << std::endl;
            //std::cout << directory_name << std::endl;
            //std::cout << entry.path() << std::endl;
            //subdivide_into_single_organoid_images(entry);
            remaining_directories.push(entry);
        }
    }

    while (!remaining_directories.empty()) {
        std::filesystem::path current_path = remaining_directories.front();
        remaining_directories.pop();

        for (const auto& entry : std::filesystem::directory_iterator(current_path)) {
            //std::cout << entry << std::endl;
            if (entry.is_directory()) {
                std::filesystem::path new_path = entry.path();
                std::string directory_name = new_path.filename().string();

                if (directory_name == "src") {
                    clear_directory_queue(remaining_directories);
                }

                if (directory_name == "python_scripts") {
                    return entry.path();
                }

                //std::cout << "directory: " << entry << std::endl;
                //std::cout << directory_name << std::endl;
                //std::cout << entry.path() << std::endl;
                //subdivide_into_single_organoid_images(entry);
                remaining_directories.push(new_path);
            }
        }


    }

    return base_path;

}

float evaluate_logisitic_function(float x, float steepness, float upper_asymptote, float lower_asymptote, float x_value_at_half_of_y_range){

    return lower_asymptote - ((lower_asymptote - upper_asymptote)/(1.0f + exp(-steepness*(x - x_value_at_half_of_y_range))));
}

float evaluate_linear_function(float x, float y_intersect, float x_intersect){

    float m = -y_intersect/x_intersect;

    return m * x + x_intersect;
}

void write_cost_arrays_as_hdf5(andres::Marray<double>& unary_cost_array, andres::Marray<double>& pair_cost_array, std::string filename){

    hid_t hdf_5_file = andres::hdf5::createFile(filename);

    hid_t unary_cost_group = andres::hdf5::createGroup(hdf_5_file,"unary_cost_group");

    andres::hdf5::save(unary_cost_group,"unary_cost_matrix",unary_cost_array);

    andres::hdf5::closeGroup(unary_cost_group);

    hid_t pair_cost_group = andres::hdf5::createGroup(hdf_5_file,"pair_cost_group");

    andres::hdf5::save(pair_cost_group,"pair_cost_matrix",pair_cost_array);

    andres::hdf5::closeGroup(pair_cost_group);

    andres::hdf5::closeFile(hdf_5_file);
}

void list_all_feasible_problem_instances(int rows, int cols){
  
    std::vector<std::vector<int>*> instance_vectors;
    
    std::vector<int> start_instance(rows);

    for(int i = 0; i < rows;i++){
        start_instance[i] = -1;

    }

    int num_created_instances = 0;

    print_instance_vector_as_vector(&start_instance);

    print_instance_vector_as_matrix(&start_instance,rows,cols);

    list_all_subinstances_recursively(&start_instance,0,rows-1,rows,cols,instance_vectors,&num_created_instances);

    print_multiple_instance_vectors_as_matrices(instance_vectors,rows,cols);

    for(int i = 0; i < instance_vectors.size();i++){
        delete(instance_vectors[i]);
    }

    //free(instance_vector);
}

void print_instance_vector_as_vector(std::vector<int>* instance_vector){

    std::cout << "(";

    for(int i = 0; i < instance_vector->size() - 1;i++){

        std::cout << (*instance_vector)[i] << ","; 
    }

    std::cout << (*instance_vector)[instance_vector->size()-1] << ")" << std::endl;
}

void print_instance_array_as_vector(int* instance_vector, int instance_size){

    std::cout << "(";

    for(int i = 0; i < instance_size - 1;i++){

        std::cout << instance_vector[i] << ","; 
    }

    std::cout << instance_vector[instance_size-1] << ")" << std::endl;
}

void print_multiple_instance_vectors_as_vectors(std::vector<std::vector<int>*>& instance_vectors, int instance_size){
    for(int index = 0; index < instance_vectors.size();index++){
        std::vector<int>* instance_vector = instance_vectors[index];        
        std::cout << "(";

        for(int i = 0; i < instance_size - 1;i++){

            std::cout << (*instance_vector)[i] << ","; 
        }

        std::cout << (*instance_vector)[instance_size-1] << ")" << "\t";

    }
    std::cout << std::endl;
}

void print_instance_vector_as_matrix(std::vector<int>* instance_vector, int instance_size_rows, int instance_size_cols){

    for(int i = 0; i < instance_size_rows; i++){

        std::cout << "(";
        for(int j = 0; j < instance_size_cols; j++){
            std::string output_token = "";
            if((*instance_vector)[i] == j){

                output_token += "1"; 
            }else{
                output_token += "0";
            }

            if(j == instance_size_cols-1){
                output_token += ")";
            }else{
                output_token += ",";
            }
            std::cout << output_token;

        }
        std::cout << std::endl;
    }

}

void print_instance_array_as_matrix(int* instance_vector, int instance_size_rows, int instance_size_cols){
    for(int i = 0; i < instance_size_rows; i++){

        std::cout << "(";
        for(int j = 0; j < instance_size_cols; j++){
            std::string output_token = "";
            if(instance_vector[i] == j){

                output_token += "1"; 
            }else{
                output_token += "0";
            }

            if(j == instance_size_cols-1){
                output_token += ")";
            }else{
                output_token += ",";
            }
            std::cout << output_token;

        }
        std::cout << std::endl;
    }

}

void print_multiple_instance_vectors_as_matrices(std::vector<std::vector<int>*>& instance_vectors, int instance_size_rows, int instance_size_cols){

    for(int i = 0; i < instance_size_rows; i++){

        for(int index = 0; index < instance_vectors.size();index++){

            std::cout << "(";

            for(int j = 0; j < instance_size_cols; j++){

                std::vector<int>* instance_vector = instance_vectors[index];

                
                std::string output_token = "";
                if((*instance_vector)[i] == j){

                    output_token += "1"; 
                }else{
                    output_token += "0";
                }

                if(j == instance_size_cols-1){
                    output_token += ")";
                }else{
                    output_token += ",";
                }
                std::cout << output_token;

            }

            std::cout << "\t";

        }
        std::cout << std::endl;
    }

}


void print_multiple_instance_arrays_as_matrices(std::vector<int*>& instance_arrays, int instance_size_rows, int instance_size_cols){
    for(int i = 0; i < instance_size_rows; i++){

        for(int index = 0; index < instance_arrays.size();index++){

            std::cout << "(";

            for(int j = 0; j < instance_size_cols; j++){

                int* instance_array = instance_arrays[index];

                
                std::string output_token = "";
                if(instance_array[i] == j){

                    output_token += "1"; 
                }else{
                    output_token += "0";
                }

                if(j == instance_size_cols-1){
                    output_token += ")";
                }else{
                    output_token += ",";
                }
                std::cout << output_token;

            }

            std::cout << "\t";

        }
        std::cout << std::endl;
    }
}

// this functions allocates heap memory for the instances stored in the all_feasible_instances vector. Make sure to free them when calling this function.
void list_all_subinstances_recursively(std::vector<int>* instance_vector,int start_row, int last_row, int total_rows, int total_cols, std::vector<std::vector<int>*>& all_feasible_instances, int* num_created_instances){
    //the num_created_instances parameter is only for debugging purposes to make sure we delete as many instances as we create.

    if(last_row >= total_rows || start_row > total_rows || start_row > last_row){
        return;
    }

    //std::cout << "default input: " << start_row << " " << last_row << " " << total_rows << " " << total_cols << std::endl;

    std::vector<int> local_instance_vector = *instance_vector;

    //std::cout << instance_vector->size() << std::endl;;

    //std::cout << std::endl;
    //std::cout << "all sub instances" << std::endl;

    //print_instance_vector_as_matrix(&local_instance_vector,total_rows,total_cols); 

    //std::cout << std::endl;
    //for(int row = 0; row < total_rows; row++){
    for(int col = -1; col < total_cols; col++){
        if(!check_if_instance_vector_column_is_occupied(local_instance_vector,col)){
            local_instance_vector[start_row] = col;

            if(start_row == last_row){
                //print_instance_vector_as_matrix(&local_instance_vector,total_rows,total_cols);
                std::vector<int>* persistent_instance_vector = new std::vector<int>;
                (*num_created_instances)++;
                *persistent_instance_vector = local_instance_vector;

                all_feasible_instances.push_back(persistent_instance_vector);

            }

            list_all_subinstances_recursively(&local_instance_vector,start_row+1,last_row,total_rows,total_cols, all_feasible_instances,num_created_instances);

            //std::cout << std::endl;
            local_instance_vector[start_row] = (*instance_vector)[start_row];
        }        
    }
    //}
}

bool check_if_instance_vector_column_is_occupied(std::vector<int>& instance_vector, int col){
    if(col == -1){
        return false;
    }

    for(int i = 0; i < instance_vector.size(); i++){

        if(instance_vector[i] == col){
            return true;
        }

    }

    return false;
}

void list_all_subinstances_recursively_custom_memory(Memory_Element* instance_vector,int start_row, int last_row, int total_rows, int total_cols, std::vector<Memory_Element>& all_feasible_instances, int* num_created_instances, Memory_Manager_Fixed_Size* mem_manager){

    if(last_row >= total_rows || start_row > total_rows || start_row > last_row){
        return;
    }

    //std::cout << "custom input: " << start_row << " " << last_row << " " << total_rows << " " << total_cols << std::endl;

    int original_value = ((int*)(instance_vector->memory_pointer))[start_row];

    Memory_Element local_instance_vector = *instance_vector;

    for(int col = -1; col < total_cols; col++){
        if(!check_if_instance_array_column_is_occupied((int*)local_instance_vector.memory_pointer,total_rows,col)){
            MEM_ELEM_WRITE(local_instance_vector,int,start_row,col);

            if(start_row == last_row){
                Memory_Element persistent_instance = allocate_new_memory_element(mem_manager);
                //std::cout << persistent_instance.bit_index << " " << persistent_instance.leaf_index << std::endl;
                (*num_created_instances)++;
                memcpy(persistent_instance.memory_pointer,local_instance_vector.memory_pointer,mem_manager->size_of_single_element);

                //print_instance_array_as_vector((int*)persistent_instance.memory_pointer,total_rows);

                all_feasible_instances.push_back(persistent_instance);
            }

            list_all_subinstances_recursively_custom_memory(&local_instance_vector,start_row+1,last_row,total_rows,total_cols, all_feasible_instances,num_created_instances,mem_manager);
            //local_instance_vector[start_row] = (*instance_vector)[start_row];
            MEM_ELEM_WRITE(local_instance_vector,int,start_row,original_value);
        }        
    }
}


bool check_if_instance_array_column_is_occupied(int* instance_array, int instance_size, int col){
    if(col == -1){
        return false;
    }

    for(int i = 0; i < instance_size; i++){

        if(instance_array[i] == col){
            return true;
        }

    }

    return false;
}

bool compare_two_instance_vectors(std::vector<int>* instance_1, std::vector<int>* instance_2){

    if(instance_1->size() != instance_2->size()){
        std::cout << "Instance sizes did not match" << std::endl;
        return false;
    }

    for(int i = 0; i < instance_1->size(); i++){
        if((*instance_1)[i] != (*instance_2)[i]){
            return false;
        }

    }

    return true;

}

bool compare_two_instance_arrays(int* instance_1, int* instance_2, int instance_size){

    for(int i = 0; i < instance_size; i++){
        if(instance_1[i] != instance_2[i]){
            return false;
        }

    }

    return true;

}

double get_difference_between_two_angles(double alpha, double beta){

    return 180.0 - abs(abs(alpha - beta) - 180.0);

}

double get_signed_difference_between_two_angles_radians(double alpha, double beta){
    double delta = beta - alpha; 
    
    delta = atan2(sin(delta),cos(delta));

    return delta;
}

double get_signed_difference_between_two_angles(double alpha, double beta){

    double delta = ((beta - alpha) * PI) / 180.0;

    delta = atan2(sin(delta),cos(delta)); 

    delta = (delta * 180.0) / PI;

    return delta;

}

cv::Vec2f rotate_vec_2d_by_radians(cv::Vec2f& vector, double alpha){

    cv::Mat affine_vector(cv::Size(1,3),CV_64F);

    affine_vector.at<double>(0,0) = vector[0];
    affine_vector.at<double>(0,1) = vector[1];
    affine_vector.at<double>(0,2) = 0;

    cv::Point2f rot_center(0.0f,0.0f);

    affine_vector = cv::getRotationMatrix2D(rot_center,alpha,1.0) * affine_vector;

    cv::Vec2f rotated_vector(affine_vector.at<double>(0,0),affine_vector.at<double>(0,1));

    return rotated_vector;
}

void assign_value_to_single_matching_result(Matching_Result& mr, Matching_Visualization_Type type, double value){
    switch(type){
        case MATCHING_VIZ_QUADRATIC_OPTIMAL:
            mr.rel_quadr_cost_optimal = value;
        break;

        case MATCHING_VIZ_QUADRATIC:
            mr.rel_quadr_cost = value;
        break;

        case MATCHING_VIZ_LIN_FEATURE:
            mr.linear_cost_per_feature = value;
        break;

        case MATCHING_VIZ_LIN_CANDIDATE:
            mr.linear_cost_per_candidate = value;
        break;

        default:
        break;

    }

}

void add_single_matching_result_to_vector(std::vector<Matching_Result>* matching_results, int id_1, int id_2, Matching_Visualization_Type type, double value, bool sparse_fill){

    if(id_1 == id_2){
        return;
    }

    bool found_existing_matching_result = false;

    if(!sparse_fill){

        for(int i = 0; i < matching_results->size();i++){

            if(((*matching_results)[i].id_1 == id_1 && (*matching_results)[i].id_2 == id_2)){

                assign_value_to_single_matching_result((*matching_results)[i],type,value);

                found_existing_matching_result = true;
                break;

            }
        }
    }

    if(!found_existing_matching_result){

        Matching_Result new_mr;
        new_mr.id_1 = id_1;
        new_mr.id_2 = id_2;

        new_mr.set_id_1 = -1;
        new_mr.set_id_2 = -1;

        new_mr.assignment = nullptr;

        assign_value_to_single_matching_result(new_mr,type,value);

        new_mr.additional_viz_data_id1_to_id2 = new Matching_Result_Additional_Viz_Data;
        //new_mr.additional_viz_data_id1_to_id2->assignment = nullptr;
        new_mr.additional_viz_data_id1_to_id2->candidate_image = nullptr;
        new_mr.additional_viz_data_id1_to_id2->feature_image = nullptr;
        new_mr.additional_viz_data_id1_to_id2->features = nullptr;
        new_mr.additional_viz_data_id1_to_id2->assigned_candidates = nullptr;

        new_mr.additional_viz_data_id2_to_id1 = new Matching_Result_Additional_Viz_Data;
        //new_mr.additional_viz_data_id2_to_id1->assignment = nullptr;
        new_mr.additional_viz_data_id2_to_id1->candidate_image = nullptr;
        new_mr.additional_viz_data_id2_to_id1->feature_image = nullptr;
        new_mr.additional_viz_data_id2_to_id1->features = nullptr;
        new_mr.additional_viz_data_id2_to_id1->assigned_candidates = nullptr;

        matching_results->push_back(new_mr);
    }

}

void write_single_feature_point_to_file(std::ofstream& output_file, Feature_Point_Data& feature_point){

    output_file << feature_point.angle << ",";
    output_file << feature_point.channel << ",";
    output_file << feature_point.col << ",";
    output_file << feature_point.row << ",";
    output_file << feature_point.local_mean << ",";
    output_file << feature_point.local_mean_forground_only << ",";
    output_file << (int)feature_point.local_search_dim << ",";
    output_file << feature_point.local_std_dev << ",";
    output_file << feature_point.local_std_dev_forground_only << ",";
    output_file << feature_point.normalize_peak_value << ",";
    output_file << feature_point.peak_value << ",";
    output_file << feature_point.relative_distance_center_boundary << ",";
    output_file << feature_point.relative_distance_center_max_distance << std::endl;

}

void write_single_feature_vector_to_file(std::ofstream& output_file, Image_Features_Pair& features){

    output_file << "features_begin," << features.image_number << std::endl;

    for(int i = 0; i < features.features->size();i++){
        Feature_Point_Data current_feature_point = (*(features.features))[i];

        write_single_feature_point_to_file(output_file,current_feature_point);
    }

    output_file << "features_end," << features.image_number << std::endl;

    output_file << "candidates_begin," << features.image_number << std::endl;

    for(int i = 0; i < features.candidates->size();i++){
        Feature_Point_Data current_candidate_point = (*(features.candidates))[i];

        write_single_feature_point_to_file(output_file,current_candidate_point);
    }

    output_file << "candidates_end," << features.image_number << std::endl;
}

void write_single_matching_result_table_to_file(std::ofstream& output_file,std::vector<Matching_Result>* matching_results,std::vector<int>& all_image_numbers,Matching_Visualization_Type output_type){

    switch(output_type){
        case MATCHING_VIZ_QUADRATIC_OPTIMAL:
            output_file << "\noptimal_quadratic_matching,";
        break;

        case MATCHING_VIZ_QUADRATIC:
            output_file << "\nquadratic_matching,";
        break;

        case MATCHING_VIZ_LIN_FEATURE:
            output_file <<  "\nlinear_by_feature,";
        break;

        case MATCHING_VIZ_LIN_CANDIDATE:
            output_file << "\nlinear_by_candidate,";
        break;

        default:
            output_file << "\n,\n";
        break;

    }

    for(int i = 0; i < all_image_numbers.size();i++){
        output_file << all_image_numbers[i]; 

        if(i < all_image_numbers.size() - 1){
            output_file << ",";
        }
    }

    output_file << "\n";

    for(int i = 0; i < all_image_numbers.size();i++){
        int current_id_1 = all_image_numbers[i];

        output_file << all_image_numbers[i] << ",";

        for(int j = 0; j < all_image_numbers.size();j++){
            int current_id_2 = all_image_numbers[j];

            if(current_id_1 == current_id_2){
                output_file << "S";

                if(j < all_image_numbers.size() - 1){
                    output_file << ",";
                }

                continue;
            }

            for(int k = 0; k < matching_results->size();k++){

                Matching_Result current_mr = (*matching_results)[k];

                if((current_mr.id_1 == current_id_1 && current_mr.id_2 == current_id_2)){

                    double output_value;

                    switch(output_type){
                        case MATCHING_VIZ_QUADRATIC_OPTIMAL:
                            output_value = current_mr.rel_quadr_cost_optimal;
                        break;

                        case MATCHING_VIZ_QUADRATIC:
                            output_value = current_mr.rel_quadr_cost;
                        break;

                        case MATCHING_VIZ_LIN_FEATURE:
                            output_value = current_mr.linear_cost_per_feature;
                        break;

                        case MATCHING_VIZ_LIN_CANDIDATE:
                            output_value = current_mr.linear_cost_per_candidate;
                        break;

                        default:
                            output_value = current_mr.rel_quadr_cost;
                        break;

                    }

                    output_file << output_value;

                    if(j < all_image_numbers.size() - 1){
                        output_file << ",";
                    }

                }

            }

        } 

        if(i < all_image_numbers.size() - 1){
            output_file << "\n";
        }
    }

}

void write_matching_results_to_file(std::filesystem::path output_file_path, std::vector<Matching_Result>* matching_results, All_Cost_Parameters& cost_params, std::vector<Image_Features_Pair>& all_feature_vectors, std::string feature_vector_file_name, bool features_were_read_from_file, const Input_Arguments args){

    std::vector<int> all_image_numbers;

    for(int i = 0; i < matching_results->size();i++){

        int image_id_1 = (*matching_results)[i].id_1;
        int image_id_2 = (*matching_results)[i].id_2;

        bool id_1_already_added = false;
        bool id_2_already_added = false;

        for(int j = 0; j < all_image_numbers.size();j++){
            int current_id = all_image_numbers[j];

            if(current_id == image_id_1){
                id_1_already_added = true;
            }

            if(current_id == image_id_2){
                id_2_already_added = true;
            }

            if(id_1_already_added && id_2_already_added){
                break;
            }

        }

        if(!id_1_already_added){
            //std::cout << image_id_1 << " added" << std::endl;
            all_image_numbers.push_back(image_id_1);
        }

        if(!id_2_already_added){
            //std::cout << image_id_2 << " added" << std::endl;
            all_image_numbers.push_back(image_id_2);
        }

    }

    std::sort(all_image_numbers.begin(),all_image_numbers.end());

    std::ofstream output_file;

    output_file.open(output_file_path);

    write_single_matching_result_table_to_file(output_file,matching_results,all_image_numbers,MATCHING_VIZ_QUADRATIC);

    write_single_matching_result_table_to_file(output_file,matching_results,all_image_numbers,MATCHING_VIZ_LIN_FEATURE);

    write_single_matching_result_table_to_file(output_file,matching_results,all_image_numbers,MATCHING_VIZ_LIN_CANDIDATE);

    write_single_matching_result_table_to_file(output_file,matching_results,all_image_numbers,MATCHING_VIZ_QUADRATIC_OPTIMAL);
    
    output_file << "\n";

    for(int i = 0; i < matching_results->size();i++){
        write_assignment_to_file(output_file,(*matching_results)[i]);
    }

    output_file << "cost_double_assignment," << cost_params.cost_double_assignment << "\n";
    output_file << "cost_mismatching_channels," << cost_params.cost_mismatching_channels << "\n";
    output_file << COLOR_COST_OFFSET_STRING << "," << cost_params.color_offset << "\n"; 
    output_file << DIST_COST_OFFSET_STRING << "," << cost_params.dist_offset << "\n";
    output_file << ANGLE_COST_OFFSET_STRING << "," << cost_params.angle_offset << "\n";
    output_file << COLOR_TO_DISTANCE_WEIGHT_STRING << "," << cost_params.color_to_dist_weight << "\n";
    output_file << UNARY_TO_QUADR_WEIGHT_STRING << "," << cost_params.unary_to_to_quadr_weight << "\n";

    write_feature_point_selection_parameters_to_file(output_file,feature_vector_file_name,features_were_read_from_file, args);

    output_file.close();

}



void write_matching_results_as_hdf5(std::string filename, std::vector<Matching_Result>* matching_results, All_Cost_Parameters& cost_params, std::vector<Image_Features_Pair>& all_feature_vectors, std::string feature_vector_file_name, bool features_were_read_from_file, const Input_Arguments args){


    size_t shape_mm[] = {all_feature_vectors.size() , all_feature_vectors.size()};

	andres::Marray<float> marray_matching_results_matrix(shape_mm,shape_mm+2,0,andres::FirstMajorOrder);

    std::vector<int> all_organoid_image_numbers;

    for(int i = 0; i < all_feature_vectors.size(); i++){
        all_organoid_image_numbers.push_back(all_feature_vectors[i].image_number);

    }

    std::sort(all_organoid_image_numbers.begin(),all_organoid_image_numbers.end());
    
    size_t shape_img_nums[] = {all_organoid_image_numbers.size()};

    andres::Marray<float> marray_img_nums(shape_img_nums,shape_img_nums+1,0,andres::FirstMajorOrder);

    for(int i = 0; i < all_organoid_image_numbers.size(); i++){
        int img_num_1 = all_organoid_image_numbers[i];

        marray_img_nums(i) = img_num_1;

        for(int j = 0; j < all_organoid_image_numbers.size(); j++){
            int img_num_2 = all_organoid_image_numbers[j];

            for(int k = 0; k < matching_results->size(); k++){
                Matching_Result cur_mr = (*matching_results)[k];

                if(cur_mr.id_1 == img_num_1 && cur_mr.id_2 == img_num_2){
                    float value_ij = cur_mr.rel_quadr_cost;

                    marray_matching_results_matrix(i,j) = value_ij;
                    break;
                }
            }
        }
    }


    std::filesystem::path data_folder = get_data_folder_path();
    data_folder.append("matching_results");
    data_folder.append(filename);

    hid_t hdf_5_file = andres::hdf5::createFile(data_folder.string());

    hid_t img_num_group = andres::hdf5::createGroup(hdf_5_file,"organoid_img_numbers");

    andres::hdf5::save(img_num_group,"organoid_img_numbers",marray_img_nums);

    andres::hdf5::closeGroup(img_num_group);

    hid_t matching_matrix_group = andres::hdf5::createGroup(hdf_5_file,"matching_matrix");

    andres::hdf5::save(matching_matrix_group,"matching_matrix",marray_matching_results_matrix);

    andres::hdf5::closeGroup(matching_matrix_group);

    hid_t assignments_group = andres::hdf5::createGroup(hdf_5_file,"assignments_group");

    for(int i = 0; i < matching_results->size(); i++){

        Matching_Result cur_mr = (*matching_results)[i];

        std::string new_group_name = "assignment_" + std::to_string(cur_mr.id_1) + "_" + std::to_string(cur_mr.id_2);

        hid_t new_assignment_group = andres::hdf5::createGroup(assignments_group,new_group_name);

        size_t shape_iassign[] = {cur_mr.assignment->size() + 2};

        andres::Marray<int> current_assignment(shape_iassign,shape_iassign+1,0,andres::FirstMajorOrder);

        current_assignment(0) = cur_mr.id_1;
        current_assignment(1) = cur_mr.id_2;

        for(int j = 0; j < cur_mr.assignment->size();j++){

            current_assignment(j + 2) = (*(cur_mr.assignment))[j];
        }

        andres::hdf5::save(new_assignment_group,new_group_name,current_assignment);

        andres::hdf5::closeGroup(new_assignment_group);
    }

    andres::hdf5::closeGroup(assignments_group);

    hid_t parameter_group = andres::hdf5::createGroup(hdf_5_file,"parameter_group");

    save_parameter_as_group_in_hdf5_file(parameter_group,COLOR_COST_OFFSET_STRING,cost_params.color_offset);
    save_parameter_as_group_in_hdf5_file(parameter_group,DIST_COST_OFFSET_STRING,cost_params.dist_offset);
    save_parameter_as_group_in_hdf5_file(parameter_group,ANGLE_COST_OFFSET_STRING,cost_params.angle_offset);
    save_parameter_as_group_in_hdf5_file(parameter_group,COLOR_TO_DISTANCE_WEIGHT_STRING,cost_params.color_to_dist_weight);
    save_parameter_as_group_in_hdf5_file(parameter_group,UNARY_TO_QUADR_WEIGHT_STRING,cost_params.unary_to_to_quadr_weight);


    if(features_were_read_from_file){
        save_string_as_group_in_hdf5_file(parameter_group,"feature_vector_file",feature_vector_file_name);
    }

    save_string_as_group_in_hdf5_file(parameter_group,"image_set",args.image_set_name);


    andres::hdf5::closeGroup(parameter_group);    

    andres::hdf5::closeFile(hdf_5_file);

}

void save_parameter_as_group_in_hdf5_file(hid_t parent_group_id, std::string parameter_name, float parameter_value){

    hid_t parameter_group = andres::hdf5::createGroup(parent_group_id,parameter_name);

    size_t shape_param_group[] = {1};

    andres::Marray<float> param_value(shape_param_group,shape_param_group+1,0,andres::FirstMajorOrder);

    param_value(0) = parameter_value;

    andres::hdf5::save(parameter_group,parameter_name,param_value);

    andres::hdf5::closeGroup(parameter_group);

}

void save_string_as_group_in_hdf5_file(hid_t parent_group_id, std::string string_name, std::string string_to_save){

    hid_t string_group = andres::hdf5::createGroup(parent_group_id,string_name);

    size_t shape_string_group[] = {string_to_save.size()};

    andres::Marray<char> string_as_array(shape_string_group,shape_string_group+1,0,andres::FirstMajorOrder);

    for(int i = 0; i < string_to_save.size();i++){
        string_as_array(i) = string_to_save[i];
    }

    andres::hdf5::save(string_group,string_name,string_as_array);

    andres::hdf5::closeGroup(string_group);

}

void write_all_feature_vectors_to_file(std::string file_name, std::vector<Image_Features_Pair>& all_feature_vectors, const Input_Arguments args){

    std::filesystem::path output_file_path = get_data_folder_path();

    output_file_path.append("feature_vectors");

    output_file_path.append(file_name);

    std::ofstream output_file;

    output_file.open(output_file_path);

    write_feature_point_selection_parameters_to_file(output_file,file_name,false, args);

    for(int i = 0; i < all_feature_vectors.size();i++){
        write_single_feature_vector_to_file(output_file,all_feature_vectors[i]);

    }

    output_file.close();
}

void read_matching_results_from_file(std::filesystem::path file_path, std::vector<Matching_Result>* matching_results, All_Cost_Parameters& cost_params, All_Feature_Point_Selection_Parameters& feature_point_selection_params){

    std::ifstream input_file;

    input_file.open(file_path);

    if (input_file.fail()) {
        std::cout << "File: " << file_path << " to read matching results from COULD NOT BE OPENED" << std::endl;
        return;
    }

    matching_results->clear();

    const int linebuffer_size = 16777216;
    //char line_buffer[linebuffer_size];

    char* line_buffer = (char*)malloc(sizeof(char) * linebuffer_size);

    std::vector<int> all_image_numbers;

    char* first_token;
    char delimiter[] = ",:";

    Matching_Visualization_Type type;
    bool line_is_first_row = false;

    bool all_image_numbers_vector_filled = false;

    bool is_first_token_of_line = false;
    int num_token_in_line = 0;
    int image_number_of_row = -1;

    bool reached_end_of_matching_results = false;

    bool currently_reading_selection_parameters = false;

    //int linecounter = 0;

    while(!input_file.eof() && !reached_end_of_matching_results){
        input_file.getline(line_buffer,linebuffer_size);
        //std::cout << line_buffer << std::endl;

        //std::cout << linecounter++ << std::endl; 

        first_token = strtok(line_buffer, delimiter);

        is_first_token_of_line = true;


        while (true) {

            if(first_token == NULL){
                break;
            }

            if(line_is_first_row && !all_image_numbers_vector_filled){

                all_image_numbers.push_back(atoi(first_token));

            }else{
                if(strcmp(first_token,"quadratic_matching") == 0){
                    type = MATCHING_VIZ_QUADRATIC;
                    line_is_first_row = true;
                    //std::cout << "quadratic_matching" << std::endl;
                }else if(strcmp(first_token,"linear_by_feature") == 0){
                    type = MATCHING_VIZ_LIN_FEATURE;
                    line_is_first_row = true;
                    //std::cout << "linear_by_feature" << std::endl;
                }else if(strcmp(first_token,"linear_by_candidate") == 0){
                    type = MATCHING_VIZ_LIN_CANDIDATE;
                    line_is_first_row = true;
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,"optimal_quadratic_matching") == 0){
                    type = MATCHING_VIZ_QUADRATIC_OPTIMAL;
                    line_is_first_row = true;
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,"cost_double_assignment") == 0){
                    first_token = strtok(NULL, delimiter);
                    cost_params.cost_double_assignment = atof(first_token);
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,"cost_mismatching_channels") == 0){
                    first_token = strtok(NULL, delimiter);
                    cost_params.cost_mismatching_channels = atof(first_token);
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,COLOR_COST_OFFSET_STRING) == 0){
                    first_token = strtok(NULL, delimiter);
                    cost_params.color_offset = atof(first_token);
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,DIST_COST_OFFSET_STRING) == 0){
                    first_token = strtok(NULL, delimiter);
                    cost_params.dist_offset = atof(first_token);
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,ANGLE_COST_OFFSET_STRING) == 0){
                    first_token = strtok(NULL, delimiter);
                    cost_params.angle_offset = atof(first_token);
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,COLOR_TO_DISTANCE_WEIGHT_STRING) == 0){
                    first_token = strtok(NULL, delimiter);
                    cost_params.color_to_dist_weight = atof(first_token);
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,UNARY_TO_QUADR_WEIGHT_STRING) == 0){
                    first_token = strtok(NULL, delimiter);
                    cost_params.unary_to_to_quadr_weight = atof(first_token);
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,"assignment") == 0){
                    read_single_assignment_from_file(matching_results);
                    break;
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,"features_begin") == 0){
                    reached_end_of_matching_results = true;
                    break;
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,"begin_selection_parameters") == 0){
                    currently_reading_selection_parameters = true;
                    //std::cout << "begin" << std::endl;
                    break;
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(strcmp(first_token,"end_selection_parameters") == 0){
                    currently_reading_selection_parameters = false;
                    //std::cout << "end" << std::endl;
                    break;
                    //std::cout << "linear_by_candidate" << std::endl;
                }else if(currently_reading_selection_parameters){
                    //std::cout << "break" << std::endl;
                    break;
                }else{
                    //here be numbers

                    if(READ_MINIMAL_MATCHING_RESULTS && type != MATCHING_VIZ_QUADRATIC){
                        break;
                    }

                    if(is_first_token_of_line){
                        image_number_of_row = atoi(first_token);
                        is_first_token_of_line = false;
                    }else{
                        double value = atof(first_token);

                        add_single_matching_result_to_vector(matching_results,image_number_of_row,all_image_numbers[num_token_in_line],type,value,READ_MINIMAL_MATCHING_RESULTS);

                        num_token_in_line++;

                    }

                }

            }

            first_token = strtok(NULL, delimiter);

        }

        if(line_is_first_row){
            line_is_first_row = false;

            all_image_numbers_vector_filled = true;
        }

        num_token_in_line = 0;

    }

    free(line_buffer);

    std::cout << "finished reading matching results " << std::endl;

}
void read_single_feature_point(char* first_token, std::vector<Feature_Point_Data>& target_vector){
    char delimiter[] = ",:";

    Feature_Point_Data feature_point;

    feature_point.angle = atof(first_token);

    first_token = strtok(NULL, delimiter);

    int channel = atoi(first_token);
    if(channel == 0){
        feature_point.channel = DAPI_CHANNEL;
    }else if(channel == 1){
        feature_point.channel = PDX1_GFP_CHANNEL;
    }else if(channel == 2){
        feature_point.channel = NEUROG3_RFP_CHANNEL;
    }else if(channel == 3){
        feature_point.channel = Phalloidon_AF647_CHANNEL;
    }else{
        feature_point.channel = INVALID_CHANNEL;
        std::cout << "Found INVALID CHANNEL while reading feature point";
    }

    first_token = strtok(NULL, delimiter);
    feature_point.col = atoi(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.row = atoi(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.local_mean = atof(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.local_mean_forground_only = atof(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.local_search_dim = (uint8_t)atoi(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.local_std_dev = atof(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.local_std_dev_forground_only = atof(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.normalize_peak_value = atof(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.peak_value = (uint16_t)atoi(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.relative_distance_center_boundary = atof(first_token);

    first_token = strtok(NULL, delimiter);
    feature_point.relative_distance_center_max_distance = atof(first_token);

    target_vector.push_back(feature_point);
}

bool read_feature_vector_from_file(std::filesystem::path file_path, int image_number, std::vector<Feature_Point_Data>& output_feature_points, std::vector<Feature_Point_Data>& output_candidate_points){
    std::ifstream input_file;

    input_file.open(file_path);

    if (input_file.fail()) {
        std::cout << "File: " << file_path << " to read feature vector from COULD NOT BE OPENED" << std::endl;
        return false;
    }

    const int linebuffer_size = 16777216;
    //char line_buffer[linebuffer_size];

    char* line_buffer = (char*)malloc(sizeof(char) * linebuffer_size);

    char* first_token;
    char delimiter[] = ",:";

    bool finished_reading_features = false;
    bool currently_reading_features = false;
    bool finished_reading_candidates = false;
    bool currently_reading_candidates = false;
    //bool finished_reading_selection_parameters = false;
    //bool currentlz_reading_selection_parameters = false;

    while(!input_file.eof()){
        input_file.getline(line_buffer,linebuffer_size);
        //std::cout << line_buffer << std::endl;

        //std::cout << line_buffer << std::endl;

        first_token = strtok(line_buffer, delimiter);


        if(first_token == NULL){
            continue;
        }

        if(finished_reading_candidates && finished_reading_features){
            return true;
            break;
        }


        if(currently_reading_features){

            if(strcmp(first_token,"features_end") == 0){
                finished_reading_features = true;
                currently_reading_features = false;
                continue;
            }

            read_single_feature_point(first_token,output_feature_points);

            continue;
        }

        if(currently_reading_candidates){
            
            if(strcmp(first_token,"candidates_end") == 0){
                finished_reading_candidates = true;
                currently_reading_candidates = false;
                continue;
            }

            read_single_feature_point(line_buffer,output_candidate_points);

            continue;
        }

        if(strcmp(first_token,"features_begin") == 0){
            first_token = strtok(NULL, delimiter);

            int found_image_number = atoi(first_token);

            if(found_image_number == image_number){
                currently_reading_features = true;
                continue;
            }

        }else if(strcmp(first_token,"candidates_begin") == 0){
            first_token = strtok(NULL, delimiter);

            int found_image_number = atoi(first_token);

            if(found_image_number == image_number){
                currently_reading_candidates = true;
                continue;
            }

        }


        

            //first_token = strtok(NULL, delimiter);
    }

    free(line_buffer);

    if(finished_reading_candidates && finished_reading_features){
        return true;
    }else{
        return false;
    }

}

void write_assignment_to_file(std::ofstream& output_file,Matching_Result& matching_result){

    if(matching_result.assignment == nullptr){
        std::cout << "assignment in write_assignment_to_file was NULL" << std::endl;
        return;
    }

    std::vector<int>* current_assignment = matching_result.assignment;

    if(current_assignment->size() == 0){
        std::cout << "assignment in write_assignment_to_file was empty" << std::endl;
        return;
    }

    output_file << "assignment," << matching_result.id_1 << "," << matching_result.id_2 << ",";

    output_file << (*current_assignment)[0];

    for(int i = 1; i < current_assignment->size();i++){
        output_file << "," << (*current_assignment)[i];
    }

    output_file << std::endl;

}

void read_single_assignment_from_file(std::vector<Matching_Result>* matching_results){

    char delimiter[] = ",";
    char* next_token;

    std::vector<int> assignment_vector;// = new std::vector<int>;

    next_token = strtok(NULL,delimiter);

    int image_id_1 = atoi(next_token);

    next_token = strtok(NULL,delimiter);

    int image_id_2 = atoi(next_token);

    next_token = strtok(NULL,delimiter);

    while(next_token != NULL){
        assignment_vector.push_back(atoi(next_token));

        next_token = strtok(NULL,delimiter);
    }

    for(int i = 0; i < matching_results->size();i++){

        Matching_Result current_mr = (*matching_results)[i];

        if(current_mr.id_1 == image_id_1 && current_mr.id_2 == image_id_2 ){

            //if((*matching_results)[i].additional_viz_data_id1_to_id2 != nullptr){
                if((*matching_results)[i].assignment == nullptr){
                    (*matching_results)[i].assignment = new std::vector<int>;
                    *((*matching_results)[i].assignment) = assignment_vector;
                }
            //}
        }

        /*
        if(current_mr.id_1 == image_id_2 && current_mr.id_2 == image_id_1 ){

            if((*matching_results)[i].additional_viz_data_id2_to_id1 != nullptr){
                if((*matching_results)[i].additional_viz_data_id2_to_id1->assignment == nullptr){
                    (*matching_results)[i].additional_viz_data_id2_to_id1->assignment = new std::vector<int>;
                    *((*matching_results)[i].additional_viz_data_id2_to_id1->assignment) = assignment_vector;
                }
            }
        }
        */

    }
}


void print_cost_parameters(All_Cost_Parameters* cost_params){

    std::cout << "Cost parameters:" << std::endl;
    std::cout << "cost_double_assignment: " << cost_params->cost_double_assignment <<std::endl;
    std::cout << "cost_mismatching_channels: " << cost_params->cost_mismatching_channels <<std::endl;
    std::cout << COLOR_COST_OFFSET_STRING << ": " << cost_params->color_offset <<std::endl;
    std::cout << DIST_COST_OFFSET_STRING << ": " << cost_params->dist_offset <<std::endl;
    std::cout << ANGLE_COST_OFFSET_STRING << ": " << cost_params->angle_offset <<std::endl;
    std::cout << COLOR_TO_DISTANCE_WEIGHT_STRING << ": " << cost_params->color_to_dist_weight <<std::endl;
    std::cout << UNARY_TO_QUADR_WEIGHT_STRING << ": " << cost_params->unary_to_to_quadr_weight <<std::endl;
}

size_t get_next_bigger_power_of_two(size_t value){

    unsigned int num_shifts = 0;

    while(value > 0){
        value = value >> 1;
        num_shifts++;
    }

    return 1 << num_shifts;
}

bool check_if_uint_is_power_of_two(unsigned int n){

    return ((n & (n-1)) == 0) && n;
}

void write_single_percentage_output_and_update_vectors(std::vector<Op_Numbers_And_Runtime>& op_numbers, int i, std::ofstream& output_file, Algorithm_Type type,std::vector<double>& minimal_percentage,std::vector<double>& average_percentage,std::vector<double>& maximal_percentage,std::vector<double>& minimal_time_per_percentage,std::vector<double>& average_time_per_percentage,std::vector<double>& maximal_time_per_percentage){
    if(op_numbers[i].runtime[type] > 0){

        double result_as_percentage = op_numbers[i].normalized_results[type]/op_numbers[i].normalized_results[GUROBI_ALGORITHM];

        if(result_as_percentage < minimal_percentage[type]){
            minimal_percentage[type] = result_as_percentage;
        }


        if(result_as_percentage > maximal_percentage[type]){
            maximal_percentage[type] = result_as_percentage;
        }

        average_percentage[type] += result_as_percentage;

        double time_per_percent = op_numbers[i].runtime[type] /(result_as_percentage * 100);

        if(time_per_percent < minimal_time_per_percentage[type]){
            minimal_time_per_percentage[type] = time_per_percent;
        }


        if(time_per_percent > maximal_time_per_percentage[type]){
            maximal_time_per_percentage[type] = time_per_percent;
        }

        average_time_per_percentage[type] += time_per_percent;

        output_file  << result_as_percentage << "," << time_per_percent << ",";
    }

}

void write_ROC_to_csv_file(std::string file_name, std::vector<TPR_FPR_Tuple>& roc){
    std::filesystem::path output_file_path = get_data_folder_path();

    output_file_path.append("execution_times");

    output_file_path.append(file_name);

    std::ofstream output_file;

    output_file.open(output_file_path);

    output_file << "tn,tp,fn,fp,fpr" << "," << "tpr" << "," << "threshold" << "," << "ks_score" << "," << "accuracy" << "," << "precision" << "," << "recall" << "," << "f1_score" << ",prec_cuts,prec_joins,recall_cuts,recall_joins" <<"\n";

    for(int i = 0; i < roc.size(); i++){

        output_file << roc[i].tn << "," << roc[i].tp << ","  << roc[i].fn << ","  << roc[i].fp << "," << roc[i].fpr << "," << roc[i].tpr << "," << roc[i].threshold << "," << roc[i].ks_score << "," << roc[i].accuracy << "," << roc[i].precision << "," << roc[i].recall << "," << roc[i].f1_score << "," << roc[i].prec_cuts << "," << roc[i].prec_joins << "," << roc[i].recall_cuts << "," << roc[i].recall_joins <<"\n"; 
    }

    output_file.close();
}

void write_execution_time_measurements_to_file(std::string file_name, std::vector<Op_Numbers_And_Runtime>& op_numbers, std::vector<Image_Features_Pair>& all_feature_vectors){

    std::filesystem::path output_file_path = get_data_folder_path();

    output_file_path.append("execution_times");

    output_file_path.append(file_name);

    std::ofstream output_file;

    output_file.open(output_file_path);

    output_file << "cyclic_row_range," << COL_SEARCH_RANGE_IN_CYCLIC_BAB_SEARCH << "\n";

    output_file << "num_features,num_candidates,total_runtime," << 
                    //"BaB greedy runtime,BaB greedy result," << 
                    "BaB cyclic greedy runtime,BaB cyclic greedy results," << 
                    "BaB cyclic greedy transp runtime,BaB cyclic greedy transp results," << 
                    //"distance matching_runtime,distance matching_result," <<
                    "distance matching neighborhood_runtime,distance matching neighborhood_result," << 
                    //"cost_matching_runtime,cost_matching_result,"<<
                    "cost_matching_angle_range_runtime,cost_matching_angle_range_result,"<<
                    "cost_matching_angle_range_with_transp_runtime,cost_matching_angle_range_with_transp_result,"<<
                    //"gurobi_algorithm_runtime,gurobi_algorithm_result <<"  
                    "\n";



    for(int i = 0; i < op_numbers.size();i++){
            output_file << op_numbers[i].num_features << "," << op_numbers[i].num_candiates << "," << op_numbers[i].total_runtime << ",";

            /*
            if(op_numbers[i].runtime[BAB_GREEDY_ALGORITHM] > 0){
                output_file  << op_numbers[i].runtime[BAB_GREEDY_ALGORITHM] << "," << op_numbers[i].normalized_results[BAB_GREEDY_ALGORITHM] << ",";
            }else{
                output_file  << 0 << "," << 0 << ",";
            }
            */
            
            if(op_numbers[i].runtime[BAB_GREEDY_CYCLIC_ALGORITHM] > 0){
                output_file << op_numbers[i].runtime[BAB_GREEDY_CYCLIC_ALGORITHM] << "," << op_numbers[i].normalized_results[BAB_GREEDY_CYCLIC_ALGORITHM] << ",";
            }else{
                output_file  << 0 << "," << 0 << ",";
            }

            if(op_numbers[i].runtime[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION] > 0){
                output_file << op_numbers[i].runtime[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION] << "," << op_numbers[i].normalized_results[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION] << ",";
            }else{
                output_file  << 0 << "," << 0 << ",";
            }
            /*
            if(op_numbers[i].runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM] > 0){
                output_file  << op_numbers[i].runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM] << "," << op_numbers[i].normalized_results[GREEDY_DISTANCE_MATCHING_ALGORITHM] << ",";
            }else{
                output_file  << 0 << "," << 0 << ",";
            }
            */
            
            if(op_numbers[i].runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] > 0){
                output_file  << op_numbers[i].runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] << "," << op_numbers[i].normalized_results[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] << ",";
            }else{
                output_file  << 0 << "," << 0 << ",";
            }
            /*
            if(op_numbers[i].runtime[GREEDY_COST_MATCHING_ALGORITHM] > 0){
                output_file << op_numbers[i].runtime[GREEDY_COST_MATCHING_ALGORITHM] << "," << op_numbers[i].normalized_results[GREEDY_COST_MATCHING_ALGORITHM] << ",";
            }else{
                output_file  << 0 << "," << 0 << ",";
            }
            */
            
            if(op_numbers[i].runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] > 0){
                output_file << op_numbers[i].runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] << "," << op_numbers[i].normalized_results[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] << ",";
            }else{
                output_file  << 0 << "," << 0 << ",";
            }

            if(op_numbers[i].runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION] > 0){
                output_file << op_numbers[i].runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION] << "," << op_numbers[i].normalized_results[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION];
            }else{
                output_file  << 0 << "," << 0 << ",";
            }
            /*
            if(op_numbers[i].runtime[GUROBI_ALGORITHM] > 0){
                output_file << "," << op_numbers[i].runtime[GUROBI_ALGORITHM] << "," << op_numbers[i].normalized_results[GUROBI_ALGORITHM];
            }else{
                output_file  << 0 << "," << 0 << ",";
            }
            */
            output_file << " \n";
            
    }

    output_file << " \n";

    output_file << "num_features," << 
                "total_runtime_avg,total_runtime_min,total_runtime_max," << 
                "cyclic_runtime_avg,cyclic_runtime_min,cyclic_runtime_max," << 
                "cyclic_with_transp_runtime_avg,cyclic_with_transp_runtime_min,cyclic_with_transp_runtime_max," << 
                "distance_runtime_avg,distance_runtime_min,distance_runtime_max," << 
                "cost_runtime_avg,cost_runtime_min,cost_runtime_max,"<<
                "cost_with_transp_runtime_avg,cost_with_transp_runtime_min,cost_with_transp_runtime_max,"<<
                "cyclic_norm_cost_per_runtime_avg,cyclic_norm_cost_per_runtime_min,cyclic_norm_cost_per_runtime_max," << 
                "cyclic_with_transp_norm_cost_per_runtime_avg,cyclic_with_transp_norm_cost_per_runtime_min,cyclic_with_transp_norm_cost_per_runtime_max," << 
                "distance_norm_cost_per_runtime_avg,distance_norm_cost_per_runtime_min,distance_norm_cost_per_runtime_max," << 
                "cost_norm_cost_per_runtime_avg,cost_norm_cost_per_runtime_min,cost_norm_cost_per_runtime_max,"<<
                "cost_with_transp_norm_cost_per_runtime_avg,cost_with_transp_norm_cost_per_runtime_min,cost_with_transp_norm_cost_per_runtime_max,"<<
                "\n";


    for(int i = 0; i < all_feature_vectors.size();i++){
        int current_num_feature_points = all_feature_vectors[i].features->size();
        int current_num_candidates = all_feature_vectors[i].candidates->size();

        int lowest_num_candidates = INT_MAX;
        int highest_num_candidates = 0;

        for(int j = 0; j < all_feature_vectors.size();j++){

            if((all_feature_vectors[j].candidates->size() < lowest_num_candidates) && (current_num_candidates != all_feature_vectors[j].candidates->size())){
                lowest_num_candidates = all_feature_vectors[j].candidates->size();
            }

            if((all_feature_vectors[j].candidates->size() > highest_num_candidates) && (current_num_candidates != all_feature_vectors[j].candidates->size())){
                highest_num_candidates = all_feature_vectors[j].candidates->size();
            }
        }

        std::cout << current_num_feature_points << " : " << lowest_num_candidates << " " << highest_num_candidates << std::endl;

        //runtime
        double total_runtime_of_min = 0;
        double avg_total_runtime = 0;
        double total_runtime_of_max = 0;

        double cyclic_runtime_of_min = 0;
        double avg_cyclic_runtime = 0;
        double cyclic_runtime_of_max = 0;

        double cyclic_with_transp_runtime_of_min = 0;
        double avg_with_transp_cyclic_runtime = 0;
        double cyclic_with_transp_runtime_of_max = 0;

        double distance_runtime_of_min = 0;
        double avg_distance_runtime = 0;
        double distance_runtime_of_max = 0;

        double cost_runtime_of_min = 0;
        double avg_cost_runtime = 0;
        double cost_runtime_of_max = 0;

        double cost_with_transp_runtime_of_min = 0;
        double avg_cost_with_transp_runtime = 0;
        double cost_with_transp_runtime_of_max = 0;

        //normalized costs
        double cyclic_norm_cost_of_min = 0;
        double avg_cyclic_norm_cost = 0;
        double cyclic_norm_cost_of_max = 0;

        double cyclic_with_transp_norm_cost_of_min = 0;
        double avg_with_transp_cyclic_norm_cost = 0;
        double cyclic_with_transp_norm_cost_of_max = 0;

        double distance_norm_cost_of_min = 0;
        double avg_distance_norm_cost = 0;
        double distance_norm_cost_of_max = 0;

        double cost_norm_cost_of_min = 0;
        double avg_cost_norm_cost = 0;
        double cost_norm_cost_of_max = 0;

        double cost_with_transp_norm_cost_of_min = 0;
        double avg_cost_with_transp_norm_cost = 0;
        double cost_with_transp_norm_cost_of_max = 0;

        int num_results = 0;

        for(int k = 0; k < op_numbers.size();k++){
            Op_Numbers_And_Runtime current_op = op_numbers[k];

            if(current_op.num_features != current_num_feature_points){
                continue;
            }

            current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION] += current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM];
            current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] += current_op.runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD];
            current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION] +=  current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE];

            num_results++;

            if(current_op.num_candiates == lowest_num_candidates){
                total_runtime_of_min = current_op.total_runtime;
                cyclic_runtime_of_min = current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM];
                cyclic_with_transp_runtime_of_min = current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION];
                distance_runtime_of_min = current_op.runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD];
                cost_runtime_of_min = current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE];
                cost_with_transp_runtime_of_min = current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION];

                cyclic_norm_cost_of_min = current_op.normalized_results[BAB_GREEDY_CYCLIC_ALGORITHM]/current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM] ;
                cyclic_with_transp_norm_cost_of_min = current_op.normalized_results[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION]/current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION];
                distance_norm_cost_of_min = current_op.normalized_results[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD]/current_op.runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD];
                cost_norm_cost_of_min = current_op.normalized_results[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE]/current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE];
                cost_with_transp_norm_cost_of_min = current_op.normalized_results[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION]/current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION];
            }

            if(current_op.num_candiates == highest_num_candidates){
                total_runtime_of_max = current_op.total_runtime;
                cyclic_runtime_of_max = current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM];
                cyclic_with_transp_runtime_of_max = current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION];
                distance_runtime_of_max = current_op.runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD];
                cost_runtime_of_max = current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE];
                cost_with_transp_runtime_of_max = current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION];

                cyclic_norm_cost_of_max = current_op.normalized_results[BAB_GREEDY_CYCLIC_ALGORITHM]/current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM];
                cyclic_with_transp_norm_cost_of_max = current_op.normalized_results[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION]/current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION];
                distance_norm_cost_of_max = current_op.normalized_results[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD]/current_op.runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD];
                cost_norm_cost_of_max = current_op.normalized_results[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE]/current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE];
                cost_with_transp_norm_cost_of_max = current_op.normalized_results[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION]/current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION];
            }

            avg_total_runtime += current_op.total_runtime;
            avg_cyclic_runtime += current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM];
            avg_with_transp_cyclic_runtime += current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION];
            avg_distance_runtime += current_op.runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD];
            avg_cost_runtime += current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE];
            avg_cost_with_transp_runtime += current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION];

            avg_cyclic_norm_cost += current_op.normalized_results[BAB_GREEDY_CYCLIC_ALGORITHM]/current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM];
            avg_with_transp_cyclic_norm_cost += current_op.normalized_results[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION]/current_op.runtime[BAB_GREEDY_CYCLIC_ALGORITHM_WITH_TRANSPOSITION];
            avg_distance_norm_cost += current_op.normalized_results[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD]/current_op.runtime[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD];
            avg_cost_norm_cost += current_op.normalized_results[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE]/current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE];
            avg_cost_with_transp_norm_cost += current_op.normalized_results[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION]/current_op.runtime[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE_WITH_TRANSPOSITION];

        }

        avg_total_runtime /= (double)num_results;
        avg_cyclic_runtime /= (double)num_results;
        avg_with_transp_cyclic_runtime /= (double)num_results;
        avg_distance_runtime /= (double)num_results;
        avg_cost_runtime /= (double)num_results;
        avg_cost_with_transp_runtime /= (double)num_results;

        avg_cyclic_norm_cost /= (double)num_results;
        avg_with_transp_cyclic_norm_cost /= (double)num_results;
        avg_distance_norm_cost /= (double)num_results;
        avg_cost_norm_cost /= (double)num_results;
        avg_cost_with_transp_norm_cost /= (double)num_results;

        output_file << current_num_feature_points << "," << 
                avg_total_runtime << "," << total_runtime_of_min << "," << total_runtime_of_max  << "," << 
                avg_cyclic_runtime << "," << cyclic_runtime_of_min << "," << cyclic_runtime_of_max << ","  << 
                avg_with_transp_cyclic_runtime << "," << cyclic_with_transp_runtime_of_min << "," << cyclic_with_transp_runtime_of_max << "," << 
                avg_distance_runtime << "," << distance_runtime_of_min << "," << distance_runtime_of_max << "," << 
                avg_cost_runtime << "," << cost_runtime_of_min << "," << cost_runtime_of_max << ","<<
                avg_cost_with_transp_runtime << "," << cost_with_transp_runtime_of_min << "," << cost_with_transp_runtime_of_max << "," <<
                avg_cyclic_norm_cost << "," << cyclic_norm_cost_of_min << "," << cyclic_norm_cost_of_max << ","  << 
                avg_with_transp_cyclic_norm_cost << "," << cyclic_with_transp_norm_cost_of_min << "," << cyclic_with_transp_norm_cost_of_max << "," << 
                avg_distance_norm_cost << "," << distance_norm_cost_of_min << "," << distance_norm_cost_of_max << "," << 
                avg_cost_norm_cost << "," << cost_norm_cost_of_min << "," << cost_norm_cost_of_max << ","<<
                avg_cost_with_transp_norm_cost << "," << cost_with_transp_norm_cost_of_min << "," << cost_with_transp_norm_cost_of_max << "," <<
                "\n";

        
    }

    /*
    output_file << " \n";
    output_file << "as percentage of optimal" << std::endl;
    output_file << "num_features,num_candidates,BaB greedy result,BaB greedy time_p_per," <<
                                                "BaB cyclic greedy results,BaB cyclic greedy time_p_per," <<
                                                "distance matching_result,distance matching_time_p_per," <<
                                                "distance matching neighborhood_result,distance matching neighborhood_time_p_per," <<
                                                "cost_matching_result,cost_matching_time_p_per," <<
                                                "cost_matching_angle_range_result,cost_matching_angle_range_time_p_per"  << "\n";

    std::vector<double> minimal_percentage;
    std::vector<double> average_percentage;
    std::vector<double> maximal_percentage;

    std::vector<double> minimal_time_per_percentage;
    std::vector<double> average_time_per_percentage;
    std::vector<double> maximal_time_per_percentage;

    minimal_percentage.resize(NUM_DIFFERENT_ALGORITHMS);
    average_percentage.resize(NUM_DIFFERENT_ALGORITHMS);
    maximal_percentage.resize(NUM_DIFFERENT_ALGORITHMS);

    minimal_time_per_percentage.resize(NUM_DIFFERENT_ALGORITHMS);
    average_time_per_percentage.resize(NUM_DIFFERENT_ALGORITHMS);
    maximal_time_per_percentage.resize(NUM_DIFFERENT_ALGORITHMS);

    for(int i = 0; i < NUM_DIFFERENT_ALGORITHMS;i++){

        minimal_percentage[i] = 1.0;
        average_percentage[i] = 0.0;
        maximal_percentage[i] = 0.0;

        minimal_time_per_percentage[i] = FLT_MAX;
        average_time_per_percentage[i] = 0.0;
        maximal_time_per_percentage[i] = 0.0;
    }

    for(int i = 0; i < op_numbers.size();i++){

            output_file << op_numbers[i].num_features << "," << op_numbers[i].num_candiates << ","; 

            write_single_percentage_output_and_update_vectors(op_numbers,i,output_file,BAB_GREEDY_ALGORITHM,minimal_percentage,average_percentage,maximal_percentage,minimal_time_per_percentage,average_time_per_percentage,maximal_time_per_percentage);
            write_single_percentage_output_and_update_vectors(op_numbers,i,output_file,BAB_GREEDY_CYCLIC_ALGORITHM,minimal_percentage,average_percentage,maximal_percentage,minimal_time_per_percentage,average_time_per_percentage,maximal_time_per_percentage);
            write_single_percentage_output_and_update_vectors(op_numbers,i,output_file,GREEDY_DISTANCE_MATCHING_ALGORITHM,minimal_percentage,average_percentage,maximal_percentage,minimal_time_per_percentage,average_time_per_percentage,maximal_time_per_percentage);
            write_single_percentage_output_and_update_vectors(op_numbers,i,output_file,GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD,minimal_percentage,average_percentage,maximal_percentage,minimal_time_per_percentage,average_time_per_percentage,maximal_time_per_percentage);
            write_single_percentage_output_and_update_vectors(op_numbers,i,output_file,GREEDY_COST_MATCHING_ALGORITHM,minimal_percentage,average_percentage,maximal_percentage,minimal_time_per_percentage,average_time_per_percentage,maximal_time_per_percentage);
            write_single_percentage_output_and_update_vectors(op_numbers,i,output_file,GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE,minimal_percentage,average_percentage,maximal_percentage,minimal_time_per_percentage,average_time_per_percentage,maximal_time_per_percentage);
            
            output_file << " \n";
            
    }

    for(int i = 0; i < NUM_DIFFERENT_ALGORITHMS;i++){

        average_percentage[i] /= (double)op_numbers.size();
        average_time_per_percentage[i] /= (double)op_numbers.size();
    }

    output_file << " \n";

    output_file << "percentage of optimal" << " \n";
    output_file << "BaB greedy," << minimal_percentage[BAB_GREEDY_ALGORITHM] << "," << maximal_percentage[BAB_GREEDY_ALGORITHM] << "," << average_percentage[BAB_GREEDY_ALGORITHM] <<" \n";
    output_file << "BaB cyclic greedy,"<< minimal_percentage[BAB_GREEDY_CYCLIC_ALGORITHM] << "," << maximal_percentage[BAB_GREEDY_CYCLIC_ALGORITHM] << "," << average_percentage[BAB_GREEDY_CYCLIC_ALGORITHM] <<" \n";
    output_file << "distance matching,"<< minimal_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM] << "," << maximal_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM] << "," << average_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM] <<" \n";
    output_file << "distance matching neighborhood,"<< minimal_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] << "," << maximal_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] << "," << average_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] <<" \n";
    output_file << "cost_matching_result,"<< minimal_percentage[GREEDY_COST_MATCHING_ALGORITHM] << "," << maximal_percentage[GREEDY_COST_MATCHING_ALGORITHM] << "," << average_percentage[GREEDY_COST_MATCHING_ALGORITHM] <<" \n";
    output_file << "cost_matching_angle_range_result,"<< minimal_percentage[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] << "," << maximal_percentage[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] << "," << average_percentage[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] <<" \n";

    output_file <<" \n";

    output_file << "time per percentage" << " \n";
    output_file << "BaB greedy," << minimal_time_per_percentage[BAB_GREEDY_ALGORITHM] << "," << maximal_time_per_percentage[BAB_GREEDY_ALGORITHM] << "," << average_time_per_percentage[BAB_GREEDY_ALGORITHM] <<" \n";
    output_file << "BaB cyclic greedy,"<< minimal_time_per_percentage[BAB_GREEDY_CYCLIC_ALGORITHM] << "," << maximal_time_per_percentage[BAB_GREEDY_CYCLIC_ALGORITHM] << "," << average_time_per_percentage[BAB_GREEDY_CYCLIC_ALGORITHM] <<" \n";
    output_file << "distance matching,"<< minimal_time_per_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM] << "," << maximal_time_per_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM] << "," << average_time_per_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM] <<" \n";
    output_file << "distance matching neighborhood,"<< minimal_time_per_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] << "," << maximal_time_per_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] << "," << average_time_per_percentage[GREEDY_DISTANCE_MATCHING_ALGORITHM_NEIGHBORHOOD] <<" \n";
    output_file << "cost_matching_result,"<< minimal_time_per_percentage[GREEDY_COST_MATCHING_ALGORITHM] << "," << maximal_time_per_percentage[GREEDY_COST_MATCHING_ALGORITHM] << "," << average_time_per_percentage[GREEDY_COST_MATCHING_ALGORITHM] <<" \n";
    output_file << "cost_matching_angle_range_result,"<< minimal_time_per_percentage[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] << "," << maximal_time_per_percentage[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] << "," << average_time_per_percentage[GREEDY_COST_MATCHING_ALGORITHM_PREV_ANGLE] <<" \n";

    output_file.close();
    */

}

void append_current_date_and_time_to_string(std::string& string){
    time_t now = time(0);
    tm *gmtm = localtime(&now);

    string += "_" + std::to_string(gmtm->tm_mday) + "_" + std::to_string(gmtm->tm_mon + 1) + "_" + std::to_string(gmtm->tm_hour) + "_" + std::to_string(gmtm->tm_min);

}

void pixel_pos_from_center_length_and_angle(int& col, int &row, int center_col, int center_row, double length, double angle, double pixels_per_unit_length){

    double radians_angle = angle * (PI / 180.0);

    double cartesian_x = length * sin(radians_angle);
    double cartesian_y = length * cos(radians_angle);

    double pixel_x = cartesian_x * pixels_per_unit_length;
    double pixel_y = cartesian_y * pixels_per_unit_length;

    col = center_col - round(pixel_x);
    row = center_row - round(pixel_y);
}

std::filesystem::path get_data_folder_path(){

    std::filesystem::path cur_path = std::filesystem::current_path(); 

    std::filesystem::path data_folder = cur_path.parent_path();

    data_folder.append("data");

    return data_folder;

}

std::filesystem::path get_image_folder_path(){

    std::filesystem::path cur_path = std::filesystem::current_path(); 

    std::filesystem::path image_folder = cur_path.parent_path().parent_path();

    image_folder.append("images");

    return image_folder;

}

void write_feature_point_selection_parameters_to_file(std::ofstream& output_file, std::string feature_vector_file_name, bool features_were_read_from_file, const Input_Arguments args){

    output_file << "begin_selection_parameters" << std::endl;
    output_file << "image_set," << args.image_set_name << std::endl;
    output_file << "red_channel_feature_acceptance_threshold," << RED_CHANNEL_FEATURE_ACCEPTANCE_THRESHOLD << std::endl;
    output_file << "red_channel_candidate_acceptance_threshold," << RED_CHANNEL_CANDIDATE_ACCEPTANCE_THRESHOLD << std::endl;
    output_file << "red_channel_distance_weight," << RED_CHANNEL_DISTANCE_WEIGHT << std::endl;
    output_file << "red_channel_presience_weight," << RED_CHANNEL_PRESIENCE_WEIGHT << std::endl; 
    output_file << "red_channel_rel_peak_val_weight," << RED_CHANNEL_REL_PEAK_VAL_WEIGHT << std::endl;
    output_file << "red_channel_number_points_weight," << RED_CHANNEL_NUMBER_POINTS_WEIGHT << std::endl;
    output_file << "red_channel_presience_cutoff," << RED_CHANNEL_PRESIENCE_CUTOFF << std::endl;
    output_file << "red_channel_peak_value_cutoff," << RED_CHANNEL_PEAK_VALUE_CUTOFF << std::endl;
    output_file << "red_channel_local_search_dim," << RED_CHANNEL_LOCAL_SEARCH_DIM_AS_PERCENTAGE_OF_IMAGE_SIZE << std::endl;
    output_file << "red_channel_minimum_separation," << RED_CHANNEL_MINIMUM_SEPARATION_AS_PERCENTAGE_OF_IMAGE_SIZE << std::endl;
    output_file << "nucleus_overlap_threshold," << NUCLEUS_OVERLAP_THRESHOLD << std::endl;

    if(features_were_read_from_file){
        output_file << "feature_vector_file," << feature_vector_file_name << std::endl;
    }

    output_file << "end_selection_parameters" << std::endl;
}

void read_selection_parameters_from_file(std::filesystem::path file_path, All_Feature_Point_Selection_Parameters& feature_point_selection_params){

    std::ifstream input_file;

    input_file.open(file_path);

    if (input_file.fail()) {
        std::cout << "File: " << file_path << " to read selection parameters from COULD NOT BE OPENED" << std::endl;
        return;
    }

    const int linebuffer_size = 16777216;
    //char line_buffer[linebuffer_size];

    char* line_buffer = (char*)malloc(sizeof(char) * linebuffer_size);

    char* first_token;
    char delimiter[] = ",:";


    while(!input_file.eof()){
        input_file.getline(line_buffer,linebuffer_size);
        //std::cout << line_buffer << std::endl;

        //std::cout << line_buffer << std::endl;

        first_token = strtok(line_buffer, delimiter);


        if(first_token == NULL){
            continue;
        }


        if(strcmp(first_token,"red_channel_feature_acceptance_threshold") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_feature_acceptance_threshold = atof(first_token);

        }if(strcmp(first_token,"red_channel_candidate_acceptance_threshold") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_candidate_acceptance_threshold = atof(first_token);

        }else if(strcmp(first_token,"red_channel_distance_weight") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_distance_weight = atof(first_token);

        }else if(strcmp(first_token,"red_channel_presience_weight") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_presience_weight = atof(first_token);

        }else if(strcmp(first_token,"red_channel_rel_peak_val_weight") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_rel_peak_val_weight = atof(first_token);

        }else if(strcmp(first_token,"red_channel_number_points_weight") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_number_points_weight = atof(first_token);

        }else if(strcmp(first_token,"red_channel_presience_cutoff") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_presience_cutoff = atof(first_token);

        }else if(strcmp(first_token,"red_channel_peak_value_cutoff") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_peak_value_cutoff = atof(first_token);

        }else if(strcmp(first_token,"red_channel_local_search_dim") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_local_search_dim = atof(first_token);

        }else if(strcmp(first_token,"red_channel_minimum_separation") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.red_channel_minimum_separation = atof(first_token);

        }else if(strcmp(first_token,"nucleus_overlap_threshold") == 0){
            first_token = strtok(NULL, delimiter);
            feature_point_selection_params.nucleus_overlap_threshold = atof(first_token);

        }else if(strcmp(first_token,"feature_vector_file") == 0){
            first_token = strtok(NULL, delimiter);

            if(first_token != NULL){
                std::string features_filename = std::string(first_token);

                if(features_filename == ""){
                    feature_point_selection_params.feature_vector_file = "no_file_given";
                }else{
                    feature_point_selection_params.feature_vector_file = features_filename;
                }
            }
        }else if(strcmp(first_token,"end_selection_parameters") == 0){
            break;
        }
    }

    input_file.close();

    free(line_buffer);
}

void fill_all_feature_selection_parameter_struct_with_currently_defined_vars(All_Feature_Point_Selection_Parameters& parameters, std::string feature_vector_file_name){
    parameters.nucleus_overlap_threshold = NUCLEUS_OVERLAP_THRESHOLD;
    parameters.red_channel_feature_acceptance_threshold = RED_CHANNEL_FEATURE_ACCEPTANCE_THRESHOLD;
    parameters.red_channel_candidate_acceptance_threshold = RED_CHANNEL_CANDIDATE_ACCEPTANCE_THRESHOLD;
    parameters.red_channel_distance_weight = RED_CHANNEL_DISTANCE_WEIGHT;
    parameters.red_channel_local_search_dim = RED_CHANNEL_LOCAL_SEARCH_DIM_AS_PERCENTAGE_OF_IMAGE_SIZE;
    parameters.red_channel_minimum_separation = RED_CHANNEL_MINIMUM_SEPARATION_AS_PERCENTAGE_OF_IMAGE_SIZE;
    parameters.red_channel_number_points_weight = RED_CHANNEL_NUMBER_POINTS_WEIGHT;
    parameters.red_channel_peak_value_cutoff = RED_CHANNEL_PEAK_VALUE_CUTOFF;
    parameters.red_channel_presience_cutoff = RED_CHANNEL_PRESIENCE_CUTOFF;
    parameters.red_channel_presience_weight = RED_CHANNEL_PRESIENCE_WEIGHT;
    parameters.red_channel_rel_peak_val_weight = RED_CHANNEL_REL_PEAK_VAL_WEIGHT;

    if(READ_FEATURE_VECTOR_FROM_FILE){
        parameters.feature_vector_file = feature_vector_file_name;
    }else{
        parameters.feature_vector_file = "no_file_given";
    }
}

bool compare_feature_selection_parameters(All_Feature_Point_Selection_Parameters& params_1,All_Feature_Point_Selection_Parameters& params_2){
    if(!check_if_doubles_are_equal(params_1.nucleus_overlap_threshold,params_2.nucleus_overlap_threshold)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_feature_acceptance_threshold,params_2.red_channel_feature_acceptance_threshold)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_candidate_acceptance_threshold,params_2.red_channel_candidate_acceptance_threshold)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_distance_weight,params_2.red_channel_distance_weight)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_local_search_dim,params_2.red_channel_local_search_dim)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_minimum_separation,params_2.red_channel_minimum_separation)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_number_points_weight,params_2.red_channel_number_points_weight)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_peak_value_cutoff,params_2.red_channel_peak_value_cutoff)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_presience_cutoff,params_2.red_channel_presience_cutoff)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_presience_weight,params_2.red_channel_presience_weight)){return false;}
    if(!check_if_doubles_are_equal(params_1.red_channel_rel_peak_val_weight,params_2.red_channel_rel_peak_val_weight)){return false;}

    return true;
}

void read_input_argument(Input_Arguments& input_arguments, int argc, char** argv, All_Cost_Parameters initial_cost_params){
    input_arguments.segment_organoids = SEGMENT_ORGANOID_IMAGES;

    input_arguments.feature_vector_file_name = FEATURE_VECTORE_FILE_NAME;
    input_arguments.matching_results_file_name = MATCHING_RESULTS_FILE_NAME;

    input_arguments.second_feature_vector_file_name = FEATURE_VECTORE_FILE_NAME_2;
    input_arguments.second_matching_results_file_name = MATCHING_RESULTS_FILE_NAME_2;

    input_arguments.read_feature_vector = READ_FEATURE_VECTOR_FROM_FILE;
    input_arguments.read_matching_results = READ_MATCHING_RESULTS_FROM_FILE;

    input_arguments.use_secondary_image_set = USE_SECONDARY_IMAGE_SET;

    input_arguments.write_feature_vector = WRITE_FEATURE_VECTOR_TO_FILE;
    input_arguments.write_matching_results = WRITE_MATCHING_RESULTS_TO_FILE;

    input_arguments.skip_visualization = SKIP_FINAL_VISUALIZATION;
    input_arguments.skip_matching = SKIP_MATCHING_CALCULATION;

    input_arguments.print_logs = PRINT_TO_LOGS;

    input_arguments.write_execution_time_measurements = WRITE_EXECUTION_TIME_MEASUREMENTS_TO_FILE;

    input_arguments.image_set_name = IMAGE_SET_NAME;
    input_arguments.second_image_set_name = IMAGE_SET_NAME_2;

    input_arguments.perform_dry_run = false;
    input_arguments.print_help_info = false;

    input_arguments.all_model_parameters = initial_cost_params;

    input_arguments.read_model_parameters_from_file = READ_MODEL_PARAMETERS_FROM_FILE;
    input_arguments.model_parameter_file_name = MODEL_PARAMETER_FILE_NAME;

    input_arguments.runtime_limit = hh_mm_ss_time_string_to_double(RUNTIME_LIMIT);

    input_arguments.lm_type = DEFAULT_LEARNING_METRIC;
    input_arguments.sel_strat = DEFAULT_SELECTION_STRATEGY;

    input_arguments.reference_clustering_file_name = DEFAULT_REFERENCE_CLUSTERING;

    input_arguments.read_learning_task = DEFAULT_READ_LEARNING_TASK;
    input_arguments.learning_task_file_name = DEFAULT_LEARNING_TASK_FILE;

    input_arguments.initial_search_step_size = DEFAULT_STEP_WIDTH;

    input_arguments.sim_ann_cooling_rate = DEFAULT_COOL_RATE_SIM_ANN;
    input_arguments.sim_ann_init_temp = DEFAULT_TEMP_SIM_ANN;
    input_arguments.sim_ann_std_dev = DEFAULT_STD_DEV_SIM_ANN;
    input_arguments.sim_ann_restart_thresh = DEFAULT_RESTART_THRESHOLD_SIM_ANN;

    input_arguments.search_order = new std::vector<Search_Dim>;

    time_t now = time(0);
    tm* gmtm = localtime(&now);

    input_arguments.current_time_string = "_" + std::to_string(gmtm->tm_mday) + "_" + std::to_string(gmtm->tm_mon + 1) + "_" + std::to_string(gmtm->tm_hour) + "_" + std::to_string(gmtm->tm_min);

    //std::cout << "argc: " << argc << std::endl;

    for(int i = 1; i < argc; i++){
        char* current_arg = argv[i];
        handle_input_argument(input_arguments,current_arg);
    }

    if(input_arguments.print_help_info){
        print_help_info();
        return;
    }

    check_file_existence_and_compatibility(input_arguments);


    if(input_arguments.read_learning_task){

        read_learning_task(input_arguments.learning_task_file_name,input_arguments);
    }

    if(input_arguments.read_model_parameters_from_file){
        std::filesystem::path data_folder_path = get_data_folder_path();

        data_folder_path.append(input_arguments.model_parameter_file_name);

        All_Cost_Parameters cost_params_from_file;

        bool read_model_params_success = read_model_parameters_from_file(data_folder_path,cost_params_from_file);

        if(read_model_params_success){
            input_arguments.all_model_parameters = cost_params_from_file;
        }
    }
    /*
    write_model_parameters_to_file("model_params_binary.mpb",input_arguments.all_model_parameters);

    All_Cost_Parameters read_binary_params;

    std::filesystem::path binary_data_folder_path = get_data_folder_path();

    binary_data_folder_path.append("model_params_binary.mpb");

    read_model_parameters_from_file(binary_data_folder_path,read_binary_params);
    */

    /*
    if(input_arguments.all_model_parameters.static_cost_offset > 0){
        input_arguments.all_model_parameters.static_cost_offset = -input_arguments.all_model_parameters.static_cost_offset;
    }

    if(input_arguments.all_model_parameters.static_pairwise_cost_offset > 0){
        input_arguments.all_model_parameters.static_pairwise_cost_offset = -input_arguments.all_model_parameters.static_pairwise_cost_offset;
    }
    */

    std::cout << std::endl;

    std::cout << "********** INPUTS **********" << std::endl;

    std::cout << std::endl;

    std::cout << SEGMENT_ORGANOIDS_ARGUMENT_STRING << ": " << input_arguments.segment_organoids << std::endl; 

    std::cout << IMAGE_SET_ARGUMENT_STRING << ": " << input_arguments.image_set_name << std::endl;
    std::cout << SKIP_VIZ_ARGUMENT_STRING << ": " << input_arguments.skip_visualization << std::endl;
    std::cout << SKIP_MATCHING_ARGUMENT_STRING << ": " << input_arguments.skip_matching << std::endl;

    std::cout << READ_FEATURES_ARGUMENT_STRING << ": " << input_arguments.read_feature_vector << std::endl;
    std::cout << FEATURE_VECTOR_FILE_NAME_ARGUMENT_STRING << ": " << input_arguments.feature_vector_file_name << std::endl;
 
    /*
    if(!input_arguments.read_feature_vector && input_arguments.feature_vector_file_name != ""){
        std::cout << "read_feature_vector was set to: " << input_arguments.read_feature_vector << " but a feature_vector_file_name: " << input_arguments.feature_vector_file_name << " was passed" << std::endl; 
    }
    */

    std::cout << READ_MATCHINGS_ARGUMENT_STRING << ": " << input_arguments.read_matching_results << std::endl;
    std::cout << MATCHING_RESULTS_FILE_NAME_ARGUMENT_STRING << ": " << input_arguments.matching_results_file_name << std::endl;

    /*
    if(!input_arguments.read_matching_results && input_arguments.matching_results_file_name != ""){
        std::cout << "read_matching_results was set to: " << input_arguments.read_feature_vector << " but a matching_results_file_name: " << input_arguments.matching_results_file_name << " was passed" << std::endl; 
    }
    */

    std::cout << WRITE_FEATURES_ARGUMENT_STRING << ": " << input_arguments.write_feature_vector << std::endl;
    std::cout << WRITE_MATCHINGS_ARGUMENT_STRING << ": " << input_arguments.write_matching_results << std::endl;

    std::cout << PRINT_LOGS_ARGUMENT_STRING << ": " << input_arguments.print_logs << std::endl;

    std::cout << WRITE_EXECUTION_TIME_MEASURMENTS_ARGUMENT_STRING << ": " << input_arguments.write_execution_time_measurements << std::endl;


    std::cout << COLOR_COST_OFFSET_ARGUMENT_STRING << ": " << input_arguments.all_model_parameters.color_offset << std::endl;
    std::cout << DIST_COST_OFFSET_ARGUMENT_STRING << ": " << input_arguments.all_model_parameters.dist_offset << std::endl;
    std::cout << ANGLE_COST_OFFSET_ARGUMENT_STRING << ": " << input_arguments.all_model_parameters.angle_offset << std::endl;
    std::cout << COLOR_TO_DISTANCE_WEIGHT_ARGUMENT_STRING << ": " << input_arguments.all_model_parameters.color_to_dist_weight << std::endl;
    std::cout << UNARY_TO_QUADR_WEIGHT_ARGUMENT_STRING << ": " << input_arguments.all_model_parameters.unary_to_to_quadr_weight << std::endl;

    std::cout << READ_MODEL_PARAMETER_FILE_ARGUMENT_STRING << ": " << input_arguments.read_model_parameters_from_file << std::endl;
    std::cout << MODEL_PARAMETER_FILE_ARGUMENT_STRING << ": " << input_arguments.model_parameter_file_name << std::endl;
    std::cout << RUNTIME_LIMIT_ARGUMENT_STRING << ": " << input_arguments.runtime_limit << std::endl;

    std::cout << LEARNING_METRIC_ARGUMENT_STRING << ": " <<    get_string_from_metric_type(input_arguments.lm_type) << std::endl;
    std::cout << LEARNING_SEARCH_STRATEGY_ARGUMENT_STRING << ": " << get_string_from_search_strategy(input_arguments.sel_strat) << std::endl;
    std::cout << SEARCH_STEP_SIZE_ARGUMENT_STRING << ": " << input_arguments.initial_search_step_size << std::endl;

    std::cout << LEARNING_TASK_FILE_ARGUMENT_STRING << ": " << input_arguments.learning_task_file_name << std::endl;
    std::cout << std::endl;

    std::cout << "********** INPUTS **********" << std::endl;

    std::cout << std::endl;

    if(input_arguments.print_logs){
        std::filesystem::path path_to_logs = get_data_folder_path();
        path_to_logs.append("logs");

        std::filesystem::path output_logs_path = path_to_logs;
        output_logs_path.append("output_logs" + input_arguments.current_time_string + ".txt");

        std::filesystem::path error_logs_path = path_to_logs;
        error_logs_path.append("error_logs" + input_arguments.current_time_string + ".txt");

        const char* out_log_c_path = (const char*)output_logs_path.c_str();
        const char* err_log_c_path = (const char*)error_logs_path.c_str();

        freopen(out_log_c_path,"w",stdout);
        freopen(err_log_c_path,"w",stderr);
    }

}

void handle_input_argument(Input_Arguments& input_args, char* current_arg){

    char* first_token;
    char* second_token;
    char delimiter[] = "=";

    first_token = strtok(current_arg, delimiter);
    second_token = strtok(NULL,delimiter);

    if(strcmp(first_token,DRY_RUN_ARGUMENT_STRING) == 0){
        input_args.perform_dry_run = true;
        return;
    }

    if(strcmp(first_token,HELP_ARGUMENT_STRING) == 0){
        input_args.print_help_info = true;
        return;
    }

    if(strcmp(first_token,IMAGE_SET_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.image_set_name = std::string(second_token);
            //std::cout << input_args.image_set_name << std::endl;
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,FEATURE_VECTOR_FILE_NAME_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.feature_vector_file_name = std::string(second_token);
            //std::cout << input_args.feature_vector_file_name << std::endl;
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,MATCHING_RESULTS_FILE_NAME_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.matching_results_file_name = std::string(second_token);
            //std::cout << input_args.matching_results_file_name << std::endl;
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,SKIP_VIZ_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.skip_visualization);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,SEGMENT_ORGANOIDS_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.segment_organoids);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,SKIP_MATCHING_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.skip_matching);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    
    if(strcmp(first_token,WRITE_EXECUTION_TIME_MEASURMENTS_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.write_execution_time_measurements);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,PRINT_LOGS_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.print_logs);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,READ_FEATURES_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.read_feature_vector);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }


    if(strcmp(first_token,READ_MATCHINGS_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.read_matching_results);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,WRITE_FEATURES_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.write_feature_vector);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,WRITE_MATCHINGS_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.write_matching_results);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,COLOR_COST_OFFSET_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.all_model_parameters.color_offset = atof(second_token);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,DIST_COST_OFFSET_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.all_model_parameters.dist_offset = atof(second_token);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,ANGLE_COST_OFFSET_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.all_model_parameters.angle_offset = atof(second_token);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,COLOR_TO_DISTANCE_WEIGHT_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.all_model_parameters.color_to_dist_weight = atof(second_token);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,UNARY_TO_QUADR_WEIGHT_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.all_model_parameters.unary_to_to_quadr_weight = atof(second_token);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,READ_MODEL_PARAMETER_FILE_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.read_model_parameters_from_file);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,MODEL_PARAMETER_FILE_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.model_parameter_file_name = std::string(second_token);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,RUNTIME_LIMIT_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.runtime_limit = hh_mm_ss_time_string_to_double(std::string(second_token));
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,LEARNING_METRIC_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.lm_type = get_metric_type_from_string(std::string(second_token));
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,LEARNING_SEARCH_STRATEGY_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.sel_strat = get_search_strategy_from_string(std::string(second_token));
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,REFERENCE_CLUSTERING_FILE_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.reference_clustering_file_name = std::string(second_token);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,READ_LEARNING_TASK_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            parse_true_false_arg_string(first_token,second_token,input_args.read_learning_task);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    if(strcmp(first_token,LEARNING_TASK_FILE_ARGUMENT_STRING) == 0){
        if(second_token != NULL){
            input_args.learning_task_file_name = std::string(second_token);
        }else{
            print_missing_argument_value_error(first_token);
        }
        return;
    }

    print_unrecognised_argument_error(first_token);
}

void print_missing_argument_value_error(char* arg_name){
    std::cout << "option " << arg_name << " was passed but no value was provided" << std::endl;

}

void print_unrecognised_argument_error(char* arg_name){
    std::cout << "unrecognised argument: " << arg_name << std::endl;
    std::cout << "pass --help to see all available arguments" << std::endl;

}

void print_unrecognised_argument_value_error(char* arg_name, char* arg_value){
    std::cout << "unrecognised value: " << arg_value << " for argument: " << arg_name << std::endl;

}

void parse_true_false_arg_string(char* arg_name, char* arg_value_string, bool& arg){

    if(strcmp(arg_value_string,"true") == 0){
        arg = true;
        return;
    }

    if(strcmp(arg_value_string,"false") == 0){
        arg = false;
        return;
    }

    print_unrecognised_argument_value_error(arg_name, arg_value_string);

}

void print_help_info(){

    std::cout << "********** HELP **********" << std::endl;

    std::cout << DRY_RUN_ARGUMENT_STRING << " " << std::endl;

    std::cout << SEGMENT_ORGANOIDS_ARGUMENT_STRING << " " << std::endl;

    std::cout << SKIP_VIZ_ARGUMENT_STRING << " " << std::endl;

    std::cout << SKIP_MATCHING_ARGUMENT_STRING << " " << std::endl;

    std::cout << IMAGE_SET_ARGUMENT_STRING << " " << std::endl;

    std::cout << READ_FEATURES_ARGUMENT_STRING << " " << std::endl;

    std::cout << FEATURE_VECTOR_FILE_NAME_ARGUMENT_STRING << " " << std::endl;

    std::cout << READ_MATCHINGS_ARGUMENT_STRING << " " << std::endl;

    std::cout << MATCHING_RESULTS_FILE_NAME_ARGUMENT_STRING << " " << std::endl;

    std::cout << WRITE_FEATURES_ARGUMENT_STRING << " " << std::endl;

    std::cout << WRITE_MATCHINGS_ARGUMENT_STRING << " " << std::endl;

    std::cout << PRINT_LOGS_ARGUMENT_STRING << " " << std::endl;

    std::cout << WRITE_EXECUTION_TIME_MEASURMENTS_ARGUMENT_STRING << " " << std::endl;

    std::cout << COLOR_COST_OFFSET_ARGUMENT_STRING << " " << std::endl;

    std::cout << DIST_COST_OFFSET_ARGUMENT_STRING << " " << std::endl;

    std::cout << ANGLE_COST_OFFSET_ARGUMENT_STRING << " " << std::endl;

    std::cout << COLOR_TO_DISTANCE_WEIGHT_ARGUMENT_STRING << " " << std::endl;

    std::cout << UNARY_TO_QUADR_WEIGHT_ARGUMENT_STRING << " " << std::endl;

    std::cout << READ_MODEL_PARAMETER_FILE_ARGUMENT_STRING << " " << std::endl;

    std::cout << MODEL_PARAMETER_FILE_ARGUMENT_STRING << " " << std::endl; 

    std::cout << RUNTIME_LIMIT_ARGUMENT_STRING << " " << std::endl; 

    std::cout << LEARNING_METRIC_ARGUMENT_STRING << " valid options: " << LM_BCE_STRING << ", " << LM_WBCE_STRING << ", " << LM_FN_FP_NUM_STRING << ", " << LM_ACC_STRING << ", " << LM_F1_SCORE_STRING << ", " << LM_TPR_TNR_AVG_STRING << ", " << LM_MCC_STRING << std::endl;

    std::cout << LEARNING_SEARCH_STRATEGY_ARGUMENT_STRING << " valid options: " <<  SEL_STRAT_EXHAUSTIVE_ADJ_STRING << ", " << SEL_STRAT_LINE_SEARCH_STRING << ", " << SEL_STRAT_NO_SEARCH_STRING << ", " << SEL_STRAT_SIM_ANN_STRING << std::endl;

    std::cout << "********** HELP **********" << std::endl;
}

bool check_file_existence_and_compatibility(Input_Arguments& args){

    std::filesystem::path matching_results_file_path = get_data_folder_path();
    matching_results_file_path.append("matching_results");
    matching_results_file_path.append(args.matching_results_file_name);

    std::filesystem::path feature_vectors_file_path = get_data_folder_path();
    feature_vectors_file_path.append("feature_vectors");
    feature_vectors_file_path.append(args.feature_vector_file_name);


    std::ifstream feature_vector_file(feature_vectors_file_path);

    bool feature_vector_file_exists = feature_vector_file.good();
    feature_vector_file.close();

    if(!feature_vector_file_exists && args.read_feature_vector){
        std::cout << "WARNING: the feature vector file does not exist or could not be opened: " << args.feature_vector_file_name << std::endl;
        std::cout << "Searched for the feature vectors at: " << feature_vectors_file_path << std::endl;
    }

    std::ifstream matching_result_file(matching_results_file_path);

    bool matching_results_file_exists = matching_result_file.good();
    matching_result_file.close();

    if(!matching_results_file_exists && args.read_matching_results){
        std::cout << "WARNING: the matching results file does not exist or could not be opened: " << args.matching_results_file_name << std::endl;
        std::cout << "Searched for the matching results at: " << matching_results_file_path << std::endl;
    }


    std::vector<int> image_numbers_in_image_set;
    std::vector<int> image_numbers_in_features;
    std::vector<int> image_numbers_in_matching_results;

    bool image_set_folder_exists = read_image_numbers_from_image_set(args,image_numbers_in_image_set);

    
    All_Feature_Point_Selection_Parameters parameters_in_feature_vector_file{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"no_file_given"};
    All_Feature_Point_Selection_Parameters parameters_in_matching_result_file{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"no_file_given"};
    All_Feature_Point_Selection_Parameters parameters_from_defines;
    fill_all_feature_selection_parameter_struct_with_currently_defined_vars(parameters_from_defines,args.feature_vector_file_name);

    bool parameters_match = true;
    bool image_numbers_match = true;

    if(args.read_feature_vector && feature_vector_file_exists){
        read_selection_parameters_from_file(feature_vectors_file_path,parameters_in_feature_vector_file);
        read_image_numbers_from_feature_vector_file(args,image_numbers_in_features);
    }

    if(args.read_matching_results && matching_results_file_exists){
        read_selection_parameters_from_file(matching_results_file_path,parameters_in_matching_result_file);
        read_image_numbers_from_matching_results_file(args,image_numbers_in_matching_results);
    }

    if(parameters_in_matching_result_file.feature_vector_file != args.feature_vector_file_name && args.read_matching_results && args.read_feature_vector){

            std::cout << "WARNING: the feature_vector file in the matching results: " << parameters_in_matching_result_file.feature_vector_file << " does not match the current feature vector file: " << args.feature_vector_file_name << std::endl;

            std::filesystem::path feature_vectors_in_matching_file_path = feature_vectors_file_path.parent_path();
            feature_vectors_in_matching_file_path.append(parameters_in_matching_result_file.feature_vector_file);

            bool feature_vectors_in_matching_file_exists = check_if_file_exists(feature_vectors_in_matching_file_path);

            if(feature_vectors_in_matching_file_exists){
                std::cout << "The feature vector file from the matching results does still exists. Will read from this file instead: " << parameters_in_matching_result_file.feature_vector_file << std::endl;
                args.feature_vector_file_name = parameters_in_matching_result_file.feature_vector_file;
            }
        }

    if(!args.skip_visualization && args.read_feature_vector && feature_vector_file_exists){
        if(image_set_folder_exists){
            image_numbers_match = compare_image_number_vectors(image_numbers_in_image_set,image_numbers_in_features);
        }

        if(!image_numbers_match){
            std::cout << "WARNING: The image numbers in the feature vectors file do NOT MATCH with the numbers from the image set." << std::endl;
            std::cout << "Image numbers in feature vectors: ";
            print_int_vector(image_numbers_in_features);
            std::cout << "Image numbers in image set: ";
            print_int_vector(image_numbers_in_image_set);
        }
    }

    if(!args.read_feature_vector && args.read_matching_results && matching_results_file_exists){
        parameters_match = compare_feature_selection_parameters(parameters_from_defines,parameters_in_matching_result_file);

        if(image_set_folder_exists){
            image_numbers_match = compare_image_number_vectors(image_numbers_in_image_set,image_numbers_in_matching_results);
        }

        if(!image_numbers_match){
            std::cout << "WARNING: The image numbers in the matching results file do NOT MATCH with the numbers from the image set." << std::endl;
            std::cout << "Image numbers in matching results: ";
            print_int_vector(image_numbers_in_matching_results);
            std::cout << "Image numbers in image set: ";
            print_int_vector(image_numbers_in_image_set);
        }

        if(!parameters_match){
            std::cout << "WARNING: Parameters did not match in the matching results file and the currently defined variables." << std::endl;
        }
    }

    if(args.read_feature_vector && args.read_matching_results && feature_vector_file_exists && matching_results_file_exists){
        parameters_match = compare_feature_selection_parameters(parameters_in_feature_vector_file,parameters_in_matching_result_file);

        if(!parameters_match){
            std::cout << "WARNING: Parameters DID NOT match in the matching results file and the feature vector file." << std::endl;
        }


        bool features_and_matching_match = compare_image_number_vectors(image_numbers_in_features,image_numbers_in_matching_results);

        bool features_and_set_match = false;
        bool matching_and_set_match = false;

        if(image_set_folder_exists){
            features_and_set_match = compare_image_number_vectors(image_numbers_in_features,image_numbers_in_image_set);
            matching_and_set_match = compare_image_number_vectors(image_numbers_in_image_set,image_numbers_in_matching_results);
        }

        image_numbers_match = features_and_matching_match && features_and_set_match && matching_and_set_match;
        
        if(!image_numbers_match){
            std::cout << "WARNING: image numbers do NOT MATCH!" << std::endl;
            std::cout << "Image numbers in image set ";
            print_int_vector(image_numbers_in_image_set);
            std::cout << "Image numbers in matching results: ";
            print_int_vector(image_numbers_in_matching_results);
            std::cout << "Image numbers in feature vectors: ";
            print_int_vector(image_numbers_in_features);
        }

    }
    
    return parameters_match && matching_results_file_exists && feature_vector_file_exists && image_numbers_match;

}

bool read_image_numbers_from_image_set(Input_Arguments& args, std::vector<int>& image_numbers){

    image_numbers.clear();

    std::filesystem::path path_to_image_set = get_image_folder_path();

    path_to_image_set.append("single_organoids");
    path_to_image_set.append("pairs");
    path_to_image_set.append(args.image_set_name);

    bool folder_exists = std::filesystem::exists(path_to_image_set);

    if(!folder_exists){

        std::cout << "WARNING the image set: " << path_to_image_set <<  " does NOT EXIST!" << std::endl;
    
        return false;
    }

    //std::cout<< path_to_image_set << std::endl;

    for (const auto & entry : std::filesystem::directory_iterator(path_to_image_set)){
        if(!entry.is_directory()){
            
            std::string filename = entry.path().filename().string();

            if(check_if_file_is_mask_image(filename)){
                continue;
            }

            image_numbers.push_back(get_image_number_from_file_name(filename));

        }
    }

    return true;
}

bool read_image_numbers_from_feature_vector_file(std::filesystem::path file_path, std::vector<int>& image_numbers){
    std::ifstream input_file;
    input_file.open(file_path);

    if (input_file.fail()) {
        std::cout << "File: " << file_path << " to read feature vector from COULD NOT BE OPENED" << std::endl;
        return false;
    }

    const int linebuffer_size = 1024;
    char line_buffer[linebuffer_size];

    char* first_token;
    char delimiter[] = ",:";


    while(!input_file.eof()){
        input_file.getline(line_buffer,linebuffer_size);
        first_token = strtok(line_buffer, delimiter);

        if(first_token == NULL){
            continue;
        }

        if(strcmp(first_token,"features_begin") == 0){
            first_token = strtok(NULL, delimiter);

            int found_image_number = atoi(first_token);

            bool number_already_found = false;
            for(int i = 0; i < image_numbers.size();i++){
                if(found_image_number == image_numbers[i]){
                    number_already_found = true;
                    break;
                }
            }

            if(!number_already_found){
                image_numbers.push_back(found_image_number);
            }
        }
    }

    return true;
}

bool read_image_numbers_from_feature_vector_file(Input_Arguments& args, std::vector<int>& image_numbers){

    std::filesystem::path feature_vectors_file_path = get_data_folder_path();
    feature_vectors_file_path.append("feature_vectors");
    feature_vectors_file_path.append(args.feature_vector_file_name);

    return read_image_numbers_from_feature_vector_file(feature_vectors_file_path,image_numbers);
}

bool read_image_numbers_from_matching_results_file(Input_Arguments& args, std::vector<int>& image_numbers){

    std::ifstream input_file;

    std::filesystem::path matching_results_file_path = get_data_folder_path();
    matching_results_file_path.append("matching_results");
    matching_results_file_path.append(args.matching_results_file_name);

    input_file.open(matching_results_file_path);

    if (input_file.fail()) {
        std::cout << "File: " << matching_results_file_path << " to read matching results from COULD NOT BE OPENED" << std::endl;
        return false;
    }

    const int linebuffer_size = 16777216;
    //char line_buffer[linebuffer_size];
    char* line_buffer = (char*)malloc(sizeof(char) * linebuffer_size);

    char* first_token;
    char delimiter[] = ",:";

    Matching_Visualization_Type type;
    bool is_line_with_image_numbers = false;

    bool all_image_numbers_vector_filled = false;


    while(!input_file.eof() && !all_image_numbers_vector_filled){
        input_file.getline(line_buffer,linebuffer_size);

        first_token = strtok(line_buffer, delimiter);

        while (true) {

            if(first_token == NULL){
                break;
            }

            if(is_line_with_image_numbers && !all_image_numbers_vector_filled){

                image_numbers.push_back(atoi(first_token));

            }else{
                if(strcmp(first_token,"quadratic_matching") == 0){
                    is_line_with_image_numbers = true;
                }else if(strcmp(first_token,"linear_by_feature") == 0){
                    is_line_with_image_numbers = true;
                }else if(strcmp(first_token,"linear_by_candidate") == 0){
                    is_line_with_image_numbers = true;
                }else if(strcmp(first_token,"optimal_quadratic_matching") == 0){
                    is_line_with_image_numbers = true;
                }
            }

            first_token = strtok(NULL, delimiter);
        }

        if(is_line_with_image_numbers){
            all_image_numbers_vector_filled = true;
        }

    }
    
    free(line_buffer);
    return true;
}


bool compare_image_number_vectors(std::vector<int>& numbers_a, std::vector<int>& numbers_b){

    if(numbers_a.size() != numbers_b.size()){
        std::cout << "image number vectors have unequal sizes! " << numbers_a.size() << " and " << numbers_b.size() << std::endl;
        return false;
    }

    for(int i = 0; i < numbers_a.size();i++){
        bool found_num_from_a_in_b = false;

        int num_in_a = numbers_a[i];

        for(int j = 0; j < numbers_b.size();j++){
            if(num_in_a == numbers_b[j]){
                found_num_from_a_in_b = true;
            }
        }
        if(!found_num_from_a_in_b){

            std::cout << "Could not find: " << num_in_a << " in both vectors! Stopped comparing" << std::endl;
            return false;
        }

    }

    return true;

}

bool check_if_file_exists(std::filesystem::path file_path){

    std::ifstream input_file;

    std::string filename = file_path.filename().string();

    input_file.open(file_path);

    if (input_file.fail()) {
        input_file.close();
        return false;
    }

    input_file.close();
    return true;
}

int get_index_of_element_in_ordered_vector(std::vector<int>& vec, int value){

    int found_index = -1;
    bool index_was_found = false;

    int half_size_of_remaining = vec.size();
    int offset = 0;

    while(half_size_of_remaining >= 1){
        half_size_of_remaining >>= 1;

        int middle_index = offset + half_size_of_remaining;

        int value_at_middle = vec[middle_index];

        if(value_at_middle == value){
            found_index = middle_index;
            index_was_found = true;
            break;
        }else if(value > value_at_middle){
            offset = middle_index;
        }
    }

    
    return found_index;
}

int get_closest_index_of_element_in_ordered_vector(std::vector<int>& vec, int value){

    int previous_value = INT_MAX;

    for(int i = 0; i < vec.size();i++){
        int current_value = vec[i];

        if(current_value == value){
            return i;
        }

        if(previous_value < value && current_value > value){
            return i-1;
        }

        previous_value = current_value;
    }

    return -1;
}

void merge_images_from_multiple_folders_into_single_folder(std::filesystem::path base_path){

    base_path.append("single_organoids");

    std::filesystem::path output_dataset_path = base_path;
    output_dataset_path.append("data_sets");
    output_dataset_path.append("dataset");

    int consecutive_organoid_image_number = 0;

    std::vector<int> image_number_mappings;

    image_number_mappings.clear();

    const auto copy_options = std::filesystem::copy_options::overwrite_existing;

    for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
        //std::cout << entry << std::endl;

        if (entry.is_directory()) {

            std::string directory_name = entry.path().filename().string();

            if(directory_name == "data_sets"){
                //std::cout << "skipped data sets folder" << std::endl;
                continue;
            }

            std::filesystem::path single_image_folder = entry;
            single_image_folder.append("good_images");

            for (const auto& image_entry : std::filesystem::directory_iterator(single_image_folder)) {
                //std::cout << image_entry << std::endl;

                std::string image_filename = image_entry.path().filename().string();

                bool was_mask_image = false;

                if(check_if_file_is_mask_image(image_filename)){
                    remove_mask_suffix(image_filename);
                    was_mask_image = true;
                }

                int image_number = get_image_number_from_file_name(image_filename);

                int mapped_image_number = -1;

                for(int i = 0; i < image_number_mappings.size();i++){
                    if(image_number_mappings[i] == image_number){
                        mapped_image_number = i;
                        break;
                    }
                }

                if(mapped_image_number == -1){
                    mapped_image_number = image_number_mappings.size();
                    image_number_mappings.push_back(image_number);
                }

                mapped_image_number += consecutive_organoid_image_number;

                std::string output_filename = "single_organoid_" + std::to_string(mapped_image_number);

                if(was_mask_image){
                    output_filename += "_mask";
                }

                output_filename += ".tif";

                std::filesystem::path output_path = output_dataset_path;
                output_path.append(output_filename);

                if(std::filesystem::is_directory(output_path.parent_path())){
                    std::filesystem::copy(image_entry,output_path,copy_options);
                }

            }

            consecutive_organoid_image_number += image_number_mappings.size();

            image_number_mappings.clear();
        }
    }
}

void check_if_image_and_mask_is_present_for_all_images(std::filesystem::path path){

    std::vector<Image_Number_and_Present_Images> image_nums_and_present_images;

    const auto copy_options = std::filesystem::copy_options::overwrite_existing;

    for (const auto& image_entry : std::filesystem::directory_iterator(path)) {
        //std::cout << entry << std::endl;

        std::string image_filename = image_entry.path().filename().string();

        bool was_mask_image = false;

        if(check_if_file_is_mask_image(image_filename)){
            remove_mask_suffix(image_filename);
            was_mask_image = true;
        }

        int image_number = get_image_number_from_file_name(image_filename);

        int found_existing_image_num = false;

        for(int i = 0; i < image_nums_and_present_images.size();i++){
            if(image_nums_and_present_images[i].image_number == image_number){
                if(was_mask_image){
                    image_nums_and_present_images[i].mask_is_present = true;
                }else{
                    image_nums_and_present_images[i].image_is_present = true;
                }
                found_existing_image_num = true;
                break;
            }
        }

        if(!found_existing_image_num){
            Image_Number_and_Present_Images new_imgapi;
            new_imgapi.image_number = image_number;
            
            if(was_mask_image){
                new_imgapi.mask_is_present = true;
                new_imgapi.image_is_present = false;
            }else{
                new_imgapi.mask_is_present = false;
                new_imgapi.image_is_present = true;
            }

            image_nums_and_present_images.push_back(new_imgapi);
        }
    }

    for(int i = 0; i < image_nums_and_present_images.size();i++){
        if(!(image_nums_and_present_images[i].mask_is_present && image_nums_and_present_images[i].image_is_present)){
            std::cout << image_nums_and_present_images[i].image_number << " " << image_nums_and_present_images[i].image_is_present << " " << image_nums_and_present_images[i].mask_is_present << std::endl;
        }
    }
}

int calculate_binomial_coefficient(int n, int k){

    if(k == 0 || k == n){
        return 1;
    }

    return calculate_binomial_coefficient(n-1,k-1) + calculate_binomial_coefficient(n-1,k);

}

void read_clustering_from_csv_file(std::string filename, std::vector<Cluster>& clustering){

    if(filename == ".gitkeep"){
        return;
    }

    std::ifstream input_file;

    std::filesystem::path file_path = get_data_folder_path();

    file_path.append("clustering_results");

    file_path.append(filename);

    input_file.open(file_path);

    if (input_file.fail()) {
        std::cout << "File: " << file_path << " to read clustering from COULD NOT BE OPENED" << std::endl;
        return;
    }

    const int linebuffer_size = 16777216;
    //char line_buffer[linebuffer_size];

    char* line_buffer = (char*)malloc(sizeof(char) * linebuffer_size);

    std::vector<int> all_image_numbers;

    char* first_token;
    char delimiter[] = ",:";

    int cluster_id = 0; 

    while(!input_file.eof()){
        input_file.getline(line_buffer,linebuffer_size);

        first_token = strtok(line_buffer, delimiter);

        if(first_token == NULL){
            break;
        }

        if(strcmp(first_token,"matching_results_file") == 0){
            continue;
        }

        if(strcmp(first_token,"clustering_threshold") == 0){
            continue;
        }



        Cluster new_cluster;

        new_cluster.members = new std::vector<int>;
        new_cluster.cost_of_cluster = 0.0;


        while(first_token != NULL){
            int new_cluster_member = atoi(first_token);

            new_cluster.members->push_back(new_cluster_member);

            first_token = strtok(NULL, delimiter);
        }

        if(new_cluster.members->size() > 0){
            clustering.push_back(new_cluster);
        }
    }


    free(line_buffer);
    std::cout << "finished reading reference clustering from file: " << filename << std::endl;
}


void write_clusters_to_csv_file(std::vector<Cluster>* clustering,std::string corresponding_matching_results_file, float used_clustering_threshold){

    if(clustering->size() == 0){
        std::cout << "size of clustering was 0 in write_clusters_to_csv_file." << std::endl;
        return;
    }

    std::filesystem::path output_folder_path = get_data_folder_path();

    output_folder_path.append("clustering_results");

    time_t now = time(0);
    tm* gmtm = localtime(&now);

    std::string date_string = "_" + std::to_string(gmtm->tm_mday) + "_" + std::to_string(gmtm->tm_mon + 1) + "_" + std::to_string(gmtm->tm_hour) + "_" + std::to_string(gmtm->tm_min);

    std::string output_filename = "clustering" + date_string + ".csv";

    output_folder_path.append(output_filename);

    std::ofstream output_file;

    output_file.open(output_folder_path);

    for(int i = 0; i < clustering->size();i++){
        Cluster current_cluster = (*(clustering))[i];

        if(current_cluster.members->size() == 0){
            continue;
        }

        for(int j = 0; j < current_cluster.members->size();j++){

            int member_id = (*(current_cluster.members))[j];

            output_file << member_id;

            if(j != current_cluster.members->size()-1){
                output_file << ",";
            }else{
                output_file << "\n";
            }

        }


    }

    output_file << "matching_results_file:" << corresponding_matching_results_file << "\n"; 
    output_file << "clustering_threshold:" << used_clustering_threshold << "\n";

    output_file.close();

    std::cout << "Clusters have been writen to file: " << output_filename << std::endl;

}

void print_cost_parameters(const All_Cost_Parameters& cost_params){

    std::cout << "Cost Params:" << std::endl;
    std::cout << COLOR_COST_OFFSET_STRING << ": " << cost_params.color_offset << std::endl;
    std::cout << DIST_COST_OFFSET_STRING << ": " << cost_params.dist_offset << std::endl;
    std::cout << ANGLE_COST_OFFSET_STRING << ": " << cost_params.angle_offset << std::endl;
    std::cout << COLOR_TO_DISTANCE_WEIGHT_STRING << ": " << cost_params.color_to_dist_weight << std::endl;
    std::cout << UNARY_TO_QUADR_WEIGHT_STRING << ": " << cost_params.unary_to_to_quadr_weight << std::endl;
    std::cout << std::endl;

}

void read_image_numbers_from_image_set_folder(std::filesystem::path image_set_path, std::vector<int>& image_numbers){

    //std::cout << image_set_path.string() << std::endl;

    for (const auto& entry : std::filesystem::directory_iterator(image_set_path)) {
        //std::cout << entry << std::endl;
        if (!entry.is_directory()) {

            std::string current_filename = entry.path().filename().string();

            if(!check_if_file_is_mask_image(current_filename)){

                int new_image_number = get_image_number_from_file_name(current_filename);
                //std::cout << new_image_number << " ";
                image_numbers.push_back(new_image_number);    
            }

        }
    }

    //std::cout << std::endl;

}

bool check_if_image_number_is_contained_in_vector(const std::vector<int>& image_number_vector, int image_number){
    for(int i = 0; i < image_number_vector.size();i++){
        if(image_number_vector[i] == image_number){
            return true;
        }
    }

    return false;
}

bool check_if_image_number_is_contained_in_clustering(const std::vector<Cluster>& clustering, int image_number){

    for(int i = 0; i < clustering.size();i++){
        if(check_if_image_number_is_contained_in_vector(*(clustering[i].members),image_number)){
            return true;
        }
    }

    return false;
}


bool check_if_all_image_numbers_are_contained_in_clustering(const std::vector<Cluster>& clustering, const std::vector<Image_Features_Pair>& all_feature_vectors){

    for(int i = 0; i < all_feature_vectors.size();i++){
        if(!check_if_image_number_is_contained_in_clustering(clustering,all_feature_vectors[i].image_number)){
            return false;
        }
    }

    return true;
}

Matching_Result get_matching_result_by_image_ids(std::vector<Matching_Result>& all_matching_results, int id_1, int id_2, bool exact_match_only){

    for(int i = 0; i < all_matching_results.size();i++){
        Matching_Result current_mr = all_matching_results[i];

        if(current_mr.id_1 == id_1 && current_mr.id_2 == id_2){
            return current_mr;
        }

        if(current_mr.id_2 == id_1 && current_mr.id_1 == id_2 && ! exact_match_only){
            return current_mr;
        }
    }

    Matching_Result empty_mr;
    empty_mr.additional_viz_data_id1_to_id2 = nullptr;
    empty_mr.additional_viz_data_id2_to_id1 = nullptr;
    empty_mr.assignment = nullptr;

    empty_mr.id_1 = -1;
    empty_mr.id_2 = -1;
    empty_mr.linear_cost_per_candidate = 0.0;
    empty_mr.linear_cost_per_feature = 0.0;
    empty_mr.rel_quadr_cost = 0.0;
    empty_mr.rel_quadr_cost_optimal = 0.0;
    empty_mr.set_id_1 = -1;
    empty_mr.set_id_2 = -1;

    return empty_mr;
}

float sum_matching_cost_between_single_element_and_cluster(Cluster& cluster, std::vector<Matching_Result>& all_matching_results, int img_num_of_single_elem, bool normalize_by_cluster_size){

    float total_matching_cost;

    for(int i = 0; i < all_matching_results.size();i++){

        Matching_Result current_mr = all_matching_results[i];

        for(int j = 0; j < cluster.members->size();j++){
            int current_cluster_mem_img_num = (*(cluster.members))[j];

            if((current_mr.id_1 == img_num_of_single_elem && current_mr.id_2 == current_cluster_mem_img_num) || (current_mr.id_2 == img_num_of_single_elem && current_mr.id_1 == current_cluster_mem_img_num) ){
                //std::cout << current_mr.id_1 << " " << current_mr.id_2 << " " << current_mr.rel_quadr_cost << std::endl;
                total_matching_cost += current_mr.rel_quadr_cost;
                break;
            }
        }

    }

    if(normalize_by_cluster_size){
        total_matching_cost /= cluster.members->size();
    }

    return total_matching_cost;
}

Cluster_Representative_Pair find_cluster_representative_pair_by_image_number(std::vector<Cluster_Representative_Pair>& all_crp, int target_img_num, std::vector<Cluster>& all_clusters){

    for(int i = 0; i < all_crp.size(); i++){
        Cluster_Representative_Pair current_crp = all_crp[i];

        Cluster corresponding_cluster = all_clusters[current_crp.cluster_index];

        for(int j = 0; j < corresponding_cluster.members->size();j++){
            int current_mem_img_num = (*(corresponding_cluster.members))[j];

            if(current_mem_img_num == target_img_num){
                return current_crp;
            }
        }
    }

    Cluster_Representative_Pair empty_crp;

    empty_crp.cluster_index = -1;
    empty_crp.representative_img_number = -1;

    return empty_crp;

}

bool check_if_img_num_is_cluster_representative(std::vector<Cluster_Representative_Pair>& all_crp, int img_num, Cluster_Representative_Pair& output_crp_of_img_num){
    for(int i = 0; i < all_crp.size(); i++){

        if (all_crp[i].representative_img_number == img_num){
            output_crp_of_img_num = all_crp[i];
            return true;
        }
        
    }

    return false;

}

std::string get_clustering_window_name_by_window_num(int window_num){

    std::string window_name_string = "";

    switch(window_num){
        case PRIMARY_CLUSTERING_WINDOW_IMG_ID:
            window_name_string += "Clusters_Primary";
            break;
        case SECONDARY_CLUSTERING_WINDOW_IMG_ID:
            window_name_string += "Clusters_Secondary";
            break;
        case COMBINED_CLUSTERING_WINDOW_IMG_ID:
            window_name_string += "Clusters_Combined";
            break;
        default:
            break;
    }

    return window_name_string;
}


void print_confusion_matrix(const Confusion_Matrix& confusion_matrix){

    std::cout << "tp: " << confusion_matrix.true_positives << " tn: " << confusion_matrix.true_negatives << " fp: " << confusion_matrix.false_positives << " fn: " << confusion_matrix.false_negatives << std::endl;
}

void calculate_confusion_matrix_of_clustering_to_pairwise_truth(Confusion_Matrix& confusion_matrix, std::vector<Matching_Result>& merged_matching_results, double threshold, const std::vector<Cluster> &clustering){

    confusion_matrix.true_positives = 0;
    confusion_matrix.true_negatives = 0;
    confusion_matrix.false_negatives = 0;
    confusion_matrix.false_positives = 0;

    for(int i = 0; i < merged_matching_results.size();i++){
        Matching_Result current_mr = merged_matching_results[i];

        int class_id1 = get_cluster_index_from_image_number(clustering,current_mr.id_1);
        int class_id2 = get_cluster_index_from_image_number(clustering,current_mr.id_2);

        bool in_same_cluster = (class_id1 == class_id2);

        if(current_mr.rel_quadr_cost < 0.1){
            if(in_same_cluster){
                confusion_matrix.false_positives++;
            }else{
                confusion_matrix.true_negatives++;
            }
        }else if(current_mr.rel_quadr_cost > 0.9){
            if(in_same_cluster){
                confusion_matrix.true_positives++;
            }else {
                confusion_matrix.false_negatives++;
            }

        }   
    }
    
}

void calculate_confusion_matrix_with_pairwise_truth(Confusion_Matrix& confusion_matrix, std::vector<Matching_Result>& merged_matching_results, double threshold, const std::vector<Matching_Result>& truth_matching_results){

    confusion_matrix.true_positives = 0;
    confusion_matrix.true_negatives = 0;
    confusion_matrix.false_negatives = 0;
    confusion_matrix.false_positives = 0;

    for(int i = 0; i < merged_matching_results.size();i++){
        Matching_Result current_mr = merged_matching_results[i];

        bool cur_mr_is_pos = (current_mr.rel_quadr_cost >= 0);

        for(int j = 0; j < truth_matching_results.size();j++){
            Matching_Result cur_truth_mr = truth_matching_results[j];

            if((cur_truth_mr.id_1 == current_mr.id_1 && cur_truth_mr.id_2 == current_mr.id_2) || (cur_truth_mr.id_1 == current_mr.id_2 && cur_truth_mr.id_2 == current_mr.id_1)){

                if(cur_truth_mr.rel_quadr_cost > 0.9){
                    if(cur_mr_is_pos){
                        confusion_matrix.true_positives++;
                    }else{
                        confusion_matrix.false_negatives++;
                    }
                }else if(cur_truth_mr.rel_quadr_cost < 0.1){
                    if(cur_mr_is_pos){
                        confusion_matrix.false_positives++;
                    }else{
                        confusion_matrix.true_negatives++;
                    }
                }

                break;
            }
        }

    }

}

void fill_member_vector_from_clustering(const std::vector<Cluster>& clustering, std::vector<Cluster_Member>& members){


    for(int c_id = 0; c_id < clustering.size();c_id++){
        Cluster current_cluster = clustering[c_id];

        for(int m_id = 0; m_id < current_cluster.members->size();m_id++){

            Cluster_Member new_cm;

            new_cm.cluster_id = c_id;

            new_cm.member_num = (*(current_cluster.members))[m_id];

            members.push_back(new_cm);
        }
    }

}

Cluster_Member find_cluster_member_by_member_num(std::vector<Cluster_Member>& all_members, int member_num){

    Cluster_Member empty_member;

    empty_member.cluster_id = -1;
    empty_member.member_num = -1;

    for(int i = 0; i < all_members.size();i++){
        Cluster_Member current_member = all_members[i];

        if(current_member.member_num == member_num){
            return current_member;
        }

    }

    std::cout << "could not find a match for member num: " << member_num << std::endl;

    return empty_member;

}


void calculate_variation_of_information_for_clusterings(const std::vector<Cluster>& reference_clustering,const std::vector<Cluster> &clustering, std::ofstream& output_file, double clustering_runtime){

    std::vector<Cluster_Member> members_of_c1;
    std::vector<Cluster_Member> members_of_c2;

    fill_member_vector_from_clustering(reference_clustering,members_of_c1);
    fill_member_vector_from_clustering(clustering,members_of_c2);

    int c1_total_size = members_of_c1.size();
    int c2_total_size = members_of_c2.size();

    size_t* labels_c1 = nullptr;
    size_t* labels_c2 = nullptr;

    if(c1_total_size != c2_total_size){

        std::cout << "ERROR in compare_clusterings the size of the clusterings does not match!" << std::endl;
        goto free_valid_clustering_vector;
    }

    labels_c1 = (size_t*)malloc(sizeof(size_t) * c1_total_size);
    labels_c2 = (size_t*)malloc(sizeof(size_t) * c2_total_size);

    for(int i = 0; i < c1_total_size;i++){
        Cluster_Member current_member_from_c1 = members_of_c1[i];

        labels_c1[i] = current_member_from_c1.cluster_id;

        Cluster_Member corresponding_mem_in_c2 = find_cluster_member_by_member_num(members_of_c2,current_member_from_c1.member_num);

        labels_c2[i] = corresponding_mem_in_c2.cluster_id;
    }

    output_file << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).precisionOfCuts() << ",";
    output_file << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).recallOfCuts() << ",";
    output_file << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).precisionOfJoins() << ",";
    output_file << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).recallOfJoins() << ",";
    output_file << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).index() << ",";
    output_file << andres::VariationOfInformation<>(labels_c1,labels_c1+c1_total_size,labels_c2).value() << ",";
    output_file << andres::VariationOfInformation<>(labels_c1,labels_c1+c1_total_size,labels_c2).valueFalseCut() << ",";
    output_file << andres::VariationOfInformation<>(labels_c1,labels_c1+c1_total_size,labels_c2).valueFalseJoin() << ",";
    output_file << clustering_runtime << ",";
    output_file << std::endl;



    //output_file << "RandIndex: " << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).index() << std::endl;
    //output_file << "RandIndex PoJ: " << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).precisionOfJoins() << std::endl;
    //output_file << "RandIndex RoJ: " << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).recallOfJoins() << std::endl;
    //output_file << "RandIndex PoC: " << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).precisionOfCuts() << std::endl;
    //output_file << "RandIndex RoC: " << andres::RandError<>(labels_c1,labels_c1+c1_total_size,labels_c2).recallOfCuts() << std::endl;
    //output_file << std::endl;
    //output_file << "VariationOfInformation: " << andres::VariationOfInformation<>(labels_c1,labels_c1+c1_total_size,labels_c2).value() << std::endl;
    //output_file << "VariationOfInformation FC: " << andres::VariationOfInformation<>(labels_c1,labels_c1+c1_total_size,labels_c2).valueFalseCut() << std::endl;
    //output_file << "VariationOfInformation FJ: " << andres::VariationOfInformation<>(labels_c1,labels_c1+c1_total_size,labels_c2).valueFalseJoin() << std::endl;

    free_valid_clustering_vector:

    free(labels_c1);
    free(labels_c2);

}

void calculate_confusion_matrix_from_clustering(Confusion_Matrix& confusion_matrix, const std::vector<Cluster>& reference_clustering, std::vector<Matching_Result>& merged_matching_results, double threshold, const std::vector<Cluster> &clustering){

    confusion_matrix.true_positives = 0;
    confusion_matrix.true_negatives = 0;
    confusion_matrix.false_negatives = 0;
    confusion_matrix.false_positives = 0;


    for(int i = 0; i < merged_matching_results.size();i++){
        Matching_Result current_mr = merged_matching_results[i];

        float similarity_value = current_mr.rel_quadr_cost;//; - threshold;

        int class_id1 = get_cluster_index_from_image_number(clustering,current_mr.id_1);
        int class_id2 = get_cluster_index_from_image_number(clustering,current_mr.id_2);

        int reference_id1 = get_cluster_index_from_image_number(reference_clustering,current_mr.id_1);
        int reference_id2 = get_cluster_index_from_image_number(reference_clustering,current_mr.id_2);

        //std::cout << current_mr.rel_quadr_cost << " " << current_mr.id_1 << " " << current_mr.id_2 << " " << class_id1 << " " << class_id2;

        if(reference_id1 == reference_id2){
            if(class_id1 != class_id2){
                //std::cout << " false negative " << std::endl;
                //separation_distances.false_negatives -= current_mr.rel_quadr_cost;
                confusion_matrix.false_negatives++;
            }else{
                //std::cout << " true positive " << std::endl;
                //separation_distances.true_positives += current_mr.rel_quadr_cost;
                confusion_matrix.true_positives++;
            }
        }else{
            if(class_id1 != class_id2){
                //std::cout << " true negative " << std::endl;
                //separation_distances.true_negatives -= current_mr.rel_quadr_cost;
                confusion_matrix.true_negatives++;
            }else{
                //std::cout << " false positive " << std::endl;
                //separation_distances.false_positives += current_mr.rel_quadr_cost;
                confusion_matrix.false_positives++;
            }
        }
    }

}


void calculate_confusion_matrix(Confusion_Matrix& confusion_matrix, std::vector<Matching_Result>& merged_matching_results, double threshold, const std::vector<Cluster> &clustering){

    confusion_matrix.true_positives = 0;
    confusion_matrix.true_negatives = 0;
    confusion_matrix.false_negatives = 0;
    confusion_matrix.false_positives = 0;

    std::vector<Matching_Result_Label> labels;

    labels.resize(merged_matching_results.size());

    //float separation_distance = 0.0;

    //Confusion_Matrix separation_distances{0.0,0.0,0.0,0.0};

    int counter = 0;

    for(int i = 0; i < merged_matching_results.size();i++){
        Matching_Result current_mr = merged_matching_results[i];

        float similarity_value = current_mr.rel_quadr_cost;//; - threshold;

        int class_id1 = get_cluster_index_from_image_number(clustering,current_mr.id_1);
        int class_id2 = get_cluster_index_from_image_number(clustering,current_mr.id_2);

        //std::cout << current_mr.rel_quadr_cost << " " << current_mr.id_1 << " " << current_mr.id_2 << " " << class_id1 << " " << class_id2;

        if(class_id1 == class_id2){
            if(similarity_value < 0){
                //std::cout << " false negative " << std::endl;
                //separation_distances.false_negatives -= current_mr.rel_quadr_cost;
                confusion_matrix.false_negatives++;
            }else{
                //std::cout << " true positive " << std::endl;
                //separation_distances.true_positives += current_mr.rel_quadr_cost;
                confusion_matrix.true_positives++;
            }
        }else{
            if(similarity_value < 0){
                //std::cout << " true negative " << std::endl;
                //separation_distances.true_negatives -= current_mr.rel_quadr_cost;
                confusion_matrix.true_negatives++;
            }else{
                //std::cout << " false positive " << std::endl;
                //separation_distances.false_positives += current_mr.rel_quadr_cost;
                confusion_matrix.false_positives++;
            }
        }
    }


    //std::cout << counter << std::endl;
    /*
    if(confusion_matrix.false_negatives == 0){
        separation_distances.false_negatives = 0;
    }else{
        separation_distances.false_negatives /= confusion_matrix.false_negatives;
    }

    if(confusion_matrix.true_negatives == 0){
        separation_distances.true_negatives = 0;
    }else{
            separation_distances.true_negatives /= confusion_matrix.true_negatives;
    }

    if(confusion_matrix.false_positives == 0){
        separation_distances.false_positives = 0;
    }else{
        separation_distances.false_positives /= confusion_matrix.false_positives;
    }

    if(confusion_matrix.true_positives == 0){
        separation_distances.true_positives = 0;
    }else{
        separation_distances.true_positives /= confusion_matrix.true_positives;
    }
    */
    
    

    //std::cout << "threshold: " << threshold << std::endl;
    //std::cout << " sep_dist_fn: " << separation_distances.false_negatives << std::endl;
    //std::cout << " sep_dist_tn: " << separation_distances.true_negatives << std::endl;
    //std::cout << " sep_dist_fp: " << separation_distances.false_positives << std::endl;
    //std::cout << " sep_dist_tp: " << separation_distances.true_positives << std::endl;
    //std::cout << " avg_dist   : " << (separation_distances.true_positives + separation_distances.false_negatives + separation_distances.true_negatives + separation_distances.false_positives) << std::endl;
    //std::cout << " accuracy: " << (separation_distances.true_positives + separation_distances.true_negatives) / (separation_distances.false_negatives + separation_distances.false_positives + separation_distances.true_negatives + separation_distances.true_positives) << std::endl; 
    //std::cout << " precision: " << (separation_distances.true_positives) / (separation_distances.false_positives + separation_distances.true_positives) << std::endl;
    //std::cout << " recall: " << (separation_distances.true_positives) / (separation_distances.false_negatives + separation_distances.true_positives) << std::endl;

    //float tpr = (confusion_matrix.true_positives) / (confusion_matrix.false_negatives + confusion_matrix.true_positives);
    //float fpr = (confusion_matrix.false_positives) / (confusion_matrix.false_positives + confusion_matrix.true_negatives);
    //std::cout << std::endl;

    //std::cout << "false negatives: " << confusion_matrix.false_negatives << std::endl;
    //std::cout << "true positives: " << confusion_matrix.true_positives << std::endl;
    //std::cout << "false positives: " << confusion_matrix.false_positives << std::endl;
    //std::cout << "true negatives: " << confusion_matrix.true_negatives << std::endl;
    //std::cout << std::endl;
    
}

TPR_FPR_Tuple get_tpr_fpr_tuple_from_confusion_matrix(const Confusion_Matrix& confusion_matrix, float used_threshold){

    float tp = confusion_matrix.true_positives;
    float tn = confusion_matrix.true_negatives;
    float fp = confusion_matrix.false_positives;
    float fn = confusion_matrix.false_negatives;

    float tpr = (confusion_matrix.true_positives) / (confusion_matrix.false_negatives + confusion_matrix.true_positives);
    float fpr = (confusion_matrix.false_positives) / (confusion_matrix.false_positives + confusion_matrix.true_negatives);
    float tnr = (confusion_matrix.true_negatives) / (confusion_matrix.true_negatives + confusion_matrix.false_positives);

    float ks_score = tpr - fpr;

    float acc = (confusion_matrix.true_positives + confusion_matrix.true_negatives) / (confusion_matrix.false_negatives + confusion_matrix.false_positives + confusion_matrix.true_negatives + confusion_matrix.true_positives);

    float prec = (confusion_matrix.true_positives) / (confusion_matrix.false_positives + confusion_matrix.true_positives);

    if((confusion_matrix.false_positives + confusion_matrix.true_positives) == 0){
        prec = 0;
    }

    float recall = (confusion_matrix.true_positives) / (confusion_matrix.false_negatives + confusion_matrix.true_positives);

    if((confusion_matrix.false_negatives + confusion_matrix.true_positives) == 0){
        recall = 0;
    }

    float f1_score = (2 * confusion_matrix.true_positives) / (2 * confusion_matrix.true_positives + confusion_matrix.false_positives + confusion_matrix.false_negatives);

    float mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));

    float prec_cuts = tn / (tn + fn);
    if((tn + fn) == 0){
        prec_cuts = 0;
    }

    float prec_joins =  tp / (tp + fp);
    if((tp + fp) == 0){
        prec_joins = 0;
    }

    float recall_cuts = tn / (tn + fp);
    if((tn + fp) == 0){
        recall_cuts = 0;
    }

    float recall_joins = tp / (tp + fn);
    if((tp + fn) == 0){
        recall_joins == 0;
    }

    float f1_joins = 2* ((prec_joins * recall_joins)/(prec_joins + recall_joins));
    if(prec_joins + recall_joins == 0){
        f1_joins = 0;
    }

    float f1_cuts = 2 * ((prec_cuts * recall_cuts)/(prec_cuts + recall_cuts));
    if(prec_cuts + recall_cuts == 0){
        f1_cuts = 0;
    }

    TPR_FPR_Tuple new_tpr_fpr_tuple{tn,tp,fn,fp,tpr,fpr,tnr,ks_score,used_threshold,acc,prec,recall,f1_score,f1_joins,f1_cuts,mcc,prec_cuts,prec_joins,recall_cuts,recall_joins};

    return new_tpr_fpr_tuple;
}

double calculate_binary_cross_entropy(std::vector<Matching_Result>& all_merged_matching_results, bool normalize_by_num_occurences, const std::vector<Cluster> &clustering){

    double bce = 0.0;

   // std::cout << all_merged_matching_results.size() << std::endl;

    double bce_of_positive_class = 0.0;
    double bce_of_negative_class = 0.0;

    int num_occurences_pos_class = 0;
    int num_occurences_neg_class = 0;

    for(int i = 0; i < all_merged_matching_results.size();i++){
        Matching_Result current_mr = all_merged_matching_results[i];

        //std::cout << current_mr.id_1 << " " << current_mr.id_2 << " " << current_mr.rel_quadr_cost;

        int label_1 = get_cluster_index_from_image_number(clustering,current_mr.id_1);// current_mr.id_1 / 10;
        int label_2 = get_cluster_index_from_image_number(clustering,current_mr.id_2);//current_mr.id_2 / 10;

        if(label_1 == label_2){
            //std::cout << " " << -std::log(current_mr.rel_quadr_cost);
            num_occurences_pos_class++;
            bce_of_positive_class += -std::log(current_mr.rel_quadr_cost);
            bce += -std::log(current_mr.rel_quadr_cost);
        }else{
            //std::cout << " " << -std::log(1.0 - current_mr.rel_quadr_cost);
            num_occurences_neg_class++;
            bce_of_negative_class += -std::log(1.0 - current_mr.rel_quadr_cost); 
            bce += -std::log(1.0 - current_mr.rel_quadr_cost);
        }
        //std::cout << std::endl;
    }

    if(num_occurences_pos_class != 0){
        bce_of_positive_class /= (double)num_occurences_pos_class;
    }

    if(num_occurences_neg_class != 0){
        bce_of_negative_class /= (double)num_occurences_neg_class;
    }

    double weighted_bce = (bce_of_positive_class + bce_of_negative_class) / 2.0;

    bce /= (double)all_merged_matching_results.size();

    //std::cout << "bce: " << bce << std::endl;
    //std::cout << "wbce: " << weighted_bce << std::endl;
    
    if(normalize_by_num_occurences){
        return weighted_bce;
    }else{
        return bce;
    }

}

bool read_model_parameters_from_file(std::filesystem::path model_param_file, All_Cost_Parameters& all_read_model_params){

    std::ifstream input_file;

    input_file.open(model_param_file);

    std::string filename_extension = model_param_file.filename().extension().string();

    //std::cout << filename_extension << std::endl;

    All_Cost_Parameters cost_params_from_file{10000.0,10000.0,1.0,1.0,1.0,1.0,1.0};

    if (input_file.fail()) {
        std::cout << "File: " << model_param_file << " to read model parameters from COULD NOT BE OPENED" << std::endl;
        return false;
    }

    if(filename_extension == ".mpb"){
        //std::cout << "reading binary" << std::endl;

        float binary_parameters[5];
        input_file.read((char*)binary_parameters,sizeof(float)*5);
        /*
        for(int  i = 0; i < 5; i++){
            std::cout << binary_parameters[i] << " ";
        }
        std::cout << std::endl;
        */
    }

    const int linebuffer_size = 16777216;
    //char line_buffer[linebuffer_size];

    char* line_buffer = (char*)malloc(sizeof(char) * linebuffer_size);

    char* first_token;
    char* second_token;
    char delimiter[] = ",:=";

    int num_read_params = 0;

    while(!input_file.eof()){
        input_file.getline(line_buffer,linebuffer_size);
        //std::cout << line_buffer << std::endl;

        //std::cout << line_buffer << std::endl;

        first_token = strtok(line_buffer, delimiter);
        second_token = strtok(NULL,delimiter);

        if(first_token == NULL){
            continue;
        }

        if(strcmp(first_token,COLOR_COST_OFFSET_STRING) == 0){
            cost_params_from_file.color_offset = atof(second_token);
            num_read_params++;
            continue;
        }

        if(strcmp(first_token,DIST_COST_OFFSET_STRING) == 0){
            cost_params_from_file.dist_offset = atof(second_token);
            num_read_params++;
            continue;
        }

        if(strcmp(first_token,ANGLE_COST_OFFSET_STRING) == 0){
            cost_params_from_file.angle_offset = atof(second_token);
            num_read_params++;
            continue;
        }

        if(strcmp(first_token,COLOR_TO_DISTANCE_WEIGHT_STRING) == 0){
            cost_params_from_file.color_to_dist_weight = atof(second_token);
            num_read_params++;
            continue;
        }

        if(strcmp(first_token,UNARY_TO_QUADR_WEIGHT_STRING) == 0){
            cost_params_from_file.unary_to_to_quadr_weight = atof(second_token);
            num_read_params++;
            continue;
        }

    }

    if(num_read_params < 5){
        std::cout << "not all parameters were present in file: " << model_param_file << std::endl;
    }

    all_read_model_params = cost_params_from_file;

    input_file.close();

    free(line_buffer);

    return true;
}

void write_model_parameters_to_file(std::string filename, All_Cost_Parameters& all_cost_parameters){

    std::filesystem::path output_file_path = get_data_folder_path();

    output_file_path.append(filename);

    std::ofstream output_file;

    output_file.open(output_file_path);

    float flat_parameter_array[5];
    flat_parameter_array[0] = all_cost_parameters.color_offset;
    flat_parameter_array[1] = all_cost_parameters.dist_offset;
    flat_parameter_array[2] = all_cost_parameters.angle_offset;
    flat_parameter_array[3] = all_cost_parameters.color_to_dist_weight;
    flat_parameter_array[4] = all_cost_parameters.unary_to_to_quadr_weight;


    output_file.write((const char*)flat_parameter_array,sizeof(float) * 5);

    output_file.close();
}

void print_model_parameters(All_Cost_Parameters& cost_params, bool short_print){
    
    if(short_print){
        std::cout << cost_params.color_offset
        << " " << cost_params.dist_offset
        << " " << cost_params.angle_offset
        << " " << cost_params.color_to_dist_weight
        << " " << cost_params.unary_to_to_quadr_weight << std::endl;
    }else{
        std::cout << std::endl;
        std::cout << COLOR_COST_OFFSET_STRING << ": " << cost_params.color_offset << std::endl;
        std::cout << DIST_COST_OFFSET_STRING << ": " << cost_params.dist_offset << std::endl;
        std::cout << ANGLE_COST_OFFSET_STRING << ": " << cost_params.angle_offset << std::endl;
        std::cout << COLOR_TO_DISTANCE_WEIGHT_STRING << ": " << cost_params.color_to_dist_weight << std::endl;
        std::cout << UNARY_TO_QUADR_WEIGHT_STRING << ": " << cost_params.unary_to_to_quadr_weight << std::endl;

    }

}

double hh_mm_ss_time_string_to_double(std::string time){
    std::string delimiter = ":";

    int pos_of_delimiter = time.find(delimiter);
    std::string hh_token = time.substr(0,pos_of_delimiter);
    time.erase(0,pos_of_delimiter + delimiter.length());

    pos_of_delimiter = time.find(delimiter);
    std::string mm_token = time.substr(0,pos_of_delimiter);
    time.erase(0,pos_of_delimiter + delimiter.length());

    pos_of_delimiter = time.find(delimiter);
    std::string ss_token = time.substr(0,pos_of_delimiter);
    time.erase(0,pos_of_delimiter + delimiter.length());

    double hours_to_microseconds = 60.0 * 60.0 * 1000.0 * 1000.0;
    double minutes_to_microseconds = 60.0 * 1000.0 * 1000.0;
    double seconds_to_microseconds = 1000.0 * 1000.0;

    double total_runtime = std::stod(hh_token) * hours_to_microseconds + std::stod(mm_token) * minutes_to_microseconds + std::stod(ss_token) * seconds_to_microseconds;

    //std::cout << "hh_token: " << hh_token << std::endl;
    //std::cout << "mm_token: " << mm_token << std::endl;
    //std::cout << "ss_token: " << ss_token << std::endl; 
    //std::cout << total_runtime << std::endl;

    return total_runtime;
}

std::string model_parameters_to_string(All_Cost_Parameters& cost_params){
    std::string model_param_string = std::to_string(cost_params.color_offset) + "_";
    model_param_string += std::to_string(cost_params.dist_offset) + "_";
    model_param_string += std::to_string(cost_params.angle_offset) + "_";
    model_param_string += std::to_string(cost_params.color_to_dist_weight) + "_";
    model_param_string += std::to_string(cost_params.unary_to_to_quadr_weight);

    return model_param_string;
}

std::string get_string_from_search_strategy(Model_Parameter_Selection_Strategy sel_strat){
    switch(sel_strat){
        case SEL_STRAT_NO_SEARCH:
            return SEL_STRAT_NO_SEARCH_STRING;
            break;

        case SEL_STRAT_EXHAUSTIVE_ADJ:
            return SEL_STRAT_EXHAUSTIVE_ADJ_STRING;
            break;

        case SEL_STRAT_LINE_SEARCH:
            return SEL_STRAT_LINE_SEARCH_STRING;
            break;

        case SEL_STRAT_SIM_ANN:
            return SEL_STRAT_SIM_ANN_STRING;
            break;
        default:
            std::cout << "Unknown selection strategy type in get_string_from_search_strategy" << std::endl;
            return "";
            break;
    }
}

std::string get_string_from_metric_type(Learning_Metric_Types metric_type){

    switch (metric_type)
    {
    case LM_BCE:
        return LM_BCE_STRING;
        break;
    case LM_WBCE:
        return LM_WBCE_STRING;
        break;    
    case LM_FN_FP_NUM:
        return LM_FN_FP_NUM_STRING;
        break;
    case LM_ACC:
        return LM_ACC_STRING;
        break;
    case LM_F1_SCORE:
        return LM_F1_SCORE_STRING;
        break;
    case LM_TPR_TNR_AVG:
        return LM_TPR_TNR_AVG_STRING;
        break;
    case LM_MCC:
        return LM_MCC_STRING;
        break;
    default:
        std::cout << "Unknown metric type in get_string_from_metric_type" << std::endl;
        return "";
        break;
    }

}

Learning_Metric_Types get_metric_type_from_string(std::string metric_type_string){

    if(strcmp(metric_type_string.c_str(),LM_BCE_STRING) == 0){
        return LM_BCE;
    }

    if(strcmp(metric_type_string.c_str(),LM_WBCE_STRING) == 0){
        return LM_WBCE;
    }

    if(strcmp(metric_type_string.c_str(),LM_FN_FP_NUM_STRING) == 0){
        return LM_FN_FP_NUM;
    }

    if(strcmp(metric_type_string.c_str(),LM_ACC_STRING) == 0){
        return LM_ACC;
    }

    if(strcmp(metric_type_string.c_str(),LM_F1_SCORE_STRING) == 0){
        return LM_F1_SCORE;
    }

    if(strcmp(metric_type_string.c_str(),LM_TPR_TNR_AVG_STRING) == 0){
        return LM_TPR_TNR_AVG;
    }

    if(strcmp(metric_type_string.c_str(),LM_MCC_STRING) == 0){
        return LM_MCC;
    }

    std::cout << "passed string: " << metric_type_string << " did not match any of the valid options: " << LM_BCE_STRING << ", " << LM_WBCE_STRING << ", " << LM_FN_FP_NUM_STRING << ", " << LM_ACC_STRING << ", " << LM_F1_SCORE_STRING << ", " << LM_TPR_TNR_AVG_STRING << ", " << LM_MCC_STRING << std::endl;
    std::cout << "using default: " << get_string_from_metric_type(DEFAULT_LEARNING_METRIC) << " instead" << std::endl;

    return DEFAULT_LEARNING_METRIC;

}

Model_Parameter_Selection_Strategy get_search_strategy_from_string(std::string sel_strat_string){

    if(strcmp(sel_strat_string.c_str(),SEL_STRAT_EXHAUSTIVE_ADJ_STRING) == 0){
        return SEL_STRAT_EXHAUSTIVE_ADJ;
    }

    if(strcmp(sel_strat_string.c_str(),SEL_STRAT_LINE_SEARCH_STRING) == 0){
        return SEL_STRAT_LINE_SEARCH;
    }

    if(strcmp(sel_strat_string.c_str(),SEL_STRAT_NO_SEARCH_STRING) == 0){
        return SEL_STRAT_NO_SEARCH;
    }

    if(strcmp(sel_strat_string.c_str(),SEL_STRAT_SIM_ANN_STRING) == 0){
        return SEL_STRAT_SIM_ANN;
    }

    std::cout << "passed string: " << sel_strat_string << " did not match any of the valid options: "  << SEL_STRAT_EXHAUSTIVE_ADJ_STRING << ", " << SEL_STRAT_LINE_SEARCH_STRING << ", " << SEL_STRAT_NO_SEARCH_STRING << ", " << SEL_STRAT_SIM_ANN_STRING << std::endl;
    std::cout << "using default: " << get_string_from_search_strategy(DEFAULT_SELECTION_STRATEGY) << " instead" << std::endl;

    return DEFAULT_SELECTION_STRATEGY;

}

int get_cluster_index_from_image_number(const std::vector<Cluster> &clustering, int image_number){

    for(int i = 0; i < clustering.size();i++){

        if(check_if_image_number_is_contained_in_vector(*(clustering[i].members),image_number)){
            return i;
        }
    }

    return -1;

}

void read_learning_task(std::string learning_task_file_name, Input_Arguments& input_args){

    std::filesystem::path learning_task_file_path = get_data_folder_path();
    learning_task_file_path.append(learning_task_file_name);

    std::ifstream input_file;

    input_file.open(learning_task_file_path);

    if (input_file.fail()) {
        std::cout << "File: " << learning_task_file_name << " to read learing task COULD NOT BE OPENED" << std::endl;
        return;
    }


    const int linebuffer_size = 16777216;
    //char line_buffer[linebuffer_size];

    char* line_buffer = (char*)malloc(sizeof(char) * linebuffer_size);

    char* first_token;
    char* second_token;
    char delimiter[] = "=";

    int num_read_params = 0;

    while(!input_file.eof()){

        //std::cout << "begin read " << std::endl;
        input_file.getline(line_buffer,linebuffer_size);
        //std::cout << line_buffer << std::endl;


        first_token = strtok(line_buffer, delimiter);
        //std::cout << first_token << std::endl;
        second_token = strtok(NULL,delimiter);

        //std::cout << "read tokens: " << first_token << " " << second_token  << std::endl;

        if(first_token == NULL){
            continue;
        }

        
        if(strcmp_ignore_leading_hyphens(first_token,FEATURE_VECTOR_FILE_NAME_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.feature_vector_file_name = std::string(second_token);
                //std::cout << input_args.feature_vector_file_name << std::endl;
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,MODEL_PARAMETER_FILE_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.model_parameter_file_name = std::string(second_token);
                input_args.read_model_parameters_from_file = true;
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,RUNTIME_LIMIT_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.runtime_limit = hh_mm_ss_time_string_to_double(std::string(second_token));
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,LEARNING_METRIC_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.lm_type = get_metric_type_from_string(std::string(second_token));
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,LEARNING_SEARCH_STRATEGY_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.sel_strat = get_search_strategy_from_string(std::string(second_token));
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,REFERENCE_CLUSTERING_FILE_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.reference_clustering_file_name = std::string(second_token);
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,SEARCH_DIM_ORDER_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                std::string sec_tok_str(second_token);
                parse_search_order_string(input_args.search_order,sec_tok_str.c_str());
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,SEARCH_STEP_SIZE_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.initial_search_step_size = atof(second_token);
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,SIM_ANN_TEMP_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.sim_ann_init_temp = atof(second_token);
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,SIM_ANN_COOLING_RATE_STRING) == 0){
            if(second_token != NULL){
                input_args.sim_ann_cooling_rate = atof(second_token);
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,SIM_ANN_STD_DEV_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.sim_ann_std_dev = atof(second_token);
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        if(strcmp_ignore_leading_hyphens(first_token,SIM_ANN_RESTART_THRES_ARGUMENT_STRING) == 0){
            if(second_token != NULL){
                input_args.sim_ann_restart_thresh = atoi(second_token);
            }else{
                print_missing_argument_value_error(first_token);
            }
            continue;
        }

        print_unrecognised_argument_error(first_token);
    }

    std::cout << "finished read learning task" << std::endl;
    input_file.close();

    free(line_buffer);

    return;

}

bool strcmp_ignore_leading_hyphens(const char* str_1, const char* str_2){

    std::string modified_str1 = std::string(str_1);
    std::string modified_str2 = std::string(str_2);

    while(modified_str1[0] == '-'){
        modified_str1 = modified_str1.substr(1);
    }

    while(modified_str2[0] == '-'){
        modified_str2 = modified_str2.substr(1);
    }

    return modified_str1.compare(modified_str2);

}

Search_Dim get_search_dim_by_name(const char* search_dim_name){

    if(strcmp(search_dim_name,"SD_ANGLE_OFFSET") == 0){
        return SD_ANGLE_OFFSET;
    }

    if(strcmp(search_dim_name,"SD_COLOR_OFFSET") == 0){
        return SD_COLOR_OFFSET;
    }

    if(strcmp(search_dim_name,"SD_DIST_OFFSET") == 0){
        return SD_DIST_OFFSET;
    }

    if(strcmp(search_dim_name,"SD_COLOR_TO_DIST_WEIGHT") == 0){
        return SD_COLOR_TO_DIST_WEIGHT;
    }

    if(strcmp(search_dim_name,"SD_UNARY_TO_QUADR_WEIGHT") == 0){
        return SD_UNARY_TO_QUADR_WEIGHT;
    }

    return SD_NOT_SET;
}

void parse_search_order_string(std::vector<Search_Dim>* search_order,const char* str){


    std::string search_order_string = std::string(str);

    std::string delimiter = ",";

    int index_of_next_delimiter = search_order_string.find(delimiter);


    while(index_of_next_delimiter != std::string::npos){
        std::string token = search_order_string.substr(0,index_of_next_delimiter);
        search_order_string = search_order_string.substr(index_of_next_delimiter+1,search_order_string.length());
        Search_Dim next_search_dim = get_search_dim_by_name(token.c_str());

        if(next_search_dim != SD_NOT_SET){
            search_order->push_back(next_search_dim);
        }

        index_of_next_delimiter = search_order_string.find(delimiter);
        //return;

    }

}

float sample_gaussian(float mu, float sigma) {

	static const float two_pi = M_PI * 2.0f;

	static float z1;
	static int generate;

	generate = !generate;

	if (!generate) {
		return z1 * sigma + mu;
	}


	float u1, u2;

	do {
		u1 = rand() * (1.0f / RAND_MAX);
		u2 = rand() * (1.0f / RAND_MAX);

	} while (u1 <= FLT_EPSILON);

	float z0;
	z0 = sqrt(-2.0f * logf(u1)) * cosf(two_pi * u2);
	z1 = sqrt(-2.0f * logf(u1)) * sinf(two_pi * u2);

	return z0 * sigma + mu;
}

float get_matching_value_from_decision_string(std::string decision_string){

    if(decision_string.compare(DEC_NOT_MADE_STRING) == 0){
        return -1.0f;
    }

    if(decision_string.compare(DEC_UNSURE_STRING) == 0){
        return 0.5f;
    }

    if(decision_string.compare(DEC_SAME_STRING) == 0){
        return 1.0f;
    }

    if(decision_string.compare(DEC_DIFF_STRING) == 0){
        return 0.0f;
    }

    return -1.0f;
}

void read_pairwise_comparison_as_matching_results(std::filesystem::path file_path, std::vector<Matching_Result>& all_matching_results){

    std::ifstream input_file;

    input_file.open(file_path);

    if (input_file.fail()) {
        std::cout << "File: " << file_path << " to read clustering from COULD NOT BE OPENED" << std::endl;
        return;
    }

    const int linebuffer_size = 16777216;
    //char line_buffer[linebuffer_size];

    char* line_buffer = (char*)malloc(sizeof(char) * linebuffer_size);

    std::vector<int> all_image_numbers;

    char* first_token;
    char delimiter[] = ",:";

    bool currently_reading_decisions = false;
    bool finished_reading_decisions = false;

    while(!input_file.eof() && !finished_reading_decisions){
        input_file.getline(line_buffer,linebuffer_size);

        first_token = strtok(line_buffer, delimiter);

        if(first_token == NULL){
            return;
        }

        if(strcmp(first_token,"image_set") == 0){
            continue;
        }

        if(strcmp(first_token,"image_numbers") == 0 ){
            continue;
        }

        if(strcmp(first_token,"begin_decisions") == 0 ){
            currently_reading_decisions = true;
            continue;
        }

        if(strcmp(first_token,"end_decisions") == 0 ){
            currently_reading_decisions = false;
            continue;
        }

        if(currently_reading_decisions){
            int first_img_num = atoi(first_token);

            first_token = strtok(NULL, delimiter);

            int second_img_num = atoi(first_token);

            first_token = strtok(NULL, delimiter);

            std::string decision_string = first_token;

            float matching_val = get_matching_value_from_decision_string(decision_string);

            Matching_Result new_mr;

            new_mr.id_1 = first_img_num;
            new_mr.id_2 = second_img_num;
            //new_mr.additional_viz_data_id1_to_id2 = nullptr;
            //new_mr.additional_viz_data_id2_to_id1 = nullptr;
            new_mr.assignment = nullptr;

            new_mr.additional_viz_data_id1_to_id2 = new Matching_Result_Additional_Viz_Data;
            //new_mr.additional_viz_data_id1_to_id2->assignment = nullptr;
            new_mr.additional_viz_data_id1_to_id2->candidate_image = nullptr;
            new_mr.additional_viz_data_id1_to_id2->feature_image = nullptr;
            new_mr.additional_viz_data_id1_to_id2->features = nullptr;
            new_mr.additional_viz_data_id1_to_id2->assigned_candidates = nullptr;

            new_mr.additional_viz_data_id2_to_id1 = new Matching_Result_Additional_Viz_Data;
            //new_mr.additional_viz_data_id2_to_id1->assignment = nullptr;
            new_mr.additional_viz_data_id2_to_id1->candidate_image = nullptr;
            new_mr.additional_viz_data_id2_to_id1->feature_image = nullptr;
            new_mr.additional_viz_data_id2_to_id1->features = nullptr;
            new_mr.additional_viz_data_id2_to_id1->assigned_candidates = nullptr;

            new_mr.rel_quadr_cost = matching_val;
            new_mr.rel_quadr_cost_optimal = matching_val;

            new_mr.linear_cost_per_candidate = 0.0;
            new_mr.linear_cost_per_feature = 0.0;

            all_matching_results.push_back(new_mr);


            new_mr;

            new_mr.id_1 = second_img_num;
            new_mr.id_2 = first_img_num;
            //new_mr.additional_viz_data_id1_to_id2 = nullptr;
            //new_mr.additional_viz_data_id2_to_id1 = nullptr;
            new_mr.assignment = nullptr;

            new_mr.additional_viz_data_id1_to_id2 = new Matching_Result_Additional_Viz_Data;
            //new_mr.additional_viz_data_id1_to_id2->assignment = nullptr;
            new_mr.additional_viz_data_id1_to_id2->candidate_image = nullptr;
            new_mr.additional_viz_data_id1_to_id2->feature_image = nullptr;
            new_mr.additional_viz_data_id1_to_id2->features = nullptr;
            new_mr.additional_viz_data_id1_to_id2->assigned_candidates = nullptr;

            new_mr.additional_viz_data_id2_to_id1 = new Matching_Result_Additional_Viz_Data;
            //new_mr.additional_viz_data_id2_to_id1->assignment = nullptr;
            new_mr.additional_viz_data_id2_to_id1->candidate_image = nullptr;
            new_mr.additional_viz_data_id2_to_id1->feature_image = nullptr;
            new_mr.additional_viz_data_id2_to_id1->features = nullptr;
            new_mr.additional_viz_data_id2_to_id1->assigned_candidates = nullptr;

            new_mr.rel_quadr_cost = matching_val;
            new_mr.rel_quadr_cost_optimal = matching_val;

            new_mr.linear_cost_per_candidate = 0.0;
            new_mr.linear_cost_per_feature = 0.0;

            all_matching_results.push_back(new_mr);
                    
        }
    }

    input_file.close();

    free(line_buffer);

}