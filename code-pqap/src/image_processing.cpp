#include "image_processing.h"
#include "python_handler.h"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <thread>
#include <random>

#define THRESHOLD 5
#define THRESHOLD_16_BIT 900

#define MAX_VALUE_16_BIT (2<<16)-1

#define WRITE_OUTPUT_IMAGES true

#define MORPH_KERNEL_SIZE 7
#define MORPH_KERNEL_SIZE_SINGLE_IMG 5

#define MORPH_CLOSE_FIRST true
#define MORPH_NUM_CLOSING_ITERATIONS 2
#define MORPH_NUM_OPENING_ITERATIONS 2

#define MORPH_NUM_CLOSING_ITERATIONS_SINGLE_IMG 1
#define MORPH_NUM_OPENING_ITERATIONS_SINGLE_IMG 1

#define MINIMAL_PIXEL_COUNT_OF_OUTPUT_IMAGE 6000

#define MAXIMAL_ASPECT_RATIO_OF_OUTPUT_IMAGE 2.0

#define NUM_RAYPAIRS_FOR_MIDPOINT_ESTIMATION 10


#define SEGMENT_USING_STARDIST true

#define USE_PEAKS_AS_SEEDS true

//#define SOFT_CROP true
#define SHOW_SINGLE_OUTPUT_IMAGES false
#define SHOW_INPUT_IMAGES false
#define SHOW_PREPROCESSING_IMAGES false

#define TOLERANCE_MATCHING_THRESHOLD 0.05

#define SYMMETRIZE_MATCHING_MATRIX true

#define SHOW_FEATURE_EXTRACTION_DEBUG false

#define FEATURE_EXTRACTION_DEBUG(expr) if(SHOW_FEATURE_EXTRACTION_DEBUG){expr}

#define SHOW_FEATURE_EXTRACTION_OUTPUT false

#define FEATURE_EXTRACTION_OUTPUT(expr) if(SHOW_FEATURE_EXTRACTION_OUTPUT){expr}

typedef enum Stardist_Scaling_Mode{
    STARDIST_QUARTER_IMAGE,
    STARDIST_DOWNSCALING,
    STARDIST_ORIGINAL_IMAGE
}Stardist_Scaling_Mode;

typedef enum Subivision_Mode{
    SUBDIVIDE_SIMPLE_THRESHOLD,
    SUBDIVIDE_LAPLACIAN,
} Subivision_Mode;

typedef struct Cluster_Center_Point{

    unsigned int num_points;
    unsigned int cluster_index;
    double acc_row;
    double acc_col;
    double acc_row_sqr;
    double acc_col_sqr;
    int min_row;
    int max_row;
    int min_col;
    int max_col;

}Cluster_Center_Point;

typedef struct Selected_Cell {
    int x;
    int y;
    int pixel_per_cell;
}Selected_Cell;

typedef struct Selected_Clustering_Window_Cells{
    int first_selected_row;
    int first_selected_col;
    int first_selected_img_num;
    int first_window_num;
    int second_selected_row;
    int second_selected_col;
    int second_selected_img_num;
    int second_window_num;
}Selected_Clustering_Window_Cells;

typedef struct Raylength_and_Dir{
    double length;
    cv::Vec2f dir;
}Raylength_and_Dir;

typedef struct Matching_Matrix_Mouse_Callback_Input{
    Selected_Cell* selected_cell;
    std::vector<cv::Mat*>* square_organoid_images;
    std::vector<Matching_Result>* all_matching_results;
    std::vector<Image_Features_Pair>* all_feature_vectors;
    int individual_image_size;
    float cost_threshold;
    Matching_Visualization_Type viz_type;
    uint32_t flags;
    std::vector<int>* image_order;
    bool use_image_order;
    All_Cost_Parameters cost_params;
}Matching_Matrix_Mouse_Callback_Input;

typedef struct Clustering_Visualization_Mouse_Callback_Input{
    int window_number;
    std::vector<cv::Mat*>* square_organoid_images;
    std::vector<Matching_Result>* all_matching_results;
    std::vector<Image_Features_Pair>* all_feature_vectors;
    int individual_image_size;
    std::vector<Single_Cluster_Layout>* cluster_layout;
    std::vector<Cluster_Representative_Pair>* selected_cluster_representatives;
    Selected_Cell* selected_cell;
    std::vector<Cluster>* all_clusters;
    Selected_Clustering_Window_Cells* selected_cells_in_clustering_windows;
    All_Cost_Parameters cost_params;
    int mouse_over_rep_img_num;
}Clustering_Visualization_Mouse_Callback_Input;

struct Queue_Feature_Point_Compare{

    bool operator()(const Feature_Point_Data& lhs, const Feature_Point_Data& rhs){

        return lhs.peak_value < rhs.peak_value;
    }
};

struct Single_Cluster_Layout_Size_Compare{
    bool operator()(const Single_Cluster_Layout& lhs, const Single_Cluster_Layout& rhs){

        return lhs.num_members > rhs.num_members;
    }
}Single_Cluster_Layout_Size_Compare;

Selected_Cell selected_cell_in_matching_matrix;
Matching_Matrix_Mouse_Callback_Input global_matching_matrix_mouse_callback_input;

Selected_Cell selected_cell_in_clustering;
Selected_Clustering_Window_Cells global_selected_clustering_window_cells;
//Clustering_Visualization_Mouse_Callback_Input global_clustering_mouse_callback_input{PRIMARY_CLUSTERING_WINDOW_IMG_ID,nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr,nullptr,-1,-1,-1,-1,-1,-1};

//Clustering_Visualization_Mouse_Callback_Input secondary_global_clustering_mouse_callback_input{SECONDARY_CLUSTERING_WINDOW_IMG_ID,nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr,nullptr,-1,-1,-1,-1,-1,-1};

//Clustering_Visualization_Mouse_Callback_Input combined_global_clustering_mouse_callback_input{COMBINED_CLUSTERING_WINDOW_IMG_ID,nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr,nullptr,-1,-1,-1,-1,-1,-1};


Clustering_Visualization_Mouse_Callback_Input global_clustering_mouse_callback_inputs[NUM_CLUSTERING_WINDOWS];

void get_coordinates_in_cluster_visualization_by_image_number(int image_number, int* col, int* row, std::vector<Single_Cluster_Layout>* cluster_layout, std::vector<Cluster>* all_clusters);

cv::Vec2i calculate_cluster_layout(std::vector<Single_Cluster_Layout>& single_cluster_layouts);

bool sort_by_angle_compare_function(Feature_Point_Data& f_point_a,Feature_Point_Data& f_point_b){
    return f_point_a.angle < f_point_b.angle;
}

void draw_cross(cv::Mat& img, const cv::Point2i& center_pos, const cv::Scalar& color, int size);

void find_optimial_viz_rotation_angle(double& rotation_angle, std::vector<int>* assignment, std::vector<Feature_Point_Data>* features,std::vector<Feature_Point_Data>* assigned_candidates );

bool check_red_channel_acceptance_metric(Feature_Point_Data* feature_point, float relative_distance_to_closest, float global_mean, int current_number_red_feature_points, int current_number_features_in_other_channels, bool are_candidates, float gradient_value);

cv::Mat get_ccw_rotation_matrix_by_image_size_and_degrees(int cols, int rows, double degrees);

cv::Mat get_ccw_rotation_matrix_by_center_point_and_degrees(int center_col, int center_row, double degrees);

void load_and_square_organoid_image(cv::Mat& organoid_image, std::filesystem::path organoid_image_path);

void rotate_image_by_degrees_ccw(cv::Mat& input, cv::Mat& output, double degrees);

void rotate_image_by_degrees_ccw_around_point(cv::Mat& input, cv::Mat& output, double degrees, int rotation_origin_col, int rotation_origin_row);

void join_feature_vectors(std::vector<Feature_Point_Data>& output_feature_vector, std::vector<Feature_Point_Data>* red_feature_vector,std::vector<Feature_Point_Data>* blue_feature_vector,std::vector<Feature_Point_Data>* green_feature_vector);

bool check_if_existing_feature_point_in_range(int row, int col, int center_value, int range, std::vector<Feature_Point_Data>* existing_feature_points);

void connect_all_feature_points(std::vector<Feature_Point_Data>* feature_points, cv::Mat& output_image);

Feature_Point_Data find_closest_feature_point(Feature_Point_Data* reference_point, std::vector<Feature_Point_Data>* feature_points, float& distance_to_closest);

Feature_Point_Data find_closest_feature_point_and_set_id(Feature_Point_Data* reference_point,int reference_point_id,std::vector<Feature_Point_Data>* feature_points, float& distance_to_closest, int* set_ids, int& closest_feature_point_id, int start_id, bool& found_new_closest);

void put_feature_points_on_image(cv::Mat& output_image, std::vector<Feature_Point_Data>* feature_points, double global_mean, double global_std_dev, double presience_threshold);

void subdivide_using_peaks(cv::Mat* filled_contours_img, cv::Mat* original_img, const Organoid_Image_Header* header, Subivision_Mode sub_mode, unsigned int& number_offset, bool was_already_subdivided, cv::Mat* watershed_input_image, cv::Vec2i root_image_size, cv::Vec2i cluster_start_point);

void subdivide_using_stardist(cv::Mat* filled_obj_img, Stardist_Scaling_Mode scaling_mode, int downscale_factor, cv::Mat* original_organoid_image, const Organoid_Image_Header* header);

void mask_laplace_img_with_distance_img(cv::Mat* laplace_img, cv::Mat* distance_img);

void calculate_spherical_feature_point_coordinates(std::vector<Feature_Point_Data>* feature_points, cv::Mat* mask_image, cv::Mat& output_image, int center_row, int center_col);

cv::Vec2i trace_line_until_organoid_boundary(cv::Mat* mask_image, cv::Vec2i start, cv::Vec2f dir);

template<typename T>
void cut_and_mask_single_organoid(Cluster_Center_Point* cluster, cv::Mat* original_img, cv::Mat* contour_img, cv::Mat* output_img, bool write_inverted_images, cv::Mat& output_mask_img);

bool check_single_output_image_conditions(Cluster_Center_Point* cluster, cv::Mat* original_img, cv::Vec2i root_image_size, cv::Vec2i cluster_start_point, bool was_already_subdivided);

void adaptive_mask_upscale(cv::Mat* downscaled_mask, int downscale_factor);

cv::Point2i get_approximated_organoid_center(cv::Mat* mask_input);

cv::Point2i get_approximated_organoid_center_ray_tracing(cv::Mat* mask_input, int num_ray_pairs);

cv::Point2i get_approximated_organoid_center_ray_tracing_opposites(cv::Mat* mask_input, int num_ray_pairs);

int calculate_local_search_dim_from_image_size(cv::Size2i image_size, double target_percentage_of_image_size);

void fill_costs_from_matching_results_vector(std::vector<double>& costs_from_matching_results, std::vector<Image_Features_Pair>* all_feature_vectors, std::vector<Matching_Result>* all_matching_results, bool use_image_order, std::vector<int>* image_order);

void fill_matching_matrix_in_feature_vector_range(int start, int end,cv::Mat* matching_matrix_img, double cost_threshold, int individual_image_size, std::vector<double>* costs_from_matching_results, std::vector<Image_Features_Pair>* all_feature_vectors, std::vector<Matching_Result>* all_matching_results, bool use_image_order, std::vector<int>* image_order);

void print_square_organoid_images(std::vector<cv::Mat*> all_square_org_img);

void draw_cross(cv::Mat& img, const cv::Point2i& center_pos, const cv::Scalar& color, int size){

    cv::Point2i upper_left = center_pos + cv::Point2i(-size,size);
    cv::Point2i upper_right = center_pos + cv::Point2i(size,size);
    cv::Point2i lower_left = center_pos + cv::Point2i(-size,-size);
    cv::Point2i lower_right = center_pos + cv::Point2i(size,-size);

    cv::line(img,upper_left,lower_right,color);
    cv::line(img,upper_right,lower_left,color);
}

void subdivide_into_single_organoid_images(cv::Mat* organoids_img, const Organoid_Image_Header* header, bool is_single_organoid_image, unsigned int& number_offset, cv::Vec2i root_image_size, cv::Vec2i cluster_start_point){

    std::filesystem::path file_path = header->full_file_path;

    std::string file_name = file_path.filename().string();

    //Organoid_Image_Header header = parse_org_img_filename(file_name);

    remove_extension_and_channel_number(file_name);

    create_folder_structure_for_organoid_image(file_path,file_name);

    cv::Mat red_channel;// = cv::Mat(organoids_img->size(),CV_16UC3);
    cv::Mat green_channel;
    cv::Mat blue_channel;


    cv::extractChannel(*organoids_img,red_channel,2);
    cv::extractChannel(*organoids_img,green_channel,1);
    cv::extractChannel(*organoids_img,blue_channel,0);

    if(!is_single_organoid_image){
        root_image_size[0] = organoids_img->cols;
        root_image_size[1] = organoids_img->rows;

        cluster_start_point[0] = 0;
        cluster_start_point[1] = 0;
    }

    if(SHOW_INPUT_IMAGES){
        cv::namedWindow("Original Image", cv::WINDOW_KEEPRATIO );
        cv::imshow("Original Image", *organoids_img);

        cv::namedWindow("Red Channel", cv::WINDOW_KEEPRATIO );
        cv::imshow("Red Channel", red_channel);

        cv::namedWindow("Green Channel", cv::WINDOW_KEEPRATIO );
        cv::imshow("Green Channel", green_channel);

        cv::namedWindow("Blue Channel", cv::WINDOW_KEEPRATIO );
        cv::imshow("Blue Channel", blue_channel);
    }


    cv::Mat organoids_img_grayscale;

    cv::Mat threshold_img;//(organoids_img,false);
    cv::Mat adaptive_threshold_img;

    cv::Mat morph_ping_img;
    cv::Mat morph_pong_img;

    cv::Mat distance_img;

    cv::cvtColor(*organoids_img,organoids_img_grayscale,cv::COLOR_RGB2GRAY,0);

    if(organoids_img->elemSize1() == 1){
        cv::threshold(*organoids_img,threshold_img,THRESHOLD,255,cv::THRESH_BINARY);  
    }

    if(organoids_img->elemSize1() == 2){

        float conversion_scale_factor = 255.0f/65535.0f;

        organoids_img->convertTo(threshold_img,CV_8U,conversion_scale_factor);
        cv::threshold(threshold_img,threshold_img,THRESHOLD,255,cv::THRESH_BINARY);
    }

    cv::threshold(threshold_img,organoids_img_grayscale,THRESHOLD,255,cv::THRESH_BINARY);
    cv::cvtColor(organoids_img_grayscale,organoids_img_grayscale,cv::COLOR_RGB2GRAY,0);
    cv::threshold(organoids_img_grayscale,organoids_img_grayscale,THRESHOLD,255,cv::THRESH_BINARY);



    unsigned int morph_kernel_size = 7;
    unsigned int num_morph_close_iters = MORPH_NUM_CLOSING_ITERATIONS;
    unsigned int num_morph_opening_iters = MORPH_NUM_OPENING_ITERATIONS;

    if(is_single_organoid_image){
        morph_kernel_size = 5;

        num_morph_close_iters = MORPH_NUM_CLOSING_ITERATIONS_SINGLE_IMG;
        num_morph_opening_iters = MORPH_NUM_OPENING_ITERATIONS_SINGLE_IMG;
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(morph_kernel_size,morph_kernel_size));

    cv::Point center(-1,-1);




    if(MORPH_CLOSE_FIRST){
        cv::morphologyEx(organoids_img_grayscale,morph_ping_img,cv::MORPH_CLOSE,kernel,center,num_morph_close_iters);
        cv::morphologyEx(morph_ping_img,morph_pong_img,cv::MORPH_OPEN,kernel,center,num_morph_opening_iters);
    }else{
        cv::morphologyEx(organoids_img_grayscale,morph_ping_img,cv::MORPH_OPEN,kernel,center,num_morph_opening_iters);
        cv::morphologyEx(morph_ping_img,morph_pong_img,cv::MORPH_CLOSE,kernel,center,num_morph_close_iters);
    }

    if(morph_pong_img.elemSize1() > 1){
        morph_pong_img.convertTo(morph_pong_img,CV_8U);
    }

    

    cv::distanceTransform(morph_pong_img,distance_img,cv::DIST_L2,3);

    
    if(is_single_organoid_image){
        cv::normalize(distance_img,distance_img,0.0f,1.0f,cv::NORM_MINMAX);
        distance_img.convertTo(distance_img,CV_8U,255);
        cv::threshold(distance_img,distance_img,50,255,cv::THRESH_BINARY);
    }
    
    
    if(SHOW_PREPROCESSING_IMAGES){

        cv::namedWindow("Grayscale Image", cv::WINDOW_KEEPRATIO );
        cv::imshow("Grayscale Image", organoids_img_grayscale);

        cv::namedWindow("Threshold Image", cv::WINDOW_KEEPRATIO );
        cv::imshow("Threshold Image", threshold_img);

        cv::namedWindow("Morph Image", cv::WINDOW_KEEPRATIO );
        cv::imshow("Morph Image", morph_pong_img);

        cv::namedWindow("Distance Image", cv::WINDOW_KEEPRATIO );
        cv::imshow("Distance Image", distance_img);

        cv::waitKey(0);

    }



    //morph_pong_img.convertTo(distance_img,CV_8UC1);
    //cv::cvtColor(morph_pong_img,distance_img,cv::COLOR_BGR2GRAY);



    //cv::normalize(distance_img,distance_img,0.0f,1.0f,cv::NORM_MINMAX);




    cv::Mat dist_8u;

    distance_img.convertTo(dist_8u,CV_8U);

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(dist_8u,contours,cv::RETR_LIST,cv::CHAIN_APPROX_NONE);

    cv::Mat markers = cv::Mat::zeros(distance_img.size(),CV_32S);

    for(int i = 0; i < contours.size();i++){
        cv::drawContours(markers,contours,static_cast<int>(i),cv::Scalar(static_cast<int>(i+128)),-1);
        //cv::drawContours(markers,contours,static_cast<int>(i),cv::Scalar(static_cast<int>(255)),-1);
    }

    cv::Mat filled_obj_img;
    markers.convertTo(filled_obj_img,CV_8U);

    cv::Mat watershed_input_image;
    cv::cvtColor(morph_pong_img,watershed_input_image,cv::COLOR_GRAY2BGR);


    if (SEGMENT_USING_STARDIST) {

        subdivide_using_stardist(&filled_obj_img,STARDIST_DOWNSCALING, 4,organoids_img,header);


    } else {
        if (USE_PEAKS_AS_SEEDS) {
            subdivide_using_peaks(&filled_obj_img, organoids_img, header, SUBDIVIDE_SIMPLE_THRESHOLD, number_offset, is_single_organoid_image, &watershed_input_image, root_image_size, cluster_start_point);
        }
        else {
            cv::circle(markers, cv::Point(3, 3), 3, cv::Scalar(255), -1);


            cv::watershed(watershed_input_image, markers);

            cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
            finalize_organoid_images<int>(&markers, dst, contours.size(), organoids_img, header, number_offset, is_single_organoid_image, root_image_size, cluster_start_point);

            cv::Mat mark;
            markers.convertTo(mark, CV_8U);

            cv::namedWindow("Mark Debug Image", cv::WINDOW_KEEPRATIO);
            cv::imshow("Mark Debug Image", mark);

            cv::namedWindow("Final Result", cv::WINDOW_KEEPRATIO);
            cv::imshow("Final Result", dst);

        }
    }


}

bool check_red_channel_acceptance_metric(Feature_Point_Data* feature_point, float relative_distance_to_closest, float global_mean, int current_number_red_feature_points, int current_number_features_in_other_channels, bool are_candidates, float gradient_value){

    float weights_total_sum =RED_CHANNEL_DISTANCE_WEIGHT + RED_CHANNEL_NUMBER_POINTS_WEIGHT + RED_CHANNEL_PRESIENCE_WEIGHT + RED_CHANNEL_REL_PEAK_VAL_WEIGHT + RED_CHANNEL_GRADIENT_WEIGHT;

    float mininmal_distance_between_points = RED_CHANNEL_MINIMUM_SEPARATION_AS_PERCENTAGE_OF_IMAGE_SIZE; // i.e. 2% of the image size

    if(relative_distance_to_closest < mininmal_distance_between_points){
        return false;
    }
    relative_distance_to_closest -= RED_CHANNEL_SEPERATION_THRESHOLD;
    relative_distance_to_closest /= RED_CHANNEL_SEPERATION_THRESHOLD;

    if(relative_distance_to_closest > 1.0f){
        relative_distance_to_closest = 1.0f;
    }

    if(feature_point->normalize_peak_value < RED_CHANNEL_MINIMUM_PEAK_VALUE){
        return false;
    }

    float total_metric_value = 0.0;

    float presience = ((float)feature_point->peak_value / (float)feature_point->local_mean_forground_only);
    float presience_cutoff = RED_CHANNEL_PRESIENCE_CUTOFF; // i.e. the point is 50% brighter then its neighborhood

    if(presience > presience_cutoff){
        presience = presience_cutoff;
    }
    //normalize presience to [0.0,1.0]
    presience /= presience_cutoff;

    float peak_value_relative_to_global_mean = feature_point->peak_value / global_mean;
    float rel_peak_value_cutoff = RED_CHANNEL_PEAK_VALUE_CUTOFF;

    if(peak_value_relative_to_global_mean > rel_peak_value_cutoff){
        peak_value_relative_to_global_mean = rel_peak_value_cutoff;
    }
    //normalize peak_value to [0.0,1.0]
    peak_value_relative_to_global_mean /= rel_peak_value_cutoff;

    float num_features_relative_to_num_features_in_other_channels = 1.0f - (float)current_number_red_feature_points / (float)current_number_features_in_other_channels;

    total_metric_value = RED_CHANNEL_DISTANCE_WEIGHT * relative_distance_to_closest;
    total_metric_value += RED_CHANNEL_NUMBER_POINTS_WEIGHT * num_features_relative_to_num_features_in_other_channels;
    total_metric_value += RED_CHANNEL_PRESIENCE_WEIGHT * presience;
    total_metric_value += RED_CHANNEL_REL_PEAK_VAL_WEIGHT  * peak_value_relative_to_global_mean;
    total_metric_value += RED_CHANNEL_GRADIENT_WEIGHT * gradient_value;

    //normalize total_metric_value to [0.0,1.0]
    total_metric_value /= weights_total_sum;

    if(are_candidates){
        return total_metric_value  > RED_CHANNEL_CANDIDATE_ACCEPTANCE_THRESHOLD;
    }else{
        return total_metric_value  > RED_CHANNEL_FEATURE_ACCEPTANCE_THRESHOLD;
    }

}

cv::Point2i get_approximated_organoid_center(cv::Mat* mask_input){

    cv::Point2i center_point;

    uint64_t center_point_row_accumulator = 0;
    uint64_t center_point_col_accumulator = 0;
    int num_points_in_center_point_accumulator = 0;

    for(int row = 0; row < mask_input->rows; row++){
    //std::cout << "begin outer row" << std::endl;
        for(int col = 0; col < mask_input->cols; col++){
            //std::cout << "begin outer col" << std::endl;

            if(mask_input->at<uint8_t>(row,col)){
                center_point_row_accumulator += row;
                center_point_col_accumulator += col;
                num_points_in_center_point_accumulator++;
            }
        }
    }

    if(num_points_in_center_point_accumulator > 0){
        center_point.x = center_point_col_accumulator / num_points_in_center_point_accumulator;
        center_point.y = center_point_row_accumulator / num_points_in_center_point_accumulator;
    }

    return center_point;
    
}

cv::Point2i get_approximated_organoid_center_ray_tracing(cv::Mat* mask_input, int num_ray_pairs){

    cv::Point2i center_point = get_approximated_organoid_center(mask_input);

    cv::Mat debug_img = mask_input->clone();

    cv::Vec2i center_vec = center_point; 
    cv::Vec2i previous_center_vec = center_vec;

    cv::Vec2f up_dir = cv::Vec2f(0.0f,1.0f);
    float angle_increment = 180.0 / (float)num_ray_pairs;

    for(int j = 0; j < 10; j++){

        std::vector<Raylength_and_Dir> rays;

        cv::Vec2f total_displacement(0.0,0.0);

        float avg_raylength = 0.0;

        for(int i = 0; i < num_ray_pairs;i++){

            cv::Vec2f first_ray_dir = rotate_vec_2d_by_radians(up_dir,angle_increment * i);
            cv::Vec2f second_ray_dir = rotate_vec_2d_by_radians(first_ray_dir,180);

            first_ray_dir /= cv::norm(first_ray_dir);
            second_ray_dir /= cv::norm(second_ray_dir);

            //std::cout << first_ray_dir << " " << second_ray_dir << std::endl;

            cv::Vec2i end_of_first_ray = trace_line_until_organoid_boundary(mask_input,center_vec,first_ray_dir);
            cv::Vec2i end_of_second_ray = trace_line_until_organoid_boundary(mask_input,center_vec,second_ray_dir);

            double line_length_of_first_ray = cv::norm(end_of_first_ray - center_vec);
            double line_length_of_second_ray = cv::norm(end_of_second_ray - center_vec);

            Raylength_and_Dir ray_1{line_length_of_first_ray,first_ray_dir};
            Raylength_and_Dir ray_2{line_length_of_second_ray,second_ray_dir};

            avg_raylength += line_length_of_first_ray;
            avg_raylength += line_length_of_second_ray;

            rays.push_back(ray_1);
            rays.push_back(ray_2);
        }

        //std::cout << rays.size() << std::endl;

        avg_raylength /= (float)(num_ray_pairs * 2);

        double total_diff = 0.0;

        for(int i = 0; i < rays.size();i++){

            double diff_from_avg = rays[i].length - avg_raylength;

            //total_diff += fabs(diff_from_avg);

            //std::cout << diff_from_avg << std::endl;

            total_displacement += rays[i].dir * (diff_from_avg); 

            debug_img = mask_input->clone();

            /*
            cv::Scalar circle_3_colour_scalar(0, 0, 60000);
            cv::circle(debug_img, center_vec , 2, circle_3_colour_scalar);

            cv::namedWindow("Mask Debug Intermediate", cv::WINDOW_KEEPRATIO);
            cv::imshow("Mask Debug Intermediate", debug_img);

            cv::waitKey(0);
            */

        }

        //std::cout << "total diff: " << total_diff << std::endl;

        //total_displacement /= (float)(num_ray_pairs * 2);

        //center_vec += total_displacement;

        if(center_vec == previous_center_vec){
            break;
        }

        previous_center_vec = center_vec;

        std::cout << "center ray_tr: " << j << " " << center_vec << std::endl;
    }



    center_point = center_vec;

    return center_point;
    
}

cv::Point2i get_approximated_organoid_center_ray_tracing_opposites(cv::Mat* mask_input, int num_ray_pairs){

    cv::Point2i center_point = get_approximated_organoid_center(mask_input);

    cv::Vec2i center_vec = center_point; 

    cv::Vec2i previous_center_vec = center_vec;

    cv::Vec2f up_dir = cv::Vec2f(0.0f,1.0f);
    float angle_increment = 180.0 / (float)num_ray_pairs;

    for(int j = 0; j < 10; j++){

        cv::Vec2f total_displacement(0.0,0.0);

        for(int i = 0; i < num_ray_pairs;i++){

            cv::Vec2f first_ray_dir = rotate_vec_2d_by_radians(up_dir,angle_increment * i);
            cv::Vec2f second_ray_dir = rotate_vec_2d_by_radians(first_ray_dir,180);

            cv::Vec2i end_of_first_ray = trace_line_until_organoid_boundary(mask_input,center_vec,first_ray_dir);
            cv::Vec2i end_of_second_ray = trace_line_until_organoid_boundary(mask_input,center_vec,second_ray_dir);

            double line_length_of_first_ray = cv::norm(end_of_first_ray - center_vec);
            double line_length_of_second_ray = cv::norm(end_of_second_ray - center_vec);

            double length_difference = fabs(line_length_of_first_ray - line_length_of_second_ray);

            double total_length = line_length_of_first_ray + line_length_of_second_ray;

            double ratio_of_first = line_length_of_first_ray / total_length;
            double ratio_of_second = line_length_of_second_ray / total_length;

            cv::Vec2f local_displacement = first_ray_dir * ratio_of_first + second_ray_dir * ratio_of_second;

            double length_of_displacement = cv::norm(local_displacement);
            double displacement_rescale = 0.0;

            if(!check_if_doubles_are_equal(length_of_displacement,0.0)){
                displacement_rescale = (length_difference / 2.0f) / length_of_displacement;
            }


            local_displacement *= displacement_rescale;

            total_displacement += local_displacement;

        }

        total_displacement /= (double)num_ray_pairs;

        center_vec += total_displacement;

        if(center_vec == previous_center_vec){
            break;
        }

        previous_center_vec = center_vec;

        //std::cout << "center opp: " << j << " " << center_vec << std::endl;

    }

    center_point = center_vec;

    return center_point;

}


template<typename T>
void get_local_peaks(cv::Mat& input, cv::Mat* mask_input, std::vector<Feature_Point_Data>& feature_points, std::vector<Feature_Point_Data>& candidates, int local_search_dim, double& global_mean, double& global_std_dev, double& global_mean_foreground_only, double& global_std_dev_foreground_only, int& center_row, int& center_col, Channel_Type channel, int num_features_in_other_channels){

    //std::cout << "Input: " << input.elemSize() << " " << input.elemSize1() << std::endl;
    //std::cout << "image size: " << input.cols << " " << input.rows << std::endl;
    //std::cout << "mask_Input: " << mask_input->elemSize() << " " << mask_input->elemSize1() << std::endl;
    //std::cout << "mask_image size: " << mask_input->cols << " " << mask_input->rows << std::endl;

    int max_image_dim = std::max<int>(input.cols,input.rows);

    float len_image_diag = sqrt((input.cols * input.cols) + (input.rows * input.rows));

    std::priority_queue<Feature_Point_Data,std::vector<Feature_Point_Data>,Queue_Feature_Point_Compare> feature_point_queue;

    T mininmal_value_to_count_as_structure = 5; 

    global_mean = 0.0;
    global_std_dev = 0.0; 

    int t_foreground_only = 0;

    int size_of_t = sizeof(T);

    T background_colour = 0;

    cv::Vec3s background_vec(0,0,0);

    T peak_colour = (1 << (8 * size_of_t)) -1;

    cv::Vec3s foreground_vec(peak_colour,peak_colour,peak_colour);

    uint64_t center_point_row_accumulator = 0;
    uint64_t center_point_col_accumulator = 0;
    int num_points_in_center_point_accumulator = 0;

    //int counter = 0;

    //T peak_min_value = peak_colour * peak_min_intensity; 

    
    cv::Mat blurred_input;
    cv::GaussianBlur(input,blurred_input,cv::Size(5,5),0,0,cv::BORDER_DEFAULT);

    /*
    cv::Mat laplacian_output;
    cv::Laplacian(blurred_input,laplacian_output,CV_16U,3);
    cv::namedWindow("laplacian_output",cv::WINDOW_FREERATIO);
    cv::imshow("laplacian_output",laplacian_output);
    */

    //cv::namedWindow("blurred_input",cv::WINDOW_FREERATIO);
    //cv::imshow("blurred_input",blurred_input);

    float rescale_factor = ((float)(1 << 8)) / ((float)(1 << (8 * blurred_input.elemSize1())));
    blurred_input.convertTo(blurred_input,CV_8U, rescale_factor);

    /*
    cv::Mat canny_input;
    blurred_input.convertTo(canny_input,CV_8U, 1.0f/256.0f);
    cv::namedWindow("canny_input",cv::WINDOW_FREERATIO);
    cv::imshow("canny_input",canny_input);


    cv::Mat detected_edges;
    cv::Canny(canny_input,detected_edges,0,100);

    cv::namedWindow("canny",cv::WINDOW_FREERATIO);
    cv::imshow("canny",detected_edges);
    */

    cv::Mat grad_x;
    cv::Mat grad_y;

    //std::cout << "before sobel" << std::endl;

    cv::Sobel(blurred_input,grad_x,CV_16S,1,0,3);
    cv::Sobel(blurred_input,grad_y,CV_16S,0,1,3);

    cv::Mat abs_grad_x;
    cv::Mat abs_grad_y;

    cv::convertScaleAbs(grad_x,abs_grad_x);
    cv::convertScaleAbs(grad_y,abs_grad_y);

    cv::Mat grad;

    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5 ,0 , grad);

    //cv::namedWindow("grad",cv::WINDOW_FREERATIO);
    //cv::imshow("grad",grad);

    //cv::waitKey(0);
    

    for(int row = 0; row < input.rows; row++){
        //std::cout << "begin outer row" << std::endl;
        for(int col = 0; col < input.cols; col++){
            //std::cout << "begin outer col" << std::endl;

            uint64_t accumulator = 0;
            uint64_t square_accumulator = 0;
            int num_points_in_accumulator = 0;

            uint64_t accumulator_foreground = 0;
            uint64_t square_accumulator_foreground = 0;
            int num_points_in_accumulator_foreground = 0;



            bool is_peak = true;

            bool passes_presience_threshold = false;

            T center_val = input.at<T>(row,col);

            if(mask_input->at<uint8_t>(row,col)){
                center_point_row_accumulator += row;
                center_point_col_accumulator += col;
                num_points_in_center_point_accumulator++;
            }

            /*
            if(row > 180 && row < 240 && col > 160 & col < 200){

                uint16_t output_print_val = center_val;

                std::cout << "center val: " << output_print_val << std::endl;
            }
            */

            int t = row * input.cols + col;

            double prefactor_for_previous = 1.0 / (1.0 + t);

            global_mean = global_mean + prefactor_for_previous * ( center_val- global_mean);
            global_std_dev = global_std_dev + prefactor_for_previous * ( center_val*center_val- global_std_dev);

            bool other_feature_point_already_exists = check_if_existing_feature_point_in_range(row,col,center_val,local_search_dim,&feature_points);

            if(center_val > mininmal_value_to_count_as_structure && !other_feature_point_already_exists){
                //int t = row * input.cols + col;
                double prefactor_for_previous = 1.0 / (1.0 + t_foreground_only);

                global_mean_foreground_only = global_mean_foreground_only + prefactor_for_previous * ( center_val- global_mean_foreground_only);
                global_std_dev_foreground_only = global_std_dev_foreground_only + prefactor_for_previous * ( center_val*center_val- global_std_dev_foreground_only);

                t_foreground_only++;
            }else{
                continue;
            }

            /*
            if(center_val < peak_min_value){
                //output.at<T>(row,col) = background_colour;
                //output.at<cv::Vec3s>(row,col) = background_vec;

                continue;
            }
            */
            for(int local_row = -local_search_dim; local_row <= local_search_dim; local_row++){
                for(int local_col = -local_search_dim; local_col <= local_search_dim; local_col++){

                    //std::cout << "local row: " << local_row << "  local col: " << local_col << std::endl;

                    if(local_col == 0 && local_row == 0){
                        continue;
                    }
                    
                    int final_row = row + local_row;
                    int final_col = col + local_col;

                    if(final_row >= 0 && final_row < input.rows && final_col >= 0 && final_col < input.cols){
                        T current_val = input.at<T>(final_row,final_col);
                        
                        /*
                        if(current_val > center_val){
                            is_peak = false;
                            //std::cout << "break out of loop" << std::endl; 
                            break;
                        }else{
                            */
                            accumulator += current_val;
                            square_accumulator += current_val * current_val;
                            num_points_in_accumulator++;

                            if(current_val > 0){
                                accumulator_foreground += current_val;
                                square_accumulator_foreground += current_val * current_val;
                                num_points_in_accumulator_foreground++;
                            }
                            //std::cout << "accumulator: " << num_points_in_accumulator << std::endl;
                        /*    
                        }
                        */
                    }
                
                }

                if(!is_peak){
                    break;
                }

            }

            /*
            if(is_peak){
                float average_value = (float)accumulator / (float)((local_search_dim * 2 + 1) *  (local_search_dim * 2 + 1) -1);

                //std::cout << num_points_in_accumulator << std::endl;

                if(((float)center_val / average_value) >= peak_min_presience){
                    passes_presience_threshold = true;
                    //background_colour = peak_colour / 2;
                }else{
                    //background_colour = peak_colour / 2;
                    background_vec = cv::Vec3s(0,peak_colour,0);

                }

            }
            */

            if(is_peak){
                //output.at<T>(row,col) = peak_colour;
                //output.at<cv::Vec3s>(row,col) = foreground_vec;
                //counter++;
                Feature_Point_Data new_feature_point;
                new_feature_point.channel = channel;
                new_feature_point.col = col;
                new_feature_point.row = row;
                new_feature_point.peak_value = center_val;
                new_feature_point.normalize_peak_value = (double)center_val / (double)peak_colour;
                new_feature_point.local_search_dim = local_search_dim;

                new_feature_point.local_mean = (double)accumulator / (double)num_points_in_accumulator;

                double local_std_dev_squared = ((double)square_accumulator / (double)num_points_in_accumulator) - new_feature_point.local_mean * new_feature_point.local_mean;

                if(local_std_dev_squared < 0.0){
                    local_std_dev_squared = 0.0;
                }

                new_feature_point.local_std_dev = sqrt(local_std_dev_squared);

                new_feature_point.local_mean_forground_only = (double)accumulator_foreground / (double)num_points_in_accumulator_foreground;

                double local_std_dev_forground_only_squared = ((double)square_accumulator_foreground / (double)num_points_in_accumulator_foreground) - new_feature_point.local_mean_forground_only * new_feature_point.local_mean_forground_only;
                
                if(local_std_dev_forground_only_squared < 0.0){
                    local_std_dev_forground_only_squared = 0.0;
                }

                new_feature_point.local_std_dev_forground_only = sqrt(local_std_dev_forground_only_squared); 

                new_feature_point.angle = 0.0f;
                new_feature_point.relative_distance_center_boundary = 0.0f;
                new_feature_point.relative_distance_center_max_distance = 0.0f;

                feature_point_queue.push(new_feature_point);

                //feature_points.push_back(new_feature_point);
            }
        }        

    }

    global_std_dev = sqrt(global_std_dev - (global_mean * global_mean));
    global_std_dev_foreground_only = sqrt(global_std_dev_foreground_only - (global_mean_foreground_only * global_mean_foreground_only));


    //globel_std_dev = 
    //if(peak_colour == 255){

    if(num_points_in_center_point_accumulator > 0){
        center_col = center_point_col_accumulator / num_points_in_center_point_accumulator;
        center_row = center_point_row_accumulator / num_points_in_center_point_accumulator;
    }


    while(!feature_point_queue.empty()){
        Feature_Point_Data current_feature_point = feature_point_queue.top();
        feature_point_queue.pop();

        float distance_to_clostest_accepted_feature_point = 0.0;
        float distance_to_closest_accepted_candidate_point = 0.0;

        find_closest_feature_point(&current_feature_point,&feature_points,distance_to_clostest_accepted_feature_point);
        find_closest_feature_point(&current_feature_point,&candidates,distance_to_closest_accepted_candidate_point);

        float distance_relative_to_img_size = distance_to_clostest_accepted_feature_point / len_image_diag;
        float candidate_distance_relative_to_img_size = distance_to_closest_accepted_candidate_point / len_image_diag;
        //std::cout << distance_to_clostest_accepted_feature_point << " " << distance_relative_to_img_size << " " << max_image_dim << std::endl;

        uint16_t gradient_value_uint16 = grad.at<uint16_t>(current_feature_point.row,current_feature_point.col);
        float gradient_value = (float)gradient_value_uint16/(float)(1 << 16);

        if(check_red_channel_acceptance_metric(&current_feature_point,distance_relative_to_img_size, global_mean,feature_points.size(),num_features_in_other_channels,false,gradient_value)){
            //std::cout << gradient_value << std::endl;
            feature_points.push_back(current_feature_point);
        }    

        if(check_red_channel_acceptance_metric(&current_feature_point,candidate_distance_relative_to_img_size, global_mean,candidates.size(),num_features_in_other_channels,true,gradient_value)){
            candidates.push_back(current_feature_point);
        }     

    }

    //global_mean = (double)mean_accumulator / (double)(input.rows * input.cols);
    //global_mean = (double)mean_accumulator / (double)(t);


    //}

    //std::cout << counter << std::endl;
}

template<typename T>
void finalize_organoid_images(cv::Mat* input, cv::Mat& output, unsigned int num_contours, cv::Mat* original_img, const Organoid_Image_Header* header, unsigned int& number_offset, bool was_already_subdivided, cv::Vec2i root_image_size, cv::Vec2i cluster_start_point){

    //std::cout << "start finalize markers" << std::endl;

    bool show_every_output_image = SHOW_SINGLE_OUTPUT_IMAGES;

    std::vector<Cluster_Center_Point> cluster_centers;
    cluster_centers.resize(num_contours);

    for(int i = 0; i < num_contours; i++){
        cluster_centers[i].acc_row = 0;
        cluster_centers[i].acc_col = 0;
        cluster_centers[i].num_points = 0;
        cluster_centers[i].cluster_index = i+1;
        cluster_centers[i].min_row = original_img->rows;
        cluster_centers[i].max_row = 0;
        cluster_centers[i].min_col = original_img->cols;
        cluster_centers[i].max_col = 0;
    }

    for (int i = 0; i < input->rows; i++)
    {
        for (int j = 0; j < input->cols; j++)
        {
            

            T index = input->at<T>(i,j);



            if (index > 0 && index <= static_cast<T>(num_contours))
            {   
                //double scaling_factor = 4.0;

                double new_row = ((double)i / (double)input->rows);// * scaling_factor;
                double new_col = ((double)j / (double)input->cols);// * scaling_factor;


                cluster_centers[index-1].acc_row += new_row;
                cluster_centers[index-1].acc_col += new_col;

                cluster_centers[index-1].acc_row_sqr += new_row * new_row;
                cluster_centers[index-1].acc_col_sqr += new_col * new_col;

                cluster_centers[index-1].num_points++;

                int scaled_row = i;// * scaling_factor;
                int scaled_col = j;// * scaling_factor;

                if(scaled_row < cluster_centers[index-1].min_row){
                    cluster_centers[index-1].min_row = scaled_row;
                }

                if(scaled_row > cluster_centers[index-1].max_row){
                    cluster_centers[index-1].max_row = scaled_row;
                }

                if(scaled_col < cluster_centers[index-1].min_col){
                    cluster_centers[index-1].min_col = scaled_col;
                }


                if(scaled_col > cluster_centers[index-1].max_col){
                    cluster_centers[index-1].max_col = scaled_col;
                }
                //output.at<cv::Vec3b>(i,j) = colors[index-1];
            }
        }
        //std::cout << i << " " << std::endl;
    }

    int cluster_num = 0;

    for(int i = 0; i < num_contours; i++){


        if(check_single_output_image_conditions(&(cluster_centers[i]),input,root_image_size,cluster_start_point,was_already_subdivided)){
            continue;
        }

        

        int clus_ind = cluster_centers[i].cluster_index;


        double center_row = cluster_centers[i].acc_row / cluster_centers[i].num_points;
        double center_col = cluster_centers[i].acc_col / cluster_centers[i].num_points;

        double center_row_sqr = cluster_centers[i].acc_row_sqr / cluster_centers[i].num_points;
        double center_col_sqr = cluster_centers[i].acc_col_sqr / cluster_centers[i].num_points;

        int dev_row = sqrt(center_row_sqr - (center_row * center_row)) * input->rows;
        int dev_col = sqrt(center_col_sqr - (center_col * center_col)) * input->cols;

        int pixel_row = center_row * input->rows;
        int pixel_col = center_col * input->cols;

        int extent_rows_to_min = abs(pixel_row - cluster_centers[i].min_row);
        int extent_rows_to_max = abs(pixel_row - cluster_centers[i].max_row);

        int extent_cols_to_min = abs(pixel_col - cluster_centers[i].min_col);
        int extent_cols_to_max = abs(pixel_col - cluster_centers[i].max_col);
     
        int b = 255;
        int g = 255;
        int r = 255;
        cv::Vec3b white_color = cv::Vec3b((uchar)b, (uchar)g, (uchar)r);

   
        if(WRITE_OUTPUT_IMAGES){

            cv::Size2d output_size(cluster_centers[i].max_col - cluster_centers[i].min_col,cluster_centers[i].max_row - cluster_centers[i].min_row);
            //cv::Rect upper_left_quarter_size(cluster_centers[i].min_row,cluster_centers[i].max_row,cluster_centers[i].min_col,cluster_centers[i].max_col);
            cv::Rect upper_left_quarter_size(   cluster_centers[i].min_col,
                                                cluster_centers[i].min_row,
                                                cluster_centers[i].max_col-cluster_centers[i].min_col,
                                                cluster_centers[i].max_row-cluster_centers[i].min_row);
            

            cv::Mat output_image;
            cv::Mat output_mask_image(output_size,CV_8U);

            if(original_img->elemSize1() == 1){
                output_image = cv::Mat(output_size,CV_8UC3);
            }else if(original_img->elemSize1() == 2){
                output_image = cv::Mat(output_size,CV_16UC3);
                //std::cout << "output size set to CV_16UC3" << std::endl;
            }else if(original_img->elemSize1() >= 3){
                std::cout << "elemsize of original image was greater then 8 or 16 bits" << std::endl; 
            }



            //cut_and_mask_single_organoid(&cluster_centers[i],original_img,input,&output_image,false);
            cut_and_mask_single_organoid<T>(&cluster_centers[i],original_img,input,&output_image,false,output_mask_image);


            if(show_every_output_image){

                cv::Mat uncropped_output_image = (*original_img)(cv::Range(cluster_centers[i].min_row,cluster_centers[i].max_row),cv::Range(cluster_centers[i].min_col,cluster_centers[i].max_col));

                //std::cout << "in show every output image (rows,cols): (" << cluster_centers[i].min_row << "," << cluster_centers[i].max_row << ") ("<< cluster_centers[i].min_col << "," << cluster_centers[i].max_col << ")" << std::endl;

                //cv::namedWindow("test mask", cv::WINDOW_KEEPRATIO );
                //cv::imshow("test mask", test_mask);

                cv::namedWindow("Cropped Single Output", cv::WINDOW_KEEPRATIO );
                cv::imshow("Cropped Single Output", output_image);

                cv::namedWindow("Single Output", cv::WINDOW_KEEPRATIO );
                cv::imshow("Single Output", uncropped_output_image);

                cv::namedWindow("Single Output Mask", cv::WINDOW_KEEPRATIO );
                cv::imshow("Single Output Mask", output_mask_image);

                int key_code = (cv::waitKey(0) & 0xEFFFFF);

                if(key_code == 27){
                    show_every_output_image = false;
                }
            }


            if(was_already_subdivided){
                std::string sub_folder_name = "/good_images/";

                float aspect_ratio = (float)output_image.rows / (float)output_image.cols;

                if(aspect_ratio > MAXIMAL_ASPECT_RATIO_OF_OUTPUT_IMAGE || aspect_ratio < (1.0 / MAXIMAL_ASPECT_RATIO_OF_OUTPUT_IMAGE)){

                    sub_folder_name = "/high_aspect_ratio/";
                }
                
                if(cluster_centers[i].num_points < MINIMAL_PIXEL_COUNT_OF_OUTPUT_IMAGE){
                    sub_folder_name = "/low_pixel_count/";
                }

                std::string file_path = get_data_folder_path();
                file_path.append("/images");
                file_path.append("/segmented_organoids/"); 
                //file_path = "../../images/raw_organoid_images/single_organoids/";

                std::string folder_name = header->full_file_path.filename().string();
                remove_extension_and_channel_number(folder_name); 
                file_path += folder_name;
                file_path += sub_folder_name;

                std::string file_path_for_mask_image = file_path;

                std::string file_name = "single_organoid_" + std::to_string(i + number_offset) + ".tif";
                file_path += file_name;

                std::string file_name_for_mask_image = "single_organoid_" + std::to_string(i + number_offset) + "_mask.tif";
                file_path_for_mask_image += file_name_for_mask_image;

                number_offset++;

                output_image *= SINGLE_ORGANOID_OUTPUT_COLOR_RESCALE_FACTOR;
                

            }else{

                cluster_start_point[0] = cluster_centers[i].min_col;
                cluster_start_point[1] = cluster_centers[i].min_row;

                subdivide_into_single_organoid_images(&output_image,header,true,number_offset,root_image_size,cluster_start_point);
            }



            output_image.release();
        }

        cluster_num++;
    }
}

void subdivide_using_peaks(cv::Mat* filled_contours_img, cv::Mat* original_img, const Organoid_Image_Header* header, Subivision_Mode sub_mode, unsigned int& number_offset, bool was_already_subdivided, cv::Mat* watershed_input_image, cv::Vec2i root_image_size, cv::Vec2i cluster_start_point){

    //cv::namedWindow("Filled Contours Image from Peaks", cv::WINDOW_KEEPRATIO );
    //cv::imshow("Filled Contours Image from Peaks", *filled_contours_img);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(7,7));
    cv::Point center(-1,-1);

    cv::Mat smoothing_kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(7,7));


    cv::Mat distance_img;

    cv::distanceTransform(*filled_contours_img,distance_img,cv::DIST_L2,3);

    cv::normalize(distance_img,distance_img,0.0f,1.0f,cv::NORM_MINMAX);

    //cv::namedWindow("Distance of Contours Image from Peaks", cv::WINDOW_KEEPRATIO );
    //cv::imshow("Distance of Contours Image from Peaks", distance_img);

    //cv::waitKey(0);

    cv::Mat contour_distances_8bit; 
    distance_img.convertTo(contour_distances_8bit,CV_8U,255);

    
    for(int i = 0; i < 15; i++){

        cv::blur(contour_distances_8bit,contour_distances_8bit,cv::Size(15,15));
    }

    //cv::namedWindow("Distance of Contours Image from Peaks Blurred", cv::WINDOW_KEEPRATIO );
    //cv::imshow("Distance of Contours Image from Peaks Blurred", contour_distances_8bit);

    cv::Mat connected_comp_input = cv::Mat(contour_distances_8bit.size(),CV_8U);

    int threshold_in_peaks = 25;

    switch(sub_mode){

        case SUBDIVIDE_SIMPLE_THRESHOLD:


            cv::threshold(contour_distances_8bit,connected_comp_input,threshold_in_peaks,255,cv::THRESH_BINARY);
            cv::erode(connected_comp_input,connected_comp_input,kernel,center,3);

        break;

        case SUBDIVIDE_LAPLACIAN:

            cv::Laplacian(contour_distances_8bit,connected_comp_input,CV_8U,15);

            for(int i = 0; i < 5; i++){

                cv::blur(connected_comp_input,connected_comp_input,cv::Size(15,15));
            }

            cv::threshold(connected_comp_input,connected_comp_input,5,255,cv::THRESH_BINARY_INV);

            cv::dilate(connected_comp_input,connected_comp_input,kernel,center,5);

            mask_laplace_img_with_distance_img(&connected_comp_input,&contour_distances_8bit);

            //cv::namedWindow("Laplace Test", cv::WINDOW_KEEPRATIO );
            //cv::imshow("Laplace Test", connected_comp_input);

        break;


    }
    cv::Mat label_img(connected_comp_input.size(),CV_32S);

    int num_labels = cv::connectedComponents(connected_comp_input, label_img, 8);


    std::vector<cv::Vec3b> colors(num_labels);
    colors[0] = cv::Vec3b(0, 0, 0);//background
    for(int label = 1; label < num_labels; ++label){
        colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    cv::Mat coloured_label_output(label_img.size(), CV_8UC3);
    for(int r = 0; r < coloured_label_output.rows; ++r){
        for(int c = 0; c < coloured_label_output.cols; ++c){
            int label = label_img.at<int>(r, c);
            cv::Vec3b &pixel = coloured_label_output.at<cv::Vec3b>(r, c);
            pixel = colors[label];
         }
     }

    //cv::namedWindow("Connected Components from Peaks", cv::WINDOW_KEEPRATIO );
    //cv::imshow( "Connected Components from Peaks", coloured_label_output );

    cv::watershed(*watershed_input_image,label_img);


    cv::Mat dst = cv::Mat::zeros(label_img.size(), CV_8UC3);
    finalize_organoid_images<int>(&label_img,dst,num_labels,original_img,header,number_offset,was_already_subdivided,root_image_size,cluster_start_point);
    //finalize_organoid_images(markers,dst,contours.size(),*organoids_img,header);

    cv::Mat mark;
    label_img.convertTo(mark,CV_8U);

    //cv::namedWindow("Mark Debug Image from Peaks", cv::WINDOW_KEEPRATIO );
    //cv::imshow("Mark Debug Image from Peaks", mark);

    // Visualize the final image
    //cv::namedWindow("Final Result from Peaks", cv::WINDOW_KEEPRATIO );
    //cv::imshow("Final Result from Peaks", dst);



}

void subdivide_using_stardist(cv::Mat* filled_obj_img, Stardist_Scaling_Mode scaling_mode, int downscale_factor, cv::Mat* original_organoid_image,const Organoid_Image_Header* header) {

    std::vector<Feature_Point_Data> feature_points;

    cv::Mat rescaled_img;

    cv::resize(*filled_obj_img, rescaled_img, cv::Size(filled_obj_img->cols / downscale_factor, filled_obj_img->rows / downscale_factor), 0.0, 0.0, cv::INTER_AREA);

    cv::Mat label_img(rescaled_img.size(), CV_16U);

    int num_different_labels = 0;


    double nms_thresh = 0.05;

    try {
        num_different_labels = python_extract_organoid_labels(global_python_handler, &rescaled_img, &label_img, nms_thresh);
    }
    catch (const std::exception& e) 
    {
        std::cout << e.what(); 
    }


    static int img_num = 0;

    std::vector<cv::Vec3b> colors(num_different_labels);
    colors[0] = cv::Vec3b(0, 0, 0);//background
    for(int label = 1; label < num_different_labels; ++label){
        uchar blue = (rand()&255);
        uchar green = (rand()&255);
        uchar red = (rand()&255);

        if(blue < 50){
            blue = 50;
        }

        if(green < 50){
            green = 50;
        }

        if(red < 50){
            red = 50;
        }

        colors[label] = cv::Vec3b( blue, green, red );

    }

    cv::Mat coloured_label_output(label_img.size(), CV_8UC3);
    for(int r = 0; r < coloured_label_output.rows; ++r){
        for(int c = 0; c < coloured_label_output.cols; ++c){
            int label = label_img.at<uint16_t>(r, c);
            cv::Vec3b &pixel = coloured_label_output.at<cv::Vec3b>(r, c);
            pixel = colors[label];
         }
    }

    std::string output_name_str = std::to_string(img_num) + "_" + std::to_string(nms_thresh) + "_coloured_label_output.png";

    std::string output_name_str_in = std::to_string(img_num) + "_" + std::to_string(nms_thresh) + "_input.png";

    std::string output_name_str_org = std::to_string(img_num) + "_" + std::to_string(nms_thresh) + "_original.png";

    img_num++;

    cv::Mat dst = cv::Mat::zeros(original_organoid_image->size(), CV_8UC3);

    unsigned int num_found_organoids = 0;

    cv::Vec2i organoid_image_size(original_organoid_image->cols,original_organoid_image->rows);

    cv::Vec2i upper_left(0, 0);

    cv::resize(label_img, label_img, filled_obj_img->size(), 0.0, 0.0, cv::INTER_NEAREST_EXACT);

    label_img = adaptive_mask_upscale_edge_tracing<uint16_t>(&label_img);

    finalize_organoid_images<uint16_t>(&label_img, dst, num_different_labels, original_organoid_image, header, num_found_organoids, true, organoid_image_size, upper_left);

}

template<typename T> 
void cut_and_mask_single_organoid(Cluster_Center_Point* cluster, cv::Mat* original_img, cv::Mat* contour_img, cv::Mat* output_img, bool write_inverted_images, cv::Mat& output_mask_img){

    static int output_counter = 0;

    //cv::Mat mask_image(output_img->size(),CV_8U);

    cv::Mat simple_inversion;
    cv::Mat custom_inversion;

    if(write_inverted_images){
        simple_inversion = cv::Mat(output_img->size(),CV_8UC3);
        custom_inversion = cv::Mat(output_img->size(),CV_8UC3);
    }


    for(int r = cluster->min_row; r < cluster->max_row; ++r){
        for(int c = cluster->min_col; c < cluster->max_col; ++c){
            int index_from_contours = contour_img->at<T>(r, c);

            int row_in_output = r - cluster->min_row;
            int col_in_output = c - cluster->min_col;

            //std::cout << index_from_contours << " : " << cluster->cluster_index << " | ";

            if(index_from_contours == cluster->cluster_index){
                cv::Vec3s color_in_original = original_img->at<cv::Vec3s>(r, c);
                output_img->at<cv::Vec3s>(row_in_output, col_in_output) = color_in_original;
                output_mask_img.at<uint8_t>(row_in_output, col_in_output) = 255;

                if(write_inverted_images){

                float red = color_in_original[0];
                float green = color_in_original[1];
                float blue = color_in_original[2];


                cv::Vec3f red_channel = cv::Vec3f(0,-0.5f,-0.5f) * red;
                cv::Vec3f green_channel = cv::Vec3f(-0.5f,0,-0.5f) * green;
                cv::Vec3f blue_channel = cv::Vec3f(-0.5f,-0.5f,0) * blue;

                cv::Vec3f combined_channels = red_channel + green_channel + blue_channel; 

                cv::Vec3f inverted_color = cv::Vec3f(255,255,255) + combined_channels;

                cv::Vec3s test = inverted_color;

                cv::Vec3s simple_inverted_color = cv::Vec3s(255,255,255) - color_in_original;

                
                simple_inversion.at<cv::Vec3s>(row_in_output, col_in_output) = simple_inverted_color;
                custom_inversion.at<cv::Vec3s>(row_in_output, col_in_output) = inverted_color;

                }


            }else{
                output_img->at<cv::Vec3s>(row_in_output, col_in_output) = cv::Vec3s(0, 0, 0);
                output_mask_img.at<uint8_t>(row_in_output, col_in_output) = 0;

                if(write_inverted_images){
                    simple_inversion.at<cv::Vec3s>(row_in_output, col_in_output) = cv::Vec3s(255, 255, 255);
                    custom_inversion.at<cv::Vec3s>(row_in_output, col_in_output) = cv::Vec3s(255, 255, 255);
                }

                
            }
         }
    }

    if(write_inverted_images){

        cv::namedWindow("Output", cv::WINDOW_KEEPRATIO );
        cv::imshow("Output", *output_img);

        cv::namedWindow("Simple Inversion", cv::WINDOW_KEEPRATIO );
        cv::imshow("Simple Inversion", simple_inversion);

        cv::namedWindow("Custom Inversion", cv::WINDOW_KEEPRATIO );
        cv::imshow("Custom Inversion", custom_inversion);


        int key_code = (cv::waitKey(0) & 0xEFFFFF);

        if(key_code == 112){

            cv::Mat combined_images(output_img->rows,output_img->cols * 3,CV_8UC3);

            for(int r = 0; r < output_img->rows; ++r){
                for(int c = 0; c < output_img->cols; ++c){
                    combined_images.at<cv::Vec3s>(r, c) = output_img->at<cv::Vec3s>(r, c);
                    combined_images.at<cv::Vec3s>(r, c + output_img->cols) = simple_inversion.at<cv::Vec3s>(r, c);
                    combined_images.at<cv::Vec3s>(r, c + output_img->cols + output_img->cols) = custom_inversion.at<cv::Vec3s>(r, c);

                }
            }

            std::string counter_str = std::to_string(output_counter);
            counter_str += ".tif";

            std::string org_str = "Original_" + counter_str;
            std::string sim_inv_str = "Simple_Inversion_" + counter_str;
            std::string cst_inv_str = "Custom_Inversion_" + counter_str;
            std::string comp_str = "Comparison" + counter_str;

            //cv::imwrite(org_str,*output_img);
            //cv::imwrite(sim_inv_str,simple_inversion);
            //cv::imwrite(cst_inv_str,custom_inversion);
            //cv::imwrite(comp_str,combined_images);

            output_counter++;

        }

    }
}

void mask_laplace_img_with_distance_img(cv::Mat* laplace_img, cv::Mat* distance_img){


    for(int r = 0; r < laplace_img->rows; ++r){
        for(int c = 0; c < laplace_img->cols; ++c){
            uint8_t laplace_value = laplace_img->at<uint8_t>(r, c);
            uint8_t distance_value = distance_img->at<uint8_t>(r, c);

            if(distance_value > 5 && laplace_value > 0){
                laplace_img->at<uint8_t>(r, c) = 255;
            }else{
                laplace_img->at<uint8_t>(r, c) = 0;
            }
         }
     }
}

bool check_single_output_image_conditions(Cluster_Center_Point* cluster, cv::Mat* original_img, cv::Vec2i root_image_size, cv::Vec2i cluster_start_point, bool was_already_subdivided){

    //std::cout << "in check condition: orig img: (" << original_img->rows << "," << original_img->cols << ")" << std::endl;

    //std::cout << "in check condition: orig img: (" << original_img->rows << "," << original_img->cols << ") cluster max-min (" << cluster->max_row << "," << cluster->max_col << ")  (" << cluster->min_row << "," << cluster->min_col << ")"  << std::endl;

    int cluster_boundary_offset_min = 2;
    int cluster_boundary_offset_max = 4;
    // clusters can have a min value of 1 and a max value of (row-2) and vice versa for cols
    // if was_already_subdivided = true then we need to set the boundary_offset_min to 2 since as stated above the min value in each cluster is 1
    // but since we now basically have a cluster in a cluster we have this minimal 1 twice
    // the boundary_offset_max is analogously 4 (because we take the row-2 twice)



    if(cluster->num_points == 0){
        return true;
    }
    if(!was_already_subdivided){
        return false;
    }
    if(cluster->min_col + cluster_start_point[0] <= cluster_boundary_offset_min){
        //std::cout << "in check condition: min col " << std::endl;
        return true;
    }
    if(cluster->min_row + cluster_start_point[1] <= cluster_boundary_offset_min){
        //std::cout << "in check condition: min row " << std::endl;
        return true;
    }
    if(cluster->max_row + cluster_start_point[1] >= root_image_size[1] - cluster_boundary_offset_max){
        //std::cout << cluster->max_row << " " << cluster_start_point[1] << " " << root_image_size[1] << " " << cluster_boundary_offset_max;
        //std::cout << "in check condition: max row " << std::endl;
        return true;
    }
    if(cluster->max_col + cluster_start_point[0] >= root_image_size[0] - cluster_boundary_offset_max){
        //std::cout << "in check condition: max col " << std::endl;
        return true;
    }
    if(cluster->max_col + cluster_start_point[0] >= (root_image_size[0] - cluster_boundary_offset_max) / 2 && cluster->min_col + cluster_start_point[0] <= (root_image_size[0] - cluster_boundary_offset_min) / 2){
        //std::cout << "in check condition: col middle " << std::endl;
        return true;
    }
    if(cluster->max_row + cluster_start_point[1] >= (root_image_size[1] - cluster_boundary_offset_max) / 2 && cluster->min_row + cluster_start_point[1] <= (root_image_size[1] - cluster_boundary_offset_min) / 2){
        //std::cout << "in check condition: row middle " << std::endl;
        return true;
    }

    return false;


}

//this function populated the output_feature_points array with the detected features points, stores the organoid image in the mat_to_store_image_in variable and returns the center point of the organoid
cv::Point2i extract_features_from_single_organoid_image(std::filesystem::path single_organoid_img_path, std::vector<Feature_Point_Data>& output_feature_points, std::vector<Feature_Point_Data>& output_candidate_points, cv::Mat* mat_to_store_image_in, bool read_features_from_file, std::filesystem::path data_file_path, int img_number_offset){

    std::string file_name = single_organoid_img_path.filename().string();

    int image_number = get_image_number_from_file_name(file_name);

    cv::Point2i center_of_organoid(-1,-1);

    static int window_number = 1;

    if (!std::filesystem::exists(single_organoid_img_path)) {
        std::cout << "WARNING: " << single_organoid_img_path.string() << "  does not exist!" << std::endl;
        return center_of_organoid;
    }
    std::string img_path_string = single_organoid_img_path.string();

    cv::Mat organoids_img = cv::imread(img_path_string,cv::IMREAD_UNCHANGED);

    if(mat_to_store_image_in != nullptr){
        *mat_to_store_image_in = organoids_img; 
    }

    std::string mask_img_path_string = img_path_string;

    remove_extension(mask_img_path_string);

    mask_img_path_string += "_mask.tif";

    cv::Mat organoid_mask_image = cv::imread(mask_img_path_string,cv::IMREAD_UNCHANGED);


    if(organoid_mask_image.channels() == 4){
        cv::cvtColor(organoid_mask_image,organoid_mask_image,cv::COLOR_RGBA2GRAY);
    }

    if(read_features_from_file){
        //All_Feature_Point_Selection_Parameters feature_selection_params;

        read_feature_vector_from_file(data_file_path,image_number + img_number_offset,output_feature_points,output_candidate_points);

        return get_approximated_organoid_center(&organoid_mask_image);
    }



    bool connect_verticies = false;

    cv::Mat red_channel;
    cv::Mat green_channel;
    cv::Mat blue_channel;

    FEATURE_EXTRACTION_DEBUG(
        std::cout << "Mask Elem size: " << organoid_mask_image.elemSize() << " Elem size 1: " << organoid_mask_image.elemSize1() << "  channels: " << organoid_mask_image.channels() << std::endl;
        std::cout << "Org Elem size: " << organoids_img.elemSize() << " Elem size 1: " << organoids_img.elemSize1() << "  channels: " << organoids_img.channels() << std::endl;
    )

    bool was_converted_from_8bit = false;

    cv::Mat originial_img_8bit_red;

    if (organoids_img.elemSize1() == 1) {
        cv::extractChannel(organoids_img,originial_img_8bit_red,2);
        organoids_img.convertTo(organoids_img, CV_16U,(float)(1<<16) / (float)(1<<8));
        was_converted_from_8bit = true;
        std::cout << "converted to CV_16U" << std::endl;
        std::cout << originial_img_8bit_red.elemSize1() << " " << originial_img_8bit_red.elemSize() << std::endl; 
    }

    cv::extractChannel(organoids_img,blue_channel,0);
    cv::extractChannel(organoids_img,green_channel,1);
    cv::extractChannel(organoids_img,red_channel,2);

    bool use_nuclei_as_feature_points = false;

    cv::Mat target_channel;

    if (use_nuclei_as_feature_points) {
        target_channel = blue_channel.clone();
    } else {
        target_channel = red_channel.clone();
    }

    cv::Mat laplacian_output;

    cv::Laplacian(target_channel,laplacian_output,CV_16U,5);


    FEATURE_EXTRACTION_DEBUG(
        cv::namedWindow("Organoid Image", cv::WINDOW_KEEPRATIO );
        cv::imshow("Organoid Image", organoids_img);


        cv::namedWindow("Red Channel", cv::WINDOW_KEEPRATIO );
        cv::imshow("Red Channel", red_channel);

        cv::namedWindow("Green Channel", cv::WINDOW_KEEPRATIO );
        cv::imshow("Green Channel", green_channel);

        cv::namedWindow("Blue Channel", cv::WINDOW_KEEPRATIO );
        cv::imshow("Blue Channel", blue_channel);

        cv::namedWindow("Organoid Mask Image", cv::WINDOW_KEEPRATIO );
        cv::imshow("Organoid Mask Image", organoid_mask_image);


        cv::namedWindow("Laplacian", cv::WINDOW_KEEPRATIO );
        cv::imshow("Laplacian", laplacian_output);

        cv::waitKey(0);
    )

    cv::Mat local_peaks_img(laplacian_output.size(),CV_16UC3);

    int search_dim = calculate_local_search_dim_from_image_size(organoids_img.size(),RED_CHANNEL_LOCAL_SEARCH_DIM_AS_PERCENTAGE_OF_IMAGE_SIZE);

    double presience_threshold = 1.5;
    double nms_threshhold = NUCLEUS_OVERLAP_THRESHOLD;

    std::vector<Feature_Point_Data> red_feature_points;
    std::vector<Feature_Point_Data> red_candidate_points;
    std::vector<Feature_Point_Data> blue_feature_points;
    std::vector<Feature_Point_Data> green_feature_points;

    while(true){

        cv::Mat mean, std_dev;

        cv::meanStdDev(red_channel,mean,std_dev);

        double global_mean = 0.0f;
        double global_std_dev = 0.0f;

        double global_mean_foreground_only = 0.0f;
        double global_std_dev_foreground_only = 0.0f;

        cv::Mat feature_point_image_green_channel;
        cv::Mat feature_point_image_blue_channel;
        cv::Mat feature_point_image_red_channel;
        cv::Mat feature_point_image_red_candidate_channel;
        cv::Mat graph_image;

        cv::cvtColor(green_channel, feature_point_image_green_channel, cv::COLOR_GRAY2RGB);
        cv::cvtColor(blue_channel, feature_point_image_blue_channel, cv::COLOR_GRAY2RGB);
        cv::cvtColor(red_channel, feature_point_image_red_channel, cv::COLOR_GRAY2RGB);
        cv::cvtColor(red_channel, feature_point_image_red_candidate_channel, cv::COLOR_GRAY2RGB);
        cv::cvtColor(target_channel, graph_image, cv::COLOR_GRAY2RGB);

        int center_col = 0;
        int center_row = 0;

        red_candidate_points.clear();
        red_feature_points.clear();
        blue_feature_points.clear();
        green_feature_points.clear();

        python_extract_feature_points(global_python_handler, &green_channel, green_feature_points, nms_threshhold,PDX1_GFP_CHANNEL);
        python_extract_feature_points(global_python_handler, &blue_channel, blue_feature_points, nms_threshhold,DAPI_CHANNEL);

        int num_features_in_green_and_blue_channels = green_feature_points.size() + blue_feature_points.size();

        // for now we get the center point of the organoid image in the get_local_peaks function since we iterate over the entire image in this function anyways.
        // in case this changes the calculation for the center point needs to be done separately
        if(was_converted_from_8bit){
            get_local_peaks<uint8_t>(originial_img_8bit_red,&organoid_mask_image, red_feature_points, red_candidate_points, search_dim, global_mean, global_std_dev, global_mean_foreground_only, global_std_dev_foreground_only,center_row,center_col,Phalloidon_AF647_CHANNEL,num_features_in_green_and_blue_channels);
            //get_local_peaks<uint8_t>(originial_img_8bit_red,&organoid_mask_image, red_candidate_points, search_dim, global_mean, global_std_dev, global_mean_foreground_only, global_std_dev_foreground_only,center_row,center_col,Phalloidon_AF647_CHANNEL,num_features_in_green_and_blue_channels);
        }else{
            get_local_peaks<uint16_t>(red_channel,&organoid_mask_image, red_feature_points,red_candidate_points, search_dim, global_mean, global_std_dev, global_mean_foreground_only, global_std_dev_foreground_only,center_row,center_col,Phalloidon_AF647_CHANNEL,num_features_in_green_and_blue_channels);
            //get_local_peaks<uint16_t>(red_channel,&organoid_mask_image, red_candidate_points, search_dim, global_mean, global_std_dev, global_mean_foreground_only, global_std_dev_foreground_only,center_row,center_col,Phalloidon_AF647_CHANNEL,num_features_in_green_and_blue_channels);
        }

        cv::Mat mask_debug;// = organoid_mask_image;

        cv::cvtColor(organoid_mask_image,mask_debug,cv::COLOR_GRAY2BGR);

        center_of_organoid.x = center_col;
        center_of_organoid.y = center_row;


        calculate_spherical_feature_point_coordinates(&red_feature_points,&organoid_mask_image,feature_point_image_red_channel,center_row,center_col);
        put_feature_points_on_image(feature_point_image_red_channel,&red_feature_points,global_mean, global_std_dev,presience_threshold);

        calculate_spherical_feature_point_coordinates(&red_candidate_points,&organoid_mask_image,feature_point_image_red_candidate_channel,center_row,center_col);
        put_feature_points_on_image(feature_point_image_red_candidate_channel,&red_candidate_points,global_mean, global_std_dev,presience_threshold);
        
        calculate_spherical_feature_point_coordinates(&green_feature_points,&organoid_mask_image,feature_point_image_green_channel,center_row,center_col);
        put_feature_points_on_image(feature_point_image_green_channel,&green_feature_points,global_mean, global_std_dev,presience_threshold);

        calculate_spherical_feature_point_coordinates(&blue_feature_points,&organoid_mask_image,feature_point_image_blue_channel,center_row,center_col);
        put_feature_points_on_image(feature_point_image_blue_channel,&blue_feature_points,global_mean, global_std_dev,presience_threshold);
        
        // sort the feature points based on their angle
        // this will make calculating the angles between pairs a bit easier
        std::sort(green_feature_points.begin(),green_feature_points.end(),sort_by_angle_compare_function);
        std::sort(blue_feature_points.begin(),blue_feature_points.end(),sort_by_angle_compare_function);
        std::sort(red_feature_points.begin(),red_feature_points.end(),sort_by_angle_compare_function);
        std::sort(red_candidate_points.begin(),red_candidate_points.end(),sort_by_angle_compare_function);


        if(connect_verticies){
            connect_all_feature_points(&red_feature_points, graph_image);
            put_feature_points_on_image(graph_image, &red_feature_points,global_mean,global_std_dev,presience_threshold);

            FEATURE_EXTRACTION_DEBUG(
                cv::namedWindow("Connected Feature Points", cv::WINDOW_KEEPRATIO);
                cv::imshow("Connected Feature Points", graph_image);
            )
        }



        FEATURE_EXTRACTION_OUTPUT(

            cv::Rect window_image_size_rect;

            cv::namedWindow("Organoid Window", cv::WINDOW_KEEPRATIO );
            cv::imshow("Organoid Window", organoids_img);

            int pixel_spacing_between_windows = 100;
            int pixel_spacing_between_rows_of_windows = 0;

            std::string green_window_name = "Feature Points Green Channel " + std::to_string(window_number);
            std::string blue_window_name = "Feature Points Blue Channel " + std::to_string(window_number);
            std::string red_window_name = "Feature Points Red Channel " + std::to_string(window_number);

            std::string red_candidate_window_name = "Candidate Points Red Channel " + std::to_string(window_number);

            cv::namedWindow(green_window_name, cv::WINDOW_KEEPRATIO );
            cv::imshow(green_window_name, feature_point_image_green_channel);
            window_image_size_rect = cv::getWindowImageRect(green_window_name);

            int row_distance = (window_number - 1) * (window_image_size_rect.height + pixel_spacing_between_rows_of_windows) + pixel_spacing_between_windows;

            cv::namedWindow(blue_window_name, cv::WINDOW_KEEPRATIO );
            cv::imshow(blue_window_name, feature_point_image_blue_channel);

            cv::namedWindow(red_window_name, cv::WINDOW_KEEPRATIO );
            cv::imshow(red_window_name, feature_point_image_red_channel);

            cv::namedWindow(red_candidate_window_name, cv::WINDOW_KEEPRATIO );
            cv::imshow(red_candidate_window_name, feature_point_image_red_candidate_channel);

            int key_code = 0;

            key_code = (cv::waitKey(0) & 0xEFFFFF );

            //std::cout << "keycode: " << key_code << std::endl;

            if(key_code == 32){
                search_dim++;
            }

            if(key_code == 101){
                search_dim--;
                //std::cout << presience_threshold << std::endl;
            }

            if(key_code == 27){
                break;
            }

            if(key_code == 113){
                presience_threshold += 0.1;
                std::cout << "presience_threshold: " << presience_threshold << std::endl;
            }

            if(key_code == 119){
                presience_threshold -= 0.1;
                std::cout << "presience_threshold: " << presience_threshold << std::endl;
            }

            if(key_code == 110){
                nms_threshhold += 0.1;
                std::cout << "nms_threshold: " << nms_threshhold << std::endl;
            }

            if(key_code == 109){
                nms_threshhold -= 0.1;
                std::cout << "nms_threshold: " << nms_threshhold << std::endl;
            }

        )else{
            break;
        }

        feature_point_image_green_channel.release();
        feature_point_image_blue_channel.release();
        feature_point_image_red_channel.release();
        feature_point_image_red_candidate_channel.release();
   
    }

    join_feature_vectors(output_feature_points,&red_feature_points,&blue_feature_points,&green_feature_points);
    join_feature_vectors(output_candidate_points,&red_candidate_points,&blue_feature_points,&green_feature_points);

    if(output_feature_points.size() == 261 || output_feature_points.size() == 224){
        std::cout << image_number << std::endl;
    }


    if(output_candidate_points.size() == 152 || output_candidate_points.size() == 381){
        std::cout << image_number << std::endl;
    }
    // sort the feature points based on their angle
    // this will make calculating the angles between pairs a bit easier
    std::sort(output_feature_points.begin(),output_feature_points.end(),sort_by_angle_compare_function);
    std::sort(output_candidate_points.begin(),output_candidate_points.end(),sort_by_angle_compare_function);


    return center_of_organoid;
}

void calculate_spherical_feature_point_coordinates(std::vector<Feature_Point_Data>* feature_points, cv::Mat* mask_image, cv::Mat& output_image, int center_row, int center_col){

    cv::Vec2i center(center_col,center_row);
    //center = cv::Vec2i(0,0);
    //cv::Point c_point = center;

    //int subject = 14;

    cv::Vec2f up_dir(0.0f,-1.0f);
    //cv::Vec2f test_dir(0.0f,-1.0f);
    //double angle_between = (acos(up_dir.ddot(test_dir)) / PI) * 180.0;
    //std::cout << "angle: " << angle_between << std::endl;

    double longest_line_length = 0.0; 

    for(int i = 0; i < feature_points->size(); i++){

        //std::cout << "iteration: " << i << " of: " << feature_points->size() << std::endl; 

        Feature_Point_Data current_feature_point = (*feature_points)[i];
        cv::Vec2i feature_point_pos(current_feature_point.col,current_feature_point.row);

        cv::Vec2f dir_to_feature_point = feature_point_pos - center;
        double length_to_feature_point = cv::norm(dir_to_feature_point);

        //std::cout << "pos: " << feature_point_pos << " dir: " << dir_to_feature_point << " center: " << center <<" len: " << length_to_feature_point << std::endl;


        dir_to_feature_point = cv::normalize(dir_to_feature_point);
        //double angle_between_lines = (acos(up_dir.ddot(dir_to_feature_point)) / PI) * 180.0;

        double atan_angle = (atan2(dir_to_feature_point[0],dir_to_feature_point[1]) / PI) * 180.0 + 180.0;

        /*
        if(dir_to_feature_point[0] > 0.0f){
            angle_between_lines = 360.0 - angle_between_lines;
        }

        std::cout << angle_between_lines << " " << atan_angle << std::endl;
        */

       //std::cout << i << std::endl;

        cv::Vec2i end = trace_line_until_organoid_boundary(mask_image,feature_point_pos,dir_to_feature_point);

        double total_line_length = cv::norm(end - center);

        if(total_line_length > longest_line_length){
            longest_line_length = total_line_length;
        }

        double length_ratio = length_to_feature_point / total_line_length;

        (*feature_points)[i].angle = atan_angle;//angle_between_lines;
        (*feature_points)[i].relative_distance_center_boundary = length_ratio;
        (*feature_points)[i].relative_distance_center_max_distance = length_to_feature_point;


        //cv::line(output_image,center,end,cv::Scalar(65000.0 * (1.0 - (angle_between_lines/360.0)),0.0,65000.0 * (angle_between_lines/360.0)));

        //cv::line(output_image,feature_point_pos,end,cv::Scalar(35000.0,35000.0,35000.0),cv::LINE_AA);
        //cv::line(output_image,center,feature_point_pos,cv::Scalar(65000.0 * (1.0 - (atan_angle/360.0)),0.0,65000.0 * (atan_angle/360.0)), cv::LINE_AA);

    }

    for(int i = 0; i < feature_points->size(); i++){
        //std::cout << "distance before: " << (*feature_points)[i].relative_distance_center_max_distance << std::endl;
        (*feature_points)[i].relative_distance_center_max_distance = (*feature_points)[i].relative_distance_center_max_distance / longest_line_length;
        //std::cout << "distance after: " << (*feature_points)[i].relative_distance_center_max_distance << std::endl; 
    }


}

cv::Vec2i trace_line_until_organoid_boundary(cv::Mat* mask_image, cv::Vec2i start, cv::Vec2f dir){

    int x_sign = get_sign<float>(dir[0]);
    int y_sign = get_sign<float>(dir[1]);

    if(fabs(dir[0] - 0.0f) < FLT_EPSILON && fabs(dir[1] - 0.0f) < FLT_EPSILON){
        return start;
    }

    cv::Vec2f current_pos = start;

    cv::Vec2f next = current_pos + cv::Vec2f(x_sign,y_sign);

    cv::Vec2i int_pos = start;

    while(mask_image->at<uint8_t>(int_pos[1],int_pos[0])){

        cv::Vec2f diff = next - current_pos;
        float t_x = (float)diff[0] / dir[0];

        if(fabs(dir[0] - 0.0f) < FLT_EPSILON){
            t_x = FLT_MAX;
        }

        float t_y = (float)diff[1] / dir[1];

        if(fabs(dir[1] - 0.0f) < FLT_EPSILON){
            t_y = FLT_MAX;
        }

        if(t_x < t_y){
            next[0] += x_sign;
            current_pos += t_x * dir;

        }else{
            next[1] += y_sign;
            current_pos += t_y * dir;
        }

        int new_col = floor(current_pos[0]);
        int new_row = floor(current_pos[1]);

        if (new_col < 0 || new_row < 0 || new_row >= mask_image->rows || new_col >= mask_image->cols) {
            break;
        }

        int_pos[0] = new_col;
        int_pos[1] = new_row;

    }

    return int_pos;
}

void put_feature_points_on_image(cv::Mat& output_image, std::vector<Feature_Point_Data>* feature_points, double global_mean, double global_std_dev, double presience_threshold){

    uint16_t peak_colour = (1 << (8 * 2)) -1;

    for(int i = 0; i < feature_points->size(); i++){
        cv::Vec3s peak_colour_vec(peak_colour, peak_colour, peak_colour);

        cv::Scalar peak_colour_scalar(peak_colour, peak_colour, peak_colour);

        Feature_Point_Data current_feature_point = (*feature_points)[i];

        //Feature_Point_Data next_feature_point = find_closest_feature_point(&current_feature_point,feature_points);

        //cv::Point2i start(current_feature_point.col,current_feature_point.row);
        //cv::Point2i end(next_feature_point.col,next_feature_point.row);

        //cv::Scalar line_colour(0,peak_colour,0);

        //cv::line(output_image,start,end,line_colour,1,cv::LINE_AA);

        //std::cout << "peak_val: " << current_feature_point.peak_value << "  mean: " << global_mean << std::endl;

        if (current_feature_point.peak_value < global_mean) {
            peak_colour_vec -= cv::Vec3s(0, 0, peak_colour);
            peak_colour_scalar -= cv::Scalar(0, 0, peak_colour);
            //std::cout << current_feature_point.col << " " << current_feature_point.row << std::endl;
        }
        
        //std::cout << current_feature_point.local_mean << "  " << current_feature_point.local_mean_forground_only << std::endl;

        if(((float)current_feature_point.peak_value / (float)current_feature_point.local_mean_forground_only) < presience_threshold){
            //peak_colour_vec -= cv::Vec3s(peak_colour,0,0);
            //peak_colour_scalar -= cv::Scalar(peak_colour, 0, 0);
        }
        

       if(current_feature_point.peak_value < global_mean + global_std_dev){
            //peak_colour_vec -= cv::Vec3s(0,peak_colour,0);
            //peak_colour_scalar -= cv::Scalar(0,peak_colour, 0);
        }

        if(peak_colour_scalar == cv::Scalar(0,0,0)){
            uint16_t half_peak_colour = peak_colour>>1;
            //std::cout << "half peak colour: " << half_peak_colour << std::endl;
            peak_colour_scalar = cv::Scalar(half_peak_colour,half_peak_colour,half_peak_colour);
        }

        //std::cout << peak_colour_scalar << std::endl;

        //peak_colour_scalar = cv::Scalar(0, peak_colour * current_feature_point.relative_distance_center_max_distance, peak_colour * (1.0 - current_feature_point.relative_distance_center_max_distance));

        peak_colour_scalar = cv::Scalar(0, 0, peak_colour);

        cv::Point2i center(current_feature_point.col, current_feature_point.row);
        cv::circle(output_image, center, 1, peak_colour_scalar,-1);
        //output_image.at<cv::Vec3s>(current_feature_point.row,current_feature_point.col) = peak_colour_vec;

    }


}

Feature_Point_Data find_closest_feature_point(Feature_Point_Data* reference_point, std::vector<Feature_Point_Data>* feature_points, float& distance_to_closest){

    /*
    if(feature_points->size() == 1){
        distance_to_closest = 0.0f;
        return (*feature_points)[0];
    }
    */
    cv::Point2i refernence_pos(reference_point->col,reference_point->row);

    Feature_Point_Data closest_feature_point;
    closest_feature_point = *reference_point;
    float shortest_distance = FLT_MAX;

    for(int i = 0; i < feature_points->size();i++){
        Feature_Point_Data new_feature_point = (*feature_points)[i];

        cv::Point2i new_feature_point_pos(new_feature_point.col,new_feature_point.row);

        cv::Point2i distance_vec = refernence_pos - new_feature_point_pos;
        
        if(distance_vec.x == 0 && distance_vec.y == 0){
            distance_to_closest = 0.0f;

            return new_feature_point;
 

        }else{
            float new_distance = sqrtf( (float)(distance_vec.x * distance_vec.x) + (float)(distance_vec.y * distance_vec.y));

            if(new_distance < shortest_distance){

                shortest_distance = new_distance;
                closest_feature_point = new_feature_point;
            }


        }

    }

    distance_to_closest = shortest_distance;
    return closest_feature_point;

}

Feature_Point_Data find_closest_feature_point_and_set_id(Feature_Point_Data* reference_point,int reference_point_id ,std::vector<Feature_Point_Data>* feature_points, float& distance_to_closest, int* set_ids, int& closest_feature_point_id, int start_id, bool& found_new_closest){

    if(feature_points->size() == 1){
        distance_to_closest = 0.0f;
        return (*feature_points)[0];
    }
    
    cv::Point2i refernence_pos(reference_point->col,reference_point->row);

    Feature_Point_Data closest_feature_point;
    closest_feature_point = *reference_point;
    float shortest_distance = distance_to_closest;

    for(int i = start_id; i < feature_points->size();i++){
        Feature_Point_Data new_feature_point = (*feature_points)[i];

        int new_feature_point_id = set_ids[i];

        if(new_feature_point_id == reference_point_id){
            continue;
        }

        cv::Point2i new_feature_point_pos(new_feature_point.col,new_feature_point.row);

        cv::Point2i distance_vec = refernence_pos - new_feature_point_pos;
        

        float new_distance = sqrtf( (float)(distance_vec.x * distance_vec.x) + (float)(distance_vec.y * distance_vec.y));

        if(new_distance < shortest_distance){
            found_new_closest = true;
            shortest_distance = new_distance;
            closest_feature_point = new_feature_point;
            closest_feature_point_id = i;
        }
    }


    //set_ids[closest_feature_point_id] = reference_point_id;
    distance_to_closest = shortest_distance;
    return closest_feature_point;
}

void connect_all_feature_points(std::vector<Feature_Point_Data>* feature_points, cv::Mat& output_image){

    int* set_ids = (int*)malloc(sizeof(int) * feature_points->size());

    for(int i = 0; i < feature_points->size();i++){
        set_ids[i] = i;
    }

    bool found_at_least_one_closest = true;
    

    while(found_at_least_one_closest){
        found_at_least_one_closest = false;

        float distance_to_closest = FLT_MAX;

        
        int reference_feature_point_index = 0;
        int closest_feature_point_id;

        for(int i = 0; i < feature_points->size();i++){
            Feature_Point_Data reference_feature_point = (*feature_points)[i];

            int reference_point_id = set_ids[i];

            bool found_new_closest = false;

            Feature_Point_Data closest_feature_point = find_closest_feature_point_and_set_id(   &reference_feature_point,
                                                                                                reference_point_id,
                                                                                                feature_points,
                                                                                                distance_to_closest,
                                                                                                set_ids,
                                                                                                closest_feature_point_id,
                                                                                                i+1,
                                                                                                found_new_closest);

            if(found_new_closest){
                reference_feature_point_index = i;
                found_at_least_one_closest = true;
            }

        }

        //std::cout << closest_feature_point_id << " " << reference_feature_point_index << std::endl;

        //go through set ids and set all that are equal to set_ids
        if (found_at_least_one_closest) {
            int id_to_replace = set_ids[closest_feature_point_id];

            for (int k = 0; k < feature_points->size(); k++) {

                if (set_ids[k] == id_to_replace) {
                    set_ids[k] = set_ids[reference_feature_point_index];
                }

            }

            Feature_Point_Data current_feature_point = (*feature_points)[reference_feature_point_index];
            Feature_Point_Data next_feature_point = (*feature_points)[closest_feature_point_id];

            cv::Point2i start(current_feature_point.col, current_feature_point.row);
            cv::Point2i end(next_feature_point.col, next_feature_point.row);

            cv::Scalar line_colour(0, 30000, 0);

            cv::line(output_image, start, end, line_colour, 1, cv::LINE_AA);

        }


    }


    free(set_ids);

}

bool check_if_existing_feature_point_in_range(int row, int col, int center_value, int range, std::vector<Feature_Point_Data>* existing_feature_points){

    for(int i = 0; i < existing_feature_points->size(); i++){

        Feature_Point_Data current_feature_point = (*existing_feature_points)[i];

        if(abs(current_feature_point.col - col) < range && abs(current_feature_point.row - row) < range && center_value <= current_feature_point.peak_value){
            return true;
        }

    }

    return false;

}

void join_feature_vectors(std::vector<Feature_Point_Data>& output_feature_vector, std::vector<Feature_Point_Data>* red_feature_vector,std::vector<Feature_Point_Data>* blue_feature_vector,std::vector<Feature_Point_Data>* green_feature_vector){

    output_feature_vector.clear();


    for(int i = 0; i < blue_feature_vector->size(); i++){

        Feature_Point_Data current_feature_point = (*blue_feature_vector)[i]; 

        output_feature_vector.push_back(current_feature_point);
    }

    for(int i = 0; i < green_feature_vector->size(); i++){

        Feature_Point_Data current_feature_point = (*green_feature_vector)[i]; 

        output_feature_vector.push_back(current_feature_point);
    }

    for(int i = 0; i < red_feature_vector->size(); i++){

        Feature_Point_Data current_feature_point = (*red_feature_vector)[i]; 

        output_feature_vector.push_back(current_feature_point);
    }

}

void show_matching_matrix(std::vector<cv::Mat*>* square_organoid_images, std::vector<Matching_Result>* all_matching_results,std::vector<Image_Features_Pair>* all_feature_vectors, int individual_image_size, float cost_threshold, Matching_Visualization_Type viz_type, uint32_t flags, std::vector<int>* image_order, bool use_image_order, All_Cost_Parameters& cost_params){

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    static bool draw_organoid_images = true;
    static bool already_allocated_image = false;
    static cv::Mat comp_img;

    static bool use_image_order_in_last_call = false;

    static bool costs_from_matching_results_was_initialized = false;
    static std::vector<double> costs_from_matching_results;


    //std::cout << "symmetrize bit: " << uint32_t_query_single_bit(&global_matching_matrix_mouse_callback_input.flags,MATCHING_MATRIX_SYMMETRIZE_BIT_INDEX) << std::endl;

    //std::cout << "before check: " << use_image_order << std::endl;

    if(image_order->size() != all_feature_vectors->size()){
        use_image_order = false;
    }

    if(!costs_from_matching_results_was_initialized || (use_image_order_in_last_call != use_image_order)){
        //std::cout << "init" << std::endl;
        fill_costs_from_matching_results_vector(costs_from_matching_results,all_feature_vectors,all_matching_results,use_image_order,image_order);
        costs_from_matching_results_was_initialized = true;
    }
    //std::cout << "after check: " << use_image_order << std::endl;

    static bool selected_cell_was_initialized = false;

    if(!selected_cell_was_initialized){
        selected_cell_in_matching_matrix.x = -1;
        selected_cell_in_matching_matrix.y = -1;
        selected_cell_in_matching_matrix.pixel_per_cell = individual_image_size;
        selected_cell_was_initialized = true;
    }

    global_matching_matrix_mouse_callback_input.all_feature_vectors = all_feature_vectors;
    global_matching_matrix_mouse_callback_input.all_matching_results = all_matching_results;
    global_matching_matrix_mouse_callback_input.cost_threshold = cost_threshold;
    global_matching_matrix_mouse_callback_input.individual_image_size = individual_image_size;
    global_matching_matrix_mouse_callback_input.selected_cell = &selected_cell_in_matching_matrix;
    global_matching_matrix_mouse_callback_input.square_organoid_images = square_organoid_images;
    global_matching_matrix_mouse_callback_input.viz_type = viz_type;
    global_matching_matrix_mouse_callback_input.flags = flags;
    global_matching_matrix_mouse_callback_input.image_order = image_order;
    global_matching_matrix_mouse_callback_input.cost_params = cost_params;

    //std::cout << "mci: " << mouse_callback_input.use_image_order << " image_order: " << use_image_order << std::endl;

    if(global_matching_matrix_mouse_callback_input.use_image_order != use_image_order){
        //std::cout << "DRAW ORGANOIDS" << std::endl;
        draw_organoid_images = true;
    }

    global_matching_matrix_mouse_callback_input.use_image_order = use_image_order;

    use_image_order_in_last_call = use_image_order;


    /*
    if(uint32_t_query_single_bit(&mouse_callback_input.flags,MATCHING_MATRIX_COLORGRADIENT_BIT_INDEX)){
        std::cout << "colorgradient bit is set" << std::endl;
    }else{
        std::cout << "colorgradient bit is NOT set" << std::endl;
    }

    if(uint32_t_query_single_bit(&mouse_callback_input.flags,MATCHING_MATRIX_NUMBER_DISPLAY_BIT_INDEX)){
        std::cout << "number display bit is set" << std::endl;
    }else{
        std::cout << "number display bit is NOT set" << std::endl;
    }
    */



    int comp_img_dim = individual_image_size * (all_feature_vectors->size() + 1);
    cv::Size comp_img_size(comp_img_dim,comp_img_dim);


    int elem_size = ((*square_organoid_images)[0])->elemSize1();

    if(!already_allocated_image){

        if(elem_size == 1){
            comp_img = cv::Mat::zeros(comp_img_size,CV_16UC3);
        }else if(elem_size == 2){
            comp_img = cv::Mat::zeros(comp_img_size,CV_16UC3);
        }

        already_allocated_image = true;
    }

    //std::cout << "individual image sizes: " << individual_image_size << std::endl;

    if(draw_organoid_images){

        for(int img_id = 0; img_id < square_organoid_images->size();img_id++){

            int mapped_img_id = img_id;

            if(use_image_order){
                mapped_img_id = (*image_order)[img_id];
            }

            cv::Mat* current_img = (*square_organoid_images)[mapped_img_id];

            cv::Mat resized_square_org_img;

            cv::resize(*current_img,resized_square_org_img,cv::Size(individual_image_size,individual_image_size));
                    
            for(int local_row = 0; local_row < individual_image_size; local_row++){
                for(int local_col = 0; local_col < individual_image_size; local_col++){

                    int row_in_comp_img = (img_id + 1) * individual_image_size + local_row;
                    int col_in_comp_img = (img_id + 1) * individual_image_size + local_col;

                    if(elem_size == 1){

                        comp_img.at<cv::Vec3w>(local_row,col_in_comp_img) = cv::Vec3w(250,250,250).mul(resized_square_org_img.at<cv::Vec3b>(local_row,local_col));
                        comp_img.at<cv::Vec3w>(row_in_comp_img,local_col) = cv::Vec3w(250,250,250).mul(resized_square_org_img.at<cv::Vec3b>(local_row,local_col));
                    }else if(elem_size == 2){
                        comp_img.at<cv::Vec3w>(local_row,col_in_comp_img) = resized_square_org_img.at<cv::Vec3w>(local_row,local_col);
                        comp_img.at<cv::Vec3w>(row_in_comp_img,local_col) = resized_square_org_img.at<cv::Vec3w>(local_row,local_col);
                    }

                    
                
                }   
            }
        }
        draw_organoid_images = false;
    }


    

    //cv::namedWindow("Comp Image", cv::WINDOW_KEEPRATIO );
    //cv::imshow("Comp Image", comp_img);

    //cv::waitKey(0);

    unsigned int num_hardware_threads = std::thread::hardware_concurrency();

    std::vector<std::thread*> additional_threads;
    additional_threads.resize(num_hardware_threads - 1);

    int ids_per_thread = all_feature_vectors->size() / num_hardware_threads;

    for(int thread = 0; thread < num_hardware_threads - 1; thread++){

        int first_id_for_thread_to_process = ids_per_thread * thread;
        int last_id_for_thread_to_process = ids_per_thread * (thread + 1);

        //std::cout << "ids to process for thread " << thread << ": " << first_id_for_thread_to_process << " to " << last_id_for_thread_to_process << std::endl;
        
        additional_threads[thread] = new std::thread(fill_matching_matrix_in_feature_vector_range,first_id_for_thread_to_process,last_id_for_thread_to_process,&comp_img,cost_threshold,individual_image_size,&costs_from_matching_results,all_feature_vectors,all_matching_results,use_image_order,image_order);
    }

    int first_id_for_masterthread_to_process = ids_per_thread * (num_hardware_threads - 1);
    int last_id_for_masterthread_to_process = all_feature_vectors->size();

    //std::cout << "ids to process for masterthread: " << first_id_for_masterthread_to_process << " to " << last_id_for_masterthread_to_process << std::endl;

    fill_matching_matrix_in_feature_vector_range(first_id_for_masterthread_to_process,last_id_for_masterthread_to_process,&comp_img,cost_threshold,individual_image_size,&costs_from_matching_results,all_feature_vectors,all_matching_results,use_image_order,image_order);

    for(int thread = 0; thread < num_hardware_threads - 1; thread++){
        additional_threads[thread]->join();
    }
    

    

    std::string window_name = MATCHING_MATRIX_WINDOW_NAME;

    if(viz_type == MATCHING_VIZ_QUADRATIC){
        //window_name += " Quadratic";

        if(uint32_t_query_single_bit(&global_matching_matrix_mouse_callback_input.flags,MATCHING_MATRIX_WRITE_TO_FILE_BIT_INDEX)){
            uint32_t_set_single_bit_to_zero(&global_matching_matrix_mouse_callback_input.flags,MATCHING_MATRIX_WRITE_TO_FILE_BIT_INDEX);

            std::string output_name = "matching_matrix";
            append_current_date_and_time_to_string(output_name);
            output_name += ".tif";
            //cv::imwrite(output_name,comp_img);
        }

    } 

    if(viz_type == MATCHING_VIZ_QUADRATIC_OPTIMAL){
        window_name += " Optimal Quadratic";
    } 

    if(viz_type == MATCHING_VIZ_LIN_FEATURE){
        window_name += " Linear relative to num features";
    }

    if(viz_type == MATCHING_VIZ_LIN_CANDIDATE){
        window_name += " Linear relative to num candidates";
    }


    cv::imshow(window_name, comp_img);

    //cv::imwrite("matching_matrix.png",comp_img);

    cv::setMouseCallback(window_name, mouse_call_back_function_matching_matrix_window,&global_matching_matrix_mouse_callback_input);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
}

void find_optimial_viz_rotation_angle(double& rotation_angle, std::vector<int>* assignment, std::vector<Feature_Point_Data>* features,std::vector<Feature_Point_Data>* assigned_candidates ){

    if(assignment == nullptr){
        std::cout << "Assignment vector was NULL in find_optimal_viz_rotation_angle" << std::endl;
        return;
    }

    if(features == nullptr){
        std::cout << "Feature vector was NULL in find_optimal_viz_rotation_angle" << std::endl;
        return;
    }

    if(assigned_candidates == nullptr){
        std::cout << "Candidate vector was NULL in find_optimal_viz_rotation_angle" << std::endl;
        return;
    }

    double currently_best_rotation_angle = 0.0;
    double currently_best_total_angle_differnce = DBL_MAX;

    for(int angle = -180; angle < 180; angle++){
        double total_angle_difference = 0.0;

        for(int i = 0; i < assignment->size();i++){

            int assigned_candidate_index = (*assignment)[i];

            if(assigned_candidate_index == -1){
                continue;
            }

            Feature_Point_Data current_feature = (*features)[i];
            Feature_Point_Data current_assigned_candidate = (*assigned_candidates)[assigned_candidate_index];

            double angle_diff = get_signed_difference_between_two_angles(current_feature.angle,current_assigned_candidate.angle);

            total_angle_difference += fabs(get_signed_difference_between_two_angles(angle,angle_diff));

        }

        if(total_angle_difference < currently_best_total_angle_differnce){
            currently_best_rotation_angle = angle;
            currently_best_total_angle_differnce = total_angle_difference;
        }


    }

    rotation_angle = currently_best_rotation_angle;

}

void show_assignment(std::vector<int>* assignment, std::vector<Feature_Point_Data>* features,std::vector<Feature_Point_Data>* assigned_candidates, cv::Mat* feature_image, cv::Mat* candidate_image, cv::Point2i center_feature_image, cv::Point2i center_candidate_image, cv::Point2i offset_feature_image, cv::Point2i offset_candidate_image, std::filesystem::path path_to_feature_image, std::filesystem::path path_to_candidate_image){

    if(assignment == nullptr){
        std::cout << "Assignment vector was NULL in show_assignment" << std::endl;
        return;
    }

    if(features == nullptr){
        std::cout << "Feature vector was NULL in show_assignment" << std::endl;
        return;
    }

    if(assigned_candidates == nullptr){
        std::cout << "Candidate vector was NULL in show_assignment" << std::endl;
        return;
    }

    if(assignment->size() != features->size()){
        std::cout << "Size of the Feature vector did not match the size of the assignment vector" << std::endl;
        return;
    }

    cv::Mat loaded_feature_image;
    cv::Mat loaded_candidate_image;

    load_and_square_organoid_image(loaded_feature_image,path_to_feature_image);
    load_and_square_organoid_image(loaded_candidate_image,path_to_candidate_image);

    center_feature_image += offset_feature_image;
    center_candidate_image += offset_candidate_image;

    cv::Point2i center_offset = center_feature_image - center_candidate_image;

    double average_rotation_in_all_channels = 0.0;

    cv::namedWindow("Feature Image", cv::WINDOW_KEEPRATIO );
    //cv::imshow("Feature Image", *feature_image);
    cv::imshow("Feature Image", loaded_feature_image);

    cv::namedWindow("Candidate Image", cv::WINDOW_KEEPRATIO );
    //cv::imshow("Candidate Image", *candidate_image);
    cv::imshow("Candidate Image", loaded_candidate_image);


    find_optimial_viz_rotation_angle(average_rotation_in_all_channels,assignment,features,assigned_candidates);

    //average_rotation_in_all_channels = 0.0;

    cv::Mat rotation_matrix = get_ccw_rotation_matrix_by_center_point_and_degrees(center_feature_image.x,center_feature_image.y,average_rotation_in_all_channels); 
    cv::Mat reverse_rotation_matrix = get_ccw_rotation_matrix_by_center_point_and_degrees(center_candidate_image.x,center_candidate_image.y,-average_rotation_in_all_channels);

    cv::Mat rotated_feature_image;
    cv::Mat rotated_candidate_image;

    cv::Mat assignments_feature_image = loaded_feature_image.clone();
    cv::Mat assignments_candidate_image = loaded_candidate_image.clone(); 

    rotate_image_by_degrees_ccw_around_point(loaded_feature_image,rotated_feature_image,average_rotation_in_all_channels,center_feature_image.x,center_feature_image.y);
    rotate_image_by_degrees_ccw_around_point(loaded_candidate_image,rotated_candidate_image,-average_rotation_in_all_channels,center_candidate_image.x,center_candidate_image.y);

    double pixel_per_unit_length_in_features = 0.0;
    double pixels_per_unit_length_in_candidates = 0.0;

    for(int i = 0; i < features->size();i++){
        Feature_Point_Data current_feature = (*features)[i];

        cv::Point2i feature_pixel_pos{current_feature.col,current_feature.row};
        cv::Vec2i dist = feature_pixel_pos - (center_feature_image - offset_feature_image);

        float length = cv::norm(dist);

        if(length > 0.0){
            pixel_per_unit_length_in_features += cv::norm(dist) / current_feature.relative_distance_center_max_distance;
        }

    }

    for(int i = 0; i < assigned_candidates->size();i++){

        Feature_Point_Data current_candidate = (*assigned_candidates)[i];

        cv::Point2i candidate_pixel_pos{current_candidate.col,current_candidate.row};
        cv::Vec2i dist = candidate_pixel_pos - (center_candidate_image - offset_candidate_image);


        float length = cv::norm(dist);

        if(length > 0.0){
            pixels_per_unit_length_in_candidates += cv::norm(dist) / current_candidate.relative_distance_center_max_distance;
        }
    }

    pixel_per_unit_length_in_features /= features->size();
    pixels_per_unit_length_in_candidates /= assigned_candidates->size();

    for(int i = 0; i < features->size();i++){

        Feature_Point_Data current_feature = (*features)[i];

        cv::Mat affine_feature_coordinates(cv::Size(1,3),CV_64F);

        affine_feature_coordinates.at<double>(0,0) = current_feature.col + offset_feature_image.x;
        affine_feature_coordinates.at<double>(0,1) = current_feature.row + offset_feature_image.y;
        affine_feature_coordinates.at<double>(0,2) = 1;

        cv::Mat multi_res = rotation_matrix * affine_feature_coordinates;

        cv::Point2i feature_pos(affine_feature_coordinates.at<double>(0,0), affine_feature_coordinates.at<double>(0,1));
        cv::Point2i rotated_feature_pos(multi_res.at<double>(0,0), multi_res.at<double>(0,1));
        
        
        //cv::Scalar dot_colour(00000, 60000, 00000);
        int dot_thickness = 2;
        cv::Scalar dot_colour(30000, 30000, 30000);

        int assigned_candidate_index = (*assignment)[i];

        if(assigned_candidate_index != -1){
            dot_thickness = 1;
            dot_colour = cv::Scalar(60000, 60000, 00000);
        }

        int relative_col = 0;
        int relative_row = 0;

        pixel_pos_from_center_length_and_angle(relative_col,relative_row,center_candidate_image.x,center_candidate_image.y,current_feature.relative_distance_center_max_distance,current_feature.angle,pixels_per_unit_length_in_candidates);

        cv::Point2i feature_pos_in_candidate_img(relative_col, relative_row);

        //feature_pos_in_candidate_img -= center_feature_image;
        //feature_pos_in_candidate_img += center_candidate_image;

        cv::circle(assignments_feature_image, feature_pos, dot_thickness, dot_colour);
        cv::circle(rotated_feature_image, rotated_feature_pos, dot_thickness, dot_colour);
        cv::circle(rotated_candidate_image, feature_pos_in_candidate_img, dot_thickness, dot_colour);
    }

    for(int i = 0; i < assigned_candidates->size();i++){

        Feature_Point_Data current_candidate = (*assigned_candidates)[i];

        cv::Mat affine_candidate_coordinates(cv::Size(1,3),CV_64F);

        affine_candidate_coordinates.at<double>(0,0) = current_candidate.col + offset_candidate_image.x;
        affine_candidate_coordinates.at<double>(0,1) = current_candidate.row + offset_candidate_image.y;
        affine_candidate_coordinates.at<double>(0,2) = 1;

        cv::Mat candidate_multi_res = reverse_rotation_matrix * affine_candidate_coordinates;

        cv::Point2i candidate_pos(affine_candidate_coordinates.at<double>(0,0), affine_candidate_coordinates.at<double>(0,1));
        cv::Point2i rotated_candidate_pos(candidate_multi_res.at<double>(0,0), candidate_multi_res.at<double>(0,1));

        int relative_col = 0;
        int relative_row = 0;

        //std::cout << center_feature_image << " " <<  current_candidate.relative_distance_center_max_distance << " " << current_candidate.angle << " " << pixel_per_unit_length_in_features << std::endl;

        pixel_pos_from_center_length_and_angle(relative_col,relative_row,center_feature_image.x,center_feature_image.y,current_candidate.relative_distance_center_max_distance,current_candidate.angle,pixel_per_unit_length_in_features);


        cv::Point2i candidate_in_feature_img(relative_col, relative_row);

        cv::Scalar dot_colour(30000, 30000, 30000);
        int cross_size = 2;

        for(int j = 0; j < assignment->size();j++){
            if((*assignment)[j] == i){
                cross_size = 1;
                dot_colour = cv::Scalar(60000, 00000, 60000);
                break;
            }
        }

        draw_cross(assignments_candidate_image, candidate_pos, dot_colour, cross_size);
        draw_cross(rotated_candidate_image, rotated_candidate_pos, dot_colour, cross_size);
        draw_cross(rotated_feature_image, candidate_in_feature_img, dot_colour, cross_size);
    }


    for(int i = 0; i < assignment->size();i++){

        int assigned_candidate_index = (*assignment)[i];

        if(assigned_candidate_index == -1){
            continue;
        }

        Feature_Point_Data current_feature = (*features)[i];
        Feature_Point_Data current_assigned_candidate = (*assigned_candidates)[assigned_candidate_index];

        double angle_diff = get_signed_difference_between_two_angles(current_feature.angle,current_assigned_candidate.angle);

        cv::Mat affine_feature_coordinates(cv::Size(1,3),CV_64F);

        affine_feature_coordinates.at<double>(0,0) = current_feature.col + offset_feature_image.x;
        affine_feature_coordinates.at<double>(0,1) = current_feature.row + offset_feature_image.y;
        affine_feature_coordinates.at<double>(0,2) = 1;

        cv::Mat multi_res = rotation_matrix * affine_feature_coordinates;

        cv::Point2i start_not_rotated(affine_feature_coordinates.at<double>(0,0), affine_feature_coordinates.at<double>(0,1));
        cv::Point2i start(multi_res.at<double>(0,0), multi_res.at<double>(0,1));

        int relative_col = 0;
        int relative_row = 0;

        pixel_pos_from_center_length_and_angle(relative_col,relative_row,center_feature_image.x,center_feature_image.y,current_assigned_candidate.relative_distance_center_max_distance,current_assigned_candidate.angle,pixel_per_unit_length_in_features);


        cv::Point2i end(relative_col, relative_row);


        cv::Scalar start_dot_colour(60000, 00000, 60000);
        cv::Scalar end_dot_colour(60000, 60000, 60000);
        cv::Scalar line_colour(60000, 60000, 60000);


        float difference_in_distance = fabs(current_feature.relative_distance_center_max_distance - current_assigned_candidate.relative_distance_center_max_distance);
        float difference_in_color = fabs(current_feature.normalize_peak_value - current_assigned_candidate.normalize_peak_value);

        line_colour = cv::Scalar(60000, 60000 - 60000 * difference_in_color, 65000);

        cv::line(rotated_feature_image, start, end, line_colour, 1, cv::LINE_AA);

        cv::Mat affine_candidate_coordinates(cv::Size(1,3),CV_64F);

        affine_candidate_coordinates.at<double>(0,0) = current_assigned_candidate.col + offset_candidate_image.x;
        affine_candidate_coordinates.at<double>(0,1) = current_assigned_candidate.row + offset_candidate_image.y;
        affine_candidate_coordinates.at<double>(0,2) = 1;

        cv::Mat candidate_multi_res = reverse_rotation_matrix * affine_candidate_coordinates;

        cv::Point2i candidate_start_not_rotated(affine_candidate_coordinates.at<double>(0,0), affine_candidate_coordinates.at<double>(0,1));
        cv::Point2i candidate_start(candidate_multi_res.at<double>(0,0), candidate_multi_res.at<double>(0,1));
        

        relative_col = 0;
        relative_row = 0;

        pixel_pos_from_center_length_and_angle(relative_col,relative_row,center_candidate_image.x,center_candidate_image.y,current_feature.relative_distance_center_max_distance,current_feature.angle,pixels_per_unit_length_in_candidates);

        cv::Point2i candidate_end(relative_col, relative_row);


        start_dot_colour = cv::Scalar(60000, 60000, 60000);
        end_dot_colour = cv::Scalar(60000, 00000, 60000);

        cv::line(rotated_candidate_image, candidate_start, candidate_end, line_colour, 1, cv::LINE_AA);

    } 

    cv::namedWindow("Assignments Feature Image", cv::WINDOW_KEEPRATIO );
    cv::imshow("Assignments Feature Image", assignments_feature_image);

    cv::namedWindow("Assignments Candidate Image", cv::WINDOW_KEEPRATIO );
    cv::imshow("Assignments Candidate Image", assignments_candidate_image);

    cv::namedWindow("Rotated Feature Image", cv::WINDOW_KEEPRATIO );
    cv::imshow("Rotated Feature Image", rotated_feature_image);

    cv::namedWindow("Rotated Candidate Image", cv::WINDOW_KEEPRATIO );
    cv::imshow("Rotated Candidate Image", rotated_candidate_image);

}

void rotate_image_by_degrees_ccw(cv::Mat& input, cv::Mat& output, double degrees){

    cv::Mat rotation_matrix = get_ccw_rotation_matrix_by_image_size_and_degrees(input.cols,input.rows,degrees);

    cv::warpAffine(input,output,rotation_matrix,cv::Size(input.cols,input.rows));
}

void rotate_image_by_degrees_ccw_around_point(cv::Mat& input, cv::Mat& output, double degrees, int rotation_origin_col, int rotation_origin_row){

    cv::Mat rotation_matrix = get_ccw_rotation_matrix_by_center_point_and_degrees(rotation_origin_col,rotation_origin_row,degrees);

    cv::warpAffine(input,output,rotation_matrix,cv::Size(input.cols,input.rows));
}

cv::Mat get_ccw_rotation_matrix_by_image_size_and_degrees(int cols, int rows, double degrees){
    cv::Point2f rotation_center((float)cols/2.0f,(float)rows/2.0f);

    cv::Mat rotation_matrix = cv::getRotationMatrix2D(rotation_center,degrees,1.0);

    return rotation_matrix;

}

cv::Mat get_ccw_rotation_matrix_by_center_point_and_degrees(int center_col, int center_row, double degrees){
    cv::Point2f rotation_center(center_col,center_row);

    cv::Mat rotation_matrix = cv::getRotationMatrix2D(rotation_center,degrees,1.0);

    return rotation_matrix;

}

void show_clustering(std::vector<cv::Mat*>* square_organoid_images, std::vector<Cluster>* all_clusters, std::vector<Matching_Result>* all_matching_results,std::vector<Image_Features_Pair>* all_feature_vectors, int individual_image_size, std::vector<Single_Cluster_Layout>* cluster_layouts, All_Cost_Parameters& cost_params, int window_number, std::vector<Cluster_Representative_Pair>* selected_cluster_representatives){

    static bool callback_input_initialized = false;

    if(!callback_input_initialized){
        selected_cell_in_clustering.x = -1;
        selected_cell_in_clustering.y = -1;
        selected_cell_in_clustering.pixel_per_cell = individual_image_size;

        global_selected_clustering_window_cells.first_selected_col = -1;
        global_selected_clustering_window_cells.first_selected_row = -1;
        global_selected_clustering_window_cells.first_selected_img_num = -1;
        global_selected_clustering_window_cells.first_window_num = -1;

        global_selected_clustering_window_cells.second_selected_col = -1;
        global_selected_clustering_window_cells.second_selected_row = -1;
        global_selected_clustering_window_cells.second_selected_img_num = -1;
        global_selected_clustering_window_cells.second_window_num = -1;

        for(int i = 0; i < NUM_CLUSTERING_WINDOWS;i++){


            Clustering_Visualization_Mouse_Callback_Input new_clustering_mouse_callback_input_struct{i,nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr,nullptr,nullptr,cost_params,-1};

            new_clustering_mouse_callback_input_struct.square_organoid_images = global_matching_matrix_mouse_callback_input.square_organoid_images;//square_organoid_images;
            new_clustering_mouse_callback_input_struct.all_feature_vectors = global_matching_matrix_mouse_callback_input.all_feature_vectors;
            new_clustering_mouse_callback_input_struct.all_matching_results = global_matching_matrix_mouse_callback_input.all_matching_results;
            new_clustering_mouse_callback_input_struct.individual_image_size = individual_image_size;
            new_clustering_mouse_callback_input_struct.selected_cell = &selected_cell_in_clustering;
            new_clustering_mouse_callback_input_struct.selected_cells_in_clustering_windows = &global_selected_clustering_window_cells;
            //new_clustering_mouse_callback_input_struct.window_number = i;

            new_clustering_mouse_callback_input_struct.cost_params = cost_params;

            global_clustering_mouse_callback_inputs[i] = new_clustering_mouse_callback_input_struct;

            //std::cout << "initial mouse over: " << global_clustering_mouse_callback_inputs[i].mouse_over_rep_img_num << std::endl;
        }
        callback_input_initialized = true;
    }
    

    int num_organoid_images = square_organoid_images->size();

    int num_organoid_images_per_side = ceil(sqrt(num_organoid_images));

    int num_clusters = all_clusters->size();

    if(cluster_layouts == nullptr){
        cluster_layouts = new std::vector<Single_Cluster_Layout>;
    }

    global_clustering_mouse_callback_inputs[window_number].all_clusters = all_clusters;
    //global_clustering_mouse_callback_inputs[window_number].all_matching_results = all_matching_results;
    global_clustering_mouse_callback_inputs[window_number].selected_cell = &selected_cell_in_clustering;
    global_clustering_mouse_callback_inputs[window_number].cluster_layout = cluster_layouts;

    //std::cout << "selected_cluster_representatives: " << selected_cluster_representatives << std::endl;
    if(selected_cluster_representatives != nullptr){
        for(int i = 0; i < NUM_CLUSTERING_WINDOWS;i++){
            global_clustering_mouse_callback_inputs[window_number].selected_cluster_representatives = selected_cluster_representatives;
        }
    }


    if(square_organoid_images == nullptr || all_clusters == nullptr || all_matching_results == nullptr || all_feature_vectors == nullptr){
        std::cout << "square_organoid_image: " << square_organoid_images << std::endl;
        std::cout << "all_clusters: " << all_clusters << std::endl;
        std::cout << "all_matching_results: " << all_matching_results << std::endl;
        std::cout << "all_feature_vectors: " << all_feature_vectors << std::endl;

        return;
    }


    cluster_layouts->clear();

    for(int i = 0; i < all_clusters->size();i++){

        Cluster current_cluster = (*all_clusters)[i];

        Single_Cluster_Layout current_cluster_layout;

        current_cluster_layout.start_col_in_overall_layout = -1;
        current_cluster_layout.start_row_in_overall_layout = -1;

        int cluster_size = current_cluster.members->size();

        current_cluster_layout.index_in_cluster_vector = i;
        current_cluster_layout.num_members = cluster_size;

        current_cluster_layout.cols = ceil(sqrt(current_cluster_layout.num_members));

        int num_rows_in_cluster_layout = 0;

        while(cluster_size > 0){
            num_rows_in_cluster_layout++;
            cluster_size -= current_cluster_layout.cols;
        }

        current_cluster_layout.rows = num_rows_in_cluster_layout;

        cluster_layouts->push_back(current_cluster_layout);

    }

    //std::cout << "individual_image_size in clustering: " << individual_image_size << std::endl;

    std::sort(cluster_layouts->begin(),cluster_layouts->end(),Single_Cluster_Layout_Size_Compare);

    cv::Vec2i cluster_layout_size = calculate_cluster_layout(*cluster_layouts);

    int num_rows = individual_image_size * all_clusters->size();

    int max_num_members_in_cluster = 0;

    for(int i = 0; i < all_clusters->size();i++){
        if((*all_clusters)[i].members->size() > max_num_members_in_cluster){
            max_num_members_in_cluster = (*all_clusters)[i].members->size();
        }
    }

    int num_cols = individual_image_size * max_num_members_in_cluster;

    //cv::Size clusters_img_size(num_cols,num_rows);

    //cv::Mat clusters_img;

    cv::Size clusters_img_with_layout_size(cluster_layout_size[0] * individual_image_size,cluster_layout_size[1] * individual_image_size);

    cv::Mat cluster_img_with_layout;

    int elem_size = ((*square_organoid_images)[0])->elemSize1();

    if(elem_size == 1){
        //clusters_img = cv::Mat::zeros(clusters_img_size,CV_16UC3);
        cluster_img_with_layout = cv::Mat::zeros(clusters_img_with_layout_size,CV_16UC3);
    }else if(elem_size == 2){
       // clusters_img = cv::Mat::zeros(clusters_img_size,CV_16UC3);
        cluster_img_with_layout = cv::Mat::zeros(clusters_img_with_layout_size,CV_16UC3);
    }

 

    int highlight_frame_thickness = 5;


    for(int cluster_layout_id = 0; cluster_layout_id < cluster_layouts->size();cluster_layout_id++){

        Single_Cluster_Layout current_layout = (*cluster_layouts)[cluster_layout_id];

        Cluster current_cluster = all_clusters->at(current_layout.index_in_cluster_vector);

        for(int member_index = 0; member_index < current_cluster.members->size();member_index++){
            int image_number = current_cluster.members->at(member_index);

            bool is_selected_representative = false;

            if(image_number == global_clustering_mouse_callback_inputs[window_number].mouse_over_rep_img_num){
                //std::cout << "mouse over:" <<  global_clustering_mouse_callback_inputs[window_number].mouse_over_rep_img_num << std::endl;
                is_selected_representative = true;
            } 

            int image_index_square_org_image_vector = -1;

            for(int i = 0; i < all_feature_vectors->size();i++){
                if(image_number == all_feature_vectors->at(i).image_number){
                    image_index_square_org_image_vector = i;
                    break;
                }
            }

            if(image_index_square_org_image_vector != -1){

                cv::Mat* current_img = (*square_organoid_images)[image_index_square_org_image_vector];

                bool is_cluster_representative = false;

                if(global_clustering_mouse_callback_inputs[window_number].selected_cluster_representatives != nullptr){
                    for(int l = 0; l < global_clustering_mouse_callback_inputs[window_number].selected_cluster_representatives->size();l++){
                        if(image_number == (*(global_clustering_mouse_callback_inputs[window_number].selected_cluster_representatives))[l].representative_img_number){
                            is_cluster_representative = true;
                            break;
                        }
                    }
                }

                int member_start_row = (current_layout.start_row_in_overall_layout + floor(member_index / current_layout.cols));
                int member_start_col = (current_layout.start_col_in_overall_layout + (member_index % current_layout.cols));

                int member_start_row_in_pixels = member_start_row * individual_image_size;
                int member_start_col_in_pixels = member_start_col * individual_image_size;

                bool is_first_selected_image =  ((global_clustering_mouse_callback_inputs[window_number].selected_cells_in_clustering_windows->first_selected_col == member_start_col) &&
                                                (global_clustering_mouse_callback_inputs[window_number].selected_cells_in_clustering_windows->first_selected_row == member_start_row) && 
                                                (global_clustering_mouse_callback_inputs[window_number].selected_cells_in_clustering_windows->first_window_num == window_number));

                bool is_second_selected_image =  ((global_clustering_mouse_callback_inputs[window_number].selected_cells_in_clustering_windows->second_selected_col == member_start_col) &&
                                                (global_clustering_mouse_callback_inputs[window_number].selected_cells_in_clustering_windows->second_selected_row == member_start_row) &&
                                                (global_clustering_mouse_callback_inputs[window_number].selected_cells_in_clustering_windows->second_window_num == window_number));
                
                for(int local_row = 0; local_row < individual_image_size; local_row++){
                    for(int local_col = 0; local_col < individual_image_size; local_col++){

                        int row_in_cluster_img = member_start_row_in_pixels + local_row;
                        int col_in_cluster_img = member_start_col_in_pixels + local_col;

                        bool pixel_is_in_highlight = (local_row < highlight_frame_thickness || local_col < highlight_frame_thickness || local_row > individual_image_size - highlight_frame_thickness || local_col > individual_image_size - highlight_frame_thickness);

                        cv::Vec3w first_highlight_color = cv::Vec3w(65000, 65000, 65000);

                        cv::Vec3w second_highlight_color = cv::Vec3w(0, 65000, 65000);

                        cv::Vec3w cluster_rep_highlight_color = cv::Vec3w(30000, 0, 0);

                        cv::Vec3w highlight_color = cluster_rep_highlight_color;

                        if(is_selected_representative){
                            highlight_color = cv::Vec3w(60000,0,0);
                        }

                        if(is_first_selected_image){
                            highlight_color = first_highlight_color;
                        }

                        if(is_second_selected_image){
                            highlight_color= second_highlight_color;
                        }

                        if((is_first_selected_image || is_second_selected_image || is_cluster_representative || is_selected_representative) && pixel_is_in_highlight){
                            if(elem_size == 1){
                                cluster_img_with_layout.at<cv::Vec3w>(row_in_cluster_img,col_in_cluster_img) = highlight_color;
                            }else if(elem_size == 2){
                                cluster_img_with_layout.at<cv::Vec3w>(row_in_cluster_img,col_in_cluster_img) = highlight_color;

                            }

                        }else{
                            if(elem_size == 1){
                                cluster_img_with_layout.at<cv::Vec3w>(row_in_cluster_img,col_in_cluster_img) = cv::Vec3w(250,250,250).mul(current_img->at<cv::Vec3b>(local_row,local_col));
                            }else if(elem_size == 2){
                                cluster_img_with_layout.at<cv::Vec3w>(row_in_cluster_img,col_in_cluster_img) = cv::Vec3w(1,1,1).mul(current_img->at<cv::Vec3w>(local_row,local_col));

                            }
                        }


                    }   
                }


            }

        }

    }


    std::string clustering_window_name = get_clustering_window_name_by_window_num(window_number);

    cv::namedWindow(clustering_window_name, cv::WINDOW_KEEPRATIO );
    cv::imshow(clustering_window_name, cluster_img_with_layout);

    //cv::imwrite("clustering.png",cluster_img_with_layout);

    cv::setMouseCallback(clustering_window_name,mouse_call_back_function_clustering_window,global_clustering_mouse_callback_inputs + window_number);

}

cv::Vec2i calculate_cluster_layout(std::vector<Single_Cluster_Layout>& single_cluster_layouts){

    std::vector<Single_Cluster_Layout> best_layout;
    std::vector<Single_Cluster_Layout> current_layout = single_cluster_layouts;

    float current_best_distance_to_optimal_row_col_ratio = FLT_MAX;

    int total_cols_in_best_layout = 0;
    int total_rows_in_best_layout = 0;

    for(int num_clusters_in_first_row = 1; num_clusters_in_first_row  <= single_cluster_layouts.size();num_clusters_in_first_row++){

        current_layout = single_cluster_layouts;

        int total_num_cols_in_current_layout = 0;
        int total_num_rows_in_current_layout = 0;

        for(int i = 0;  i < num_clusters_in_first_row; i++){

            current_layout[i].start_row_in_overall_layout = 0;
            current_layout[i].start_col_in_overall_layout = total_num_cols_in_current_layout;
 
            total_num_cols_in_current_layout += current_layout[i].cols;

            if(i < (num_clusters_in_first_row - 1)){
                total_num_cols_in_current_layout++;
                //we want a single free column between each cluster
            }

            if(i == 0){
                //we take the rows of the largest cluster in the first row 
                total_num_rows_in_current_layout = current_layout[i].rows;
                //and add the seperating row to that
                total_num_rows_in_current_layout++;
            }
        }

        int col_in_current_row = 0;

        int index_of_first_in_current_row = num_clusters_in_first_row;

        for(int i = num_clusters_in_first_row; i < current_layout.size();i++){

            if(col_in_current_row + current_layout[i].cols > total_num_cols_in_current_layout){
                // we need to perform a linebreak
                total_num_rows_in_current_layout += current_layout[index_of_first_in_current_row].rows;
                total_num_rows_in_current_layout++;

                index_of_first_in_current_row = i;

                col_in_current_row = 0;
            }

            current_layout[i].start_col_in_overall_layout = col_in_current_row;
            current_layout[i].start_row_in_overall_layout = total_num_rows_in_current_layout;

            col_in_current_row += current_layout[i].cols;
            col_in_current_row++;
        }

        if(index_of_first_in_current_row < current_layout.size()){
            total_num_rows_in_current_layout += current_layout[index_of_first_in_current_row].rows;
        }

        float ratio = (float)total_num_rows_in_current_layout / (float)total_num_cols_in_current_layout;

        float dist_to_ratio_of_one = fabs(ratio - 1.0f);

        if(dist_to_ratio_of_one < current_best_distance_to_optimal_row_col_ratio){
            best_layout = current_layout;

            current_best_distance_to_optimal_row_col_ratio = dist_to_ratio_of_one;
            total_cols_in_best_layout = total_num_cols_in_current_layout;
            total_rows_in_best_layout = total_num_rows_in_current_layout;

        }else{
            break;
        }
    }

    cv::Vec2i total_layout_size;
    total_layout_size[0] = total_cols_in_best_layout;
    total_layout_size[1] = total_rows_in_best_layout;

    single_cluster_layouts = best_layout;

    return total_layout_size;
}

void mouse_call_back_function_matching_matrix_window(int event, int x, int y, int flags, void* userdata){
    
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (userdata != nullptr) {

            Matching_Matrix_Mouse_Callback_Input* input = (Matching_Matrix_Mouse_Callback_Input*)userdata;

            Selected_Cell* selected_cell = ((Matching_Matrix_Mouse_Callback_Input*)userdata)->selected_cell;
            
            int x_coord = floorf((float)(x - selected_cell->pixel_per_cell) / (float)selected_cell->pixel_per_cell);
            int y_coord = floorf((float)(y - selected_cell->pixel_per_cell) / (float)selected_cell->pixel_per_cell);


            if (x_coord >= 0 && y_coord >= 0) {
                selected_cell->x = x_coord;
                selected_cell->y = y_coord;

                int feature_id = input->all_feature_vectors->at(y_coord).image_number;
                int candidate_id = input->all_feature_vectors->at(x_coord).image_number;

                if(input->use_image_order && input->image_order->size() > 0){
                    feature_id = input->all_feature_vectors->at(input->image_order->at(y_coord)).image_number;
                    candidate_id = input->all_feature_vectors->at(input->image_order->at(x_coord)).image_number;
                }

                if(global_clustering_mouse_callback_inputs[0].all_clusters != nullptr && global_clustering_mouse_callback_inputs[0].cluster_layout != nullptr){

                    get_coordinates_in_cluster_visualization_by_image_number(feature_id,&global_clustering_mouse_callback_inputs[0].selected_cells_in_clustering_windows->first_selected_col,&global_clustering_mouse_callback_inputs[0].selected_cells_in_clustering_windows->first_selected_row,global_clustering_mouse_callback_inputs[0].cluster_layout,global_clustering_mouse_callback_inputs[0].all_clusters);

                    get_coordinates_in_cluster_visualization_by_image_number(candidate_id,&global_clustering_mouse_callback_inputs[0].selected_cells_in_clustering_windows->second_selected_col,&global_clustering_mouse_callback_inputs[0].selected_cells_in_clustering_windows->second_selected_row,global_clustering_mouse_callback_inputs[0].cluster_layout,global_clustering_mouse_callback_inputs[0].all_clusters);

                    global_clustering_mouse_callback_inputs[0].selected_cells_in_clustering_windows->first_window_num = 0;
                    global_clustering_mouse_callback_inputs[0].selected_cells_in_clustering_windows->second_window_num = 0;

                    show_clustering(input->square_organoid_images,global_clustering_mouse_callback_inputs[0].all_clusters,global_clustering_mouse_callback_inputs[0].all_matching_results,input->all_feature_vectors,global_clustering_mouse_callback_inputs[0].individual_image_size,global_clustering_mouse_callback_inputs[0].cluster_layout,global_clustering_mouse_callback_inputs[0].cost_params);
                }

                for(int i = 0; i < input->all_matching_results->size();i++){
                    Matching_Result current_matching_result = input->all_matching_results->at(i);

                    if(current_matching_result.id_1 == feature_id && current_matching_result.id_2 == candidate_id){
                        //std::cout << current_matching_result.rel_quadr_cost << " " << current_matching_result.additional_viz_data_id1_to_id2->assignment->size() << std::endl;

                        Matching_Result_Additional_Viz_Data* viz = current_matching_result.additional_viz_data_id1_to_id2;
                        if(viz != nullptr){

                            if(current_matching_result.assignment != nullptr && viz->features != nullptr && viz->assigned_candidates != nullptr){
                                if(current_matching_result.assignment->size() != 0 && viz->features->size() != 0 && viz->assigned_candidates->size() != 0){
                                    show_assignment(current_matching_result.assignment,viz->features,viz->assigned_candidates,viz->feature_image,viz->candidate_image,viz->center_feature_image,viz->center_candidate_image,viz->offset_feature_image,viz->offset_candidate_image,viz->path_to_feature_image,viz->path_to_candidate_image);
                                    get_cost_breakdown_for_single_problem_instance(current_matching_result.assignment,viz->features,viz->assigned_candidates,global_matching_matrix_mouse_callback_input.cost_params);
                                }else{
                                    std::cout << "Could not show assignment in mouse_call_back_funtion" << std::endl;
                                    std::cout << "assignment size: " << current_matching_result.assignment->size() << std::endl;
                                    std::cout << "feature vector size: " << viz->features->size() << std::endl;
                                    std::cout << "canididates vector size: " << viz->assigned_candidates->size() << std::endl;
                                }
                            }else{
                                std::cout << "Non valid pointers in mouse_call_back_funtion" << std::endl;
                                std::cout << "Assignment vector address: " << current_matching_result.assignment << std::endl;
                                std::cout << "Feature vector address: " << viz->features << std::endl;
                                std::cout << "Candidates vector address: " << viz->assigned_candidates << std::endl;
                            }

                        }
                        break;
                    }

                }
                
            }

            show_matching_matrix(input->square_organoid_images,input->all_matching_results,input->all_feature_vectors,input->individual_image_size,input->cost_threshold,input->viz_type,input->flags,input->image_order,input->use_image_order, global_matching_matrix_mouse_callback_input.cost_params);
        }
        else {
            std::cout << "user data was null" << std::endl;
        }
    }
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        //std::cout << "Right button of the mouse is clicked - reset selected cell" << std::endl;
        if (userdata != nullptr) {
            Matching_Matrix_Mouse_Callback_Input* input = (Matching_Matrix_Mouse_Callback_Input*)userdata;

            Selected_Cell* selected_cell = ((Matching_Matrix_Mouse_Callback_Input*)userdata)->selected_cell;

            if(selected_cell->x != -1 && selected_cell->y != -1){
                cv::destroyWindow("Feature Image");
                cv::destroyWindow("Candidate Image");
                cv::destroyWindow("Assignments Feature Image");
                cv::destroyWindow("Assignments Candidate Image");
                cv::destroyWindow("Rotated Feature Image");
                cv::destroyWindow("Rotated Candidate Image");
            }

            selected_cell->x = -1;
            selected_cell->y = -1;

            show_matching_matrix(input->square_organoid_images,input->all_matching_results,input->all_feature_vectors,input->individual_image_size,input->cost_threshold,input->viz_type,input->flags,input->image_order,input->use_image_order,global_matching_matrix_mouse_callback_input.cost_params);
        }
    }
    else if (event == cv::EVENT_MBUTTONDOWN)
    {
        //std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }/*
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;

    }
    */
}

void mouse_call_back_function_clustering_window(int event, int x, int y, int flags, void* userdata){

    if(event == cv::EVENT_MOUSEMOVE){
        Clustering_Visualization_Mouse_Callback_Input* input = (Clustering_Visualization_Mouse_Callback_Input*)userdata;

        if(input->selected_cluster_representatives == nullptr){
            //std::cout << "input->selected_cluster_representatives was null" << std::endl; 
            return;
        }

        Selected_Cell* selected_cell = input->selected_cell;

        int x_coord = floorf((float)(x) / (float)selected_cell->pixel_per_cell);
        int y_coord = floorf((float)(y) / (float)selected_cell->pixel_per_cell);

        bool mouse_over_no_cluster = true;

        for(int i = 0; i < input->cluster_layout->size();i++){
            Single_Cluster_Layout single_layout = input->cluster_layout->at(i);

            if(x_coord >= single_layout.start_col_in_overall_layout && y_coord >= single_layout.start_row_in_overall_layout && x_coord < single_layout.start_col_in_overall_layout + single_layout.cols && y_coord < single_layout.start_row_in_overall_layout + single_layout.rows){

                mouse_over_no_cluster = false;

                int row_offset_in_cluster = y_coord - single_layout.start_row_in_overall_layout;
                int col_offset_in_cluster = x_coord - single_layout.start_col_in_overall_layout;

                int member_id = col_offset_in_cluster + single_layout.cols * row_offset_in_cluster;

                if(member_id >= single_layout.num_members){
                    return;
                }

                Cluster current_cluster = (*input->all_clusters)[single_layout.index_in_cluster_vector];

                int image_number = current_cluster.members->at(member_id);

                bool img_is_representative = false;

                int prev_rep_img_num = input->mouse_over_rep_img_num;

                if(prev_rep_img_num == image_number){
                    return;
                } 

                for(int j = 0; j < input->selected_cluster_representatives->size();j++){
                    int rep_img_num = (*(input->selected_cluster_representatives))[j].representative_img_number;

                    if(rep_img_num == image_number){
                        img_is_representative = true;
                    }

                }

                if(img_is_representative){
                    for(int k = 0; k < NUM_CLUSTERING_WINDOWS;k++){
                        global_clustering_mouse_callback_inputs[k].mouse_over_rep_img_num = image_number;
                        Clustering_Visualization_Mouse_Callback_Input* cur_in = global_clustering_mouse_callback_inputs + k;
                        show_clustering(cur_in->square_organoid_images,cur_in->all_clusters,cur_in->all_matching_results,cur_in->all_feature_vectors,cur_in->individual_image_size,cur_in->cluster_layout,cur_in->cost_params,cur_in->window_number,cur_in->selected_cluster_representatives);
                    }

                    //std::cout << "set to: " << image_number << std::endl;
                }else{
                    if(input->mouse_over_rep_img_num == -1){
                        return;
                    }

                    //std::cout << "set to: -1" << std::endl;

                    for(int k = 0; k < NUM_CLUSTERING_WINDOWS;k++){
                        global_clustering_mouse_callback_inputs[k].mouse_over_rep_img_num = -1;
                        Clustering_Visualization_Mouse_Callback_Input* cur_in = global_clustering_mouse_callback_inputs + k;
                        show_clustering(cur_in->square_organoid_images,cur_in->all_clusters,cur_in->all_matching_results,cur_in->all_feature_vectors,cur_in->individual_image_size,cur_in->cluster_layout,cur_in->cost_params,cur_in->window_number,cur_in->selected_cluster_representatives);
                    }
                }
            }
        }


        if(mouse_over_no_cluster){
            if(input->mouse_over_rep_img_num != -1){
                //std::cout << "set to: -1" << std::endl;
                for(int k = 0; k < NUM_CLUSTERING_WINDOWS;k++){
                    global_clustering_mouse_callback_inputs[k].mouse_over_rep_img_num = -1;
                    Clustering_Visualization_Mouse_Callback_Input* cur_in = global_clustering_mouse_callback_inputs + k;
                    show_clustering(cur_in->square_organoid_images,cur_in->all_clusters,cur_in->all_matching_results,cur_in->all_feature_vectors,cur_in->individual_image_size,cur_in->cluster_layout,cur_in->cost_params,cur_in->window_number,cur_in->selected_cluster_representatives);
                }
            }
        }
    }
    
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        Clustering_Visualization_Mouse_Callback_Input* input = (Clustering_Visualization_Mouse_Callback_Input*)userdata;

        //std::cout << "individual image size in clustering callback " << input->individual_image_size << std::endl;

        Selected_Cell* selected_cell = input->selected_cell;

        int x_coord = floorf((float)(x) / (float)selected_cell->pixel_per_cell);
        int y_coord = floorf((float)(y) / (float)selected_cell->pixel_per_cell);
        //std::cout << "Left button is clicked - selected: " << x_coord << " " << y_coord << std::endl;

        for(int i = 0; i < input->cluster_layout->size();i++){
            Single_Cluster_Layout single_layout = input->cluster_layout->at(i);

            if(x_coord >= single_layout.start_col_in_overall_layout && y_coord >= single_layout.start_row_in_overall_layout && x_coord < single_layout.start_col_in_overall_layout + single_layout.cols && y_coord < single_layout.start_row_in_overall_layout + single_layout.rows){

                int row_offset_in_cluster = y_coord - single_layout.start_row_in_overall_layout;
                int col_offset_in_cluster = x_coord - single_layout.start_col_in_overall_layout;

                int member_id = col_offset_in_cluster + single_layout.cols * row_offset_in_cluster;

                if(member_id >= single_layout.num_members){
                    return;
                }

                Cluster current_cluster = (*input->all_clusters)[single_layout.index_in_cluster_vector];

                int image_number = current_cluster.members->at(member_id);

                if(input->selected_cells_in_clustering_windows->second_selected_img_num != -1 || (input->selected_cells_in_clustering_windows->second_selected_img_num == -1 && input->selected_cells_in_clustering_windows->first_selected_img_num == -1)){

                    input->selected_cells_in_clustering_windows->first_selected_col = x_coord;
                    input->selected_cells_in_clustering_windows->first_selected_row = y_coord;
                    input->selected_cells_in_clustering_windows->first_selected_img_num = image_number;
                    input->selected_cells_in_clustering_windows->first_window_num = input->window_number;

                    input->selected_cells_in_clustering_windows->second_selected_col = -1;
                    input->selected_cells_in_clustering_windows->second_selected_row = -1;
                    input->selected_cells_in_clustering_windows->second_selected_img_num = -1;
                    input->selected_cells_in_clustering_windows->second_window_num = -1;

                    show_clustering(input->square_organoid_images,input->all_clusters,input->all_matching_results,input->all_feature_vectors,input->individual_image_size,input->cluster_layout,input->cost_params,input->window_number);
                }else if(input->selected_cells_in_clustering_windows->first_selected_img_num != -1 && input->selected_cells_in_clustering_windows->second_selected_img_num == -1 && input->selected_cells_in_clustering_windows->first_selected_img_num != image_number){

                    input->selected_cells_in_clustering_windows->second_selected_col = x_coord;
                    input->selected_cells_in_clustering_windows->second_selected_row = y_coord;
                    input->selected_cells_in_clustering_windows->second_selected_img_num = image_number;
                    input->selected_cells_in_clustering_windows->second_window_num = input->window_number;

                    for(int j = 0; j < input->all_matching_results->size();j++){
                        Matching_Result current_matching_result = (*input->all_matching_results)[j];

                        if(current_matching_result.id_1 == input->selected_cells_in_clustering_windows->first_selected_img_num && current_matching_result.id_2 == input->selected_cells_in_clustering_windows->second_selected_img_num){

                            Matching_Result_Additional_Viz_Data* viz = current_matching_result.additional_viz_data_id1_to_id2;
                            if(viz != nullptr){

                                if(current_matching_result.assignment != nullptr && viz->features != nullptr && viz->assigned_candidates != nullptr){
                                    if(current_matching_result.assignment->size() != 0 && viz->features->size() != 0 && viz->assigned_candidates->size() != 0){
                                        show_assignment(current_matching_result.assignment,viz->features,viz->assigned_candidates,viz->feature_image,viz->candidate_image,viz->center_feature_image,viz->center_candidate_image,viz->offset_feature_image,viz->offset_candidate_image,viz->path_to_feature_image,viz->path_to_candidate_image);
                                        get_cost_breakdown_for_single_problem_instance(current_matching_result.assignment,viz->features,viz->assigned_candidates,input->cost_params);
                                    }else{
                                        std::cout << "Could not show assignment in mouse_call_back_function_clustering_window" << std::endl;
                                        std::cout << "assignment size: " << current_matching_result.assignment->size() << std::endl;
                                        std::cout << "feature vector size: " << viz->features->size() << std::endl;
                                        std::cout << "canididates vector size: " << viz->assigned_candidates->size() << std::endl;
                                    }
                                }else{
                                    std::cout << "Non valid pointers in mouse_call_back_function_clustering_window" << std::endl;
                                    std::cout << "Assignment vector address: " << current_matching_result.assignment << std::endl;
                                    std::cout << "Feature vector address: " << viz->features << std::endl;
                                    std::cout << "Candidates vector address: " << viz->assigned_candidates << std::endl;
                                }

                            }
                            break;
                        }
                    }

                    int index_of_features = -1;
                    int index_of_candidates = -1;

                    for(int j = 0; j < input->all_feature_vectors->size();j++){

                        Image_Features_Pair current_ifp = (*input->all_feature_vectors)[j];

                        if(current_ifp.image_number == input->selected_cells_in_clustering_windows->first_selected_img_num){
                            index_of_features = j;
                        }

                        if(current_ifp.image_number == input->selected_cells_in_clustering_windows->second_selected_img_num){
                            index_of_candidates = j;
                        }

                        if(index_of_features != -1 && index_of_candidates != -1){
                            break;
                        }
                    }

                    //std::cout << "index of feature: " << index_of_features << " index of candidates: " << index_of_candidates << std::endl;

                    bool remapped_features = false;
                    bool remapped_candidates = false;

                    if(global_matching_matrix_mouse_callback_input.image_order != nullptr && global_matching_matrix_mouse_callback_input.use_image_order){

                        for(int l = 0; l < global_matching_matrix_mouse_callback_input.image_order->size();l++ ){
                            if(((*global_matching_matrix_mouse_callback_input.image_order)[l] == index_of_features) && !remapped_features){
                                //std::cout << "remapping features " << std::endl;
                                index_of_features = l;
                                remapped_features = true;
                            }

                            if(((*global_matching_matrix_mouse_callback_input.image_order)[l] == index_of_candidates) && !remapped_candidates){
                                //std::cout << "remapping candidates" << std::endl;
                                index_of_candidates = l;
                                remapped_candidates = true;
                            }
                            //std::cout << l << " " << global_matching_matrix_mouse_callback_input.image_order->at(l) << std::endl;

                            if(remapped_features && remapped_candidates){
                                break;
                            }
                        }
                        //std::cout << "remapped: " << index_of_features << " " << index_of_candidates << std::endl;
                    }

                    global_matching_matrix_mouse_callback_input.selected_cell->y = index_of_features;
                    global_matching_matrix_mouse_callback_input.selected_cell->x = index_of_candidates;
                
                    //std::cout << index_of_features << " " << index_of_candidates << std::endl;
                    show_matching_matrix(input->square_organoid_images,input->all_matching_results,input->all_feature_vectors,global_matching_matrix_mouse_callback_input.individual_image_size,global_matching_matrix_mouse_callback_input.cost_threshold,global_matching_matrix_mouse_callback_input.viz_type,global_matching_matrix_mouse_callback_input.flags,global_matching_matrix_mouse_callback_input.image_order,global_matching_matrix_mouse_callback_input.use_image_order,global_matching_matrix_mouse_callback_input.cost_params);
                    show_clustering(input->square_organoid_images,input->all_clusters,input->all_matching_results,input->all_feature_vectors,input->individual_image_size,input->cluster_layout,input->cost_params,input->window_number);
                }
            }

        }
    }
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        //std::cout << "Right button is clicked - reset selected cell in clustering window" << std::endl;

        Clustering_Visualization_Mouse_Callback_Input* input = (Clustering_Visualization_Mouse_Callback_Input*)userdata;

        global_selected_clustering_window_cells.first_selected_col = -1;
        global_selected_clustering_window_cells.first_selected_row = -1;
        global_selected_clustering_window_cells.first_selected_img_num = -1;
        global_selected_clustering_window_cells.first_window_num = -1;

        global_selected_clustering_window_cells.second_selected_col = -1;
        global_selected_clustering_window_cells.second_selected_row = -1;
        global_selected_clustering_window_cells.second_selected_img_num = -1;
        global_selected_clustering_window_cells.second_window_num = -1;

        show_clustering(input->square_organoid_images,input->all_clusters,input->all_matching_results,input->all_feature_vectors,input->individual_image_size,input->cluster_layout,input->cost_params);
    }
}

void reset_clustering_visualization_selection(){

    global_selected_clustering_window_cells.first_selected_col = -1;
    global_selected_clustering_window_cells.first_selected_row = -1;
    global_selected_clustering_window_cells.first_selected_img_num = -1;
    global_selected_clustering_window_cells.first_window_num = -1;

    global_selected_clustering_window_cells.second_selected_col = -1;
    global_selected_clustering_window_cells.second_selected_row = -1;
    global_selected_clustering_window_cells.second_selected_img_num = -1;
    global_selected_clustering_window_cells.second_window_num = -1;

}

void update_selected_cells_after_image_order_switch(bool use_image_order){

    int index_of_features = global_matching_matrix_mouse_callback_input.selected_cell->y;
    int index_of_candidates = global_matching_matrix_mouse_callback_input.selected_cell->x;


    if(use_image_order){
        bool remapped_features = false;
        bool remapped_candidates = false;

        //std::cout << "reverse search" << std::endl;
        if(global_matching_matrix_mouse_callback_input.image_order != nullptr){

            for(int l = 0; l < global_matching_matrix_mouse_callback_input.image_order->size();l++ ){
                if(((*global_matching_matrix_mouse_callback_input.image_order)[l] == index_of_features) && !remapped_features){
                    //std::cout << "remapping features " << std::endl;
                    index_of_features = l;
                    remapped_features = true;
                }

                if(((*global_matching_matrix_mouse_callback_input.image_order)[l] == index_of_candidates) && !remapped_candidates){
                    //std::cout << "remapping candidates" << std::endl;
                    index_of_candidates = l;
                    remapped_candidates = true;
                }
                //std::cout << l << " " << global_matching_matrix_mouse_callback_input.image_order->at(l) << std::endl;

                if(remapped_features && remapped_candidates){
                    break;
                }
            }
        }
    }else{
        //std::cout << "direct mapping" << std::endl;

        index_of_features = (*global_matching_matrix_mouse_callback_input.image_order)[index_of_features];
        index_of_candidates = (*global_matching_matrix_mouse_callback_input.image_order)[index_of_candidates];
    }

    global_matching_matrix_mouse_callback_input.selected_cell->y = index_of_features;
    global_matching_matrix_mouse_callback_input.selected_cell->x = index_of_candidates;    
    //std::cout << "remapped: " << index_of_features << " " << index_of_candidates << std::endl;
}

void get_coordinates_in_cluster_visualization_by_image_number(int image_number, int* col, int* row, std::vector<Single_Cluster_Layout>* cluster_layout, std::vector<Cluster>* all_clusters){

    for(int i = 0; i < cluster_layout->size();i++){

        Single_Cluster_Layout current_layout = (*cluster_layout)[i];

        Cluster current_cluster = (*all_clusters)[current_layout.index_in_cluster_vector];

        for(int j = 0; j < current_cluster.members->size();j++){

            int current_image_number = (*current_cluster.members)[j];

            if(current_image_number == image_number){
                *row = (current_layout.start_row_in_overall_layout + floor(j / current_layout.cols));
                *col = (current_layout.start_col_in_overall_layout + (j % current_layout.cols)); 
                //std::cout << *row << " " << *col << std::endl;
                return;
            }
        }
    }
    
}

int calculate_local_search_dim_from_image_size(cv::Size2i image_size, double target_percentage_of_image_size){

    int largest_dim = image_size.width;

    if(largest_dim < image_size.height){
        largest_dim = image_size.height;
    }

    return roundf((double)largest_dim * target_percentage_of_image_size);
}

void load_and_square_organoid_image(cv::Mat& squared_organoid_image, std::filesystem::path organoid_image_path){

    cv::Mat original_organoid_image = cv::imread(organoid_image_path.string());

    int max_dim = std::max<int>(original_organoid_image.cols,original_organoid_image.rows);

    cv::Size square_img_size(max_dim,max_dim);

    int elem_size = original_organoid_image.elemSize1();

    //std::cout << "ELEM SIZE: " << elem_size << " " << new_organoid_image->elemSize() << std::endl;

    int square_img_type = CV_16UC3;
    if( elem_size != 2 && elem_size != 1){
        std::cout << "unsupported elem size in load_and_square_organoid_image: " << elem_size << std::endl;
    }

    if(original_organoid_image.channels() == 4){
        cv::cvtColor(original_organoid_image,original_organoid_image,cv::COLOR_BGRA2BGR);
    }

    squared_organoid_image = cv::Mat::zeros(square_img_size,square_img_type);

    int col_offset = (max_dim - original_organoid_image.cols) >> 1;
    int row_offset = (max_dim - original_organoid_image.rows) >> 1;


    for(int local_row = 0; local_row < original_organoid_image.rows;local_row++){
        for(int local_col = 0; local_col < original_organoid_image.cols;local_col++){

            int shifted_row = local_row + row_offset;
            int shifted_col = local_col + col_offset;

            double scaling_factor = (double)(1 << 16) / (double)(1 << 8);

            if(elem_size == 1){
                squared_organoid_image.at<cv::Vec3w>(shifted_row,shifted_col) = cv::Vec3w(scaling_factor,scaling_factor,scaling_factor).mul(original_organoid_image.at<cv::Vec3b>(local_row,local_col));
                //new_square_organoid_image.at<cv::Vec3b>(shifted_row,shifted_col) = new_organoid_image->at<cv::Vec3b>(local_row,local_col);
            }else if(elem_size == 2){
                squared_organoid_image.at<cv::Vec3w>(shifted_row,shifted_col) = original_organoid_image.at<cv::Vec3w>(local_row,local_col) ;
            }

        }
    }

    //squared_organoid_image *= 2;

}

void fill_costs_from_matching_results_vector(std::vector<double>& costs_from_matching_results, std::vector<Image_Features_Pair>* all_feature_vectors, std::vector<Matching_Result>* all_matching_results, bool use_image_order, std::vector<int>* image_order){

    costs_from_matching_results.clear();

    costs_from_matching_results.resize(all_feature_vectors->size() * all_feature_vectors->size());

    for(int i = 0; i < all_feature_vectors->size();i++){

        int first_image_num = (*all_feature_vectors)[i].image_number;

        if(use_image_order){
            first_image_num = (*all_feature_vectors)[(*image_order)[i]].image_number;
        }

        for(int j = 0; j < all_feature_vectors->size();j++){

            int second_image_num = (*all_feature_vectors)[j].image_number;

            if(use_image_order){
                second_image_num = (*all_feature_vectors)[(*image_order)[j]].image_number;
            }


            int index = i * all_feature_vectors->size() + j;
            Matching_Result matching_result;
            double cost = 0.0;

            for(int k = 0; k < all_matching_results->size();k++){

                Matching_Result current_matching_result = (*all_matching_results)[k];

                if((current_matching_result.id_1 == first_image_num && current_matching_result.id_2 == second_image_num)){

                    cost = current_matching_result.rel_quadr_cost;
                    break;
                }
            }

            costs_from_matching_results[index] = cost;

        }
    }

}

void fill_matching_matrix_in_feature_vector_range(int start, int end,cv::Mat* matching_matrix_img, double cost_threshold, int individual_image_size, std::vector<double>* costs_from_matching_results, std::vector<Image_Features_Pair>* all_feature_vectors, std::vector<Matching_Result>* all_matching_results, bool use_image_order, std::vector<int>* image_order){

    for(int i = start; i < end;i++){

        int first_image_num = (*all_feature_vectors)[i].image_number;

        if(use_image_order){
            first_image_num = (*all_feature_vectors)[(*image_order)[i]].image_number;
        }

        for(int j = 0; j < all_feature_vectors->size();j++){

            if(i == j){
                continue;
            }

            int second_image_num = (*all_feature_vectors)[j].image_number;

            if(use_image_order){
                second_image_num = (*all_feature_vectors)[(*image_order)[j]].image_number;
            }

            //std::cout << " " << second_image_num;

            Matching_Result matching_result;

            bool found_matching_result = true;
            int index = i * all_feature_vectors->size() + j;

            float cost = (*costs_from_matching_results)[index];
            //std::cout << cost << " " << std::endl;

            if(uint32_t_query_single_bit(&global_matching_matrix_mouse_callback_input.flags,MATCHING_MATRIX_SYMMETRIZE_BIT_INDEX)) {

                    int transposed_index = j * all_feature_vectors->size() + i;

                    double cost_at_transposed_index = (*costs_from_matching_results)[transposed_index];

                    if(cost_at_transposed_index < cost){
                        //std::cout << cost_at_transposed_index << " " << cost << std::endl;
                        cost = cost_at_transposed_index;
                    }
            }

            
            /*
            for(int k = 0; k < all_matching_results->size();k++){

                Matching_Result current_matching_result = (*all_matching_results)[k];

                if (!uint32_t_query_single_bit(&global_matching_matrix_mouse_callback_input.flags,MATCHING_MATRIX_SYMMETRIZE_BIT_INDEX)) {
                    if((current_matching_result.id_1 == first_image_num && current_matching_result.id_2 == second_image_num)){
                        matching_result = current_matching_result;
                        found_matching_result = true;
                        cost = matching_result.rel_quadr_cost;
                        break;
                    }
                }
                else {
                    if ((current_matching_result.id_1 == first_image_num && current_matching_result.id_2 == second_image_num) || (current_matching_result.id_2 == first_image_num && current_matching_result.id_1 == second_image_num)) {
                        matching_result = current_matching_result;
                        found_matching_result = true;
                        if (cost > matching_result.rel_quadr_cost) {
                            cost = matching_result.rel_quadr_cost;
                        }
                        //break;
                    }

                }
            
            }
            
            */

            if(!found_matching_result){
                continue;
            }
            
            cv::Vec3w highlight_color(65000,65000,0);

            cv::Vec3w midcolor(65000, 65000, 65000);
            cv::Vec3w negative_color(0, 0, 65000);
            cv::Vec3w positive_color(65000, 0, 0);

            cv::Vec3w color_unexpected_value(65000,0,0);

            cv::Vec3w color(0,0,0);

            if(uint32_t_query_single_bit(&global_matching_matrix_mouse_callback_input.flags,MATCHING_MATRIX_COLORGRADIENT_BIT_INDEX)){

                highlight_color = cv::Vec3w(0,65000,0);

                float distance = cost - cost_threshold;

                float intervall_length = 0.1f;


                if (distance >= 0.0) {

                    //intervall_length = 1.0f - cost_threshold;
                    //distance = (cost - cost_threshold) / intervall_length;
                    distance /= intervall_length;

                    color = midcolor * (1.0f - distance) + positive_color * distance;
                }else {
                    //intervall_length = cost_threshold;
                    //distance = cost / intervall_length;
                    distance /= intervall_length;

                    distance *= -1;
                    color = midcolor * (1.0f - distance) + negative_color * distance;
                }

            }else{
                color = cv::Vec3w(0,0,65000);

                if(cost > cost_threshold){
                    color = cv::Vec3w(0,65000,0);
                }

                if(fabs(cost / cost_threshold - 1.0) < TOLERANCE_MATCHING_THRESHOLD){
                    color = cv::Vec3w(0,65000,65000);
                }

                if(cost < 0){
                    color = color_unexpected_value;
                }
            }

            //std::cout << cost << " " << cost_threshold << " " << color << " " << first_image_num << " " << second_image_num << std::endl;

            int highlight_frame_thickness = 3;
            int highlight_row_col_marker_thickness = 2;

            for(int local_row = 0; local_row < individual_image_size; local_row++){
                for(int local_col = 0; local_col < individual_image_size; local_col++){

                    int row_in_comp_img = (i + 1) * individual_image_size + local_row;
                    int col_in_comp_img = (j + 1) * individual_image_size + local_col;


                    bool is_selected_cell = (j == selected_cell_in_matching_matrix.x) && (i == selected_cell_in_matching_matrix.y) && (local_row < highlight_frame_thickness ||  local_row > individual_image_size - highlight_frame_thickness || local_col < highlight_frame_thickness ||  local_col > individual_image_size - highlight_frame_thickness);
                    bool is_correct_row = (i == selected_cell_in_matching_matrix.y) && (local_row < highlight_row_col_marker_thickness ||  local_row >= individual_image_size - highlight_row_col_marker_thickness);
                    bool is_correct_col = (j == selected_cell_in_matching_matrix.x) && (local_col < highlight_row_col_marker_thickness ||  local_col >= individual_image_size - highlight_row_col_marker_thickness);

                    if(is_correct_col || is_correct_row || is_selected_cell){
                       
                        matching_matrix_img->at<cv::Vec3w>(row_in_comp_img, col_in_comp_img) = highlight_color;
                    }else {
                        matching_matrix_img->at<cv::Vec3w>(row_in_comp_img, col_in_comp_img) = color;

                    }
                }

            }
            
            if(uint32_t_query_single_bit(&global_matching_matrix_mouse_callback_input.flags,MATCHING_MATRIX_NUMBER_DISPLAY_BIT_INDEX)){
                std::string cost_string = std::to_string(cost);
                cost_string.resize(4);

                int baseline = 0;

                double font_scale = individual_image_size / 128.0;

                cv::Size bb_size = cv::getTextSize(cost_string, cv::FONT_HERSHEY_DUPLEX, font_scale,1,&baseline);

                //std::cout << baseline << " " << bb_size;

                cv::putText(*matching_matrix_img, //target image
                cost_string, //text
                cv::Point((j + 1) * individual_image_size + individual_image_size / 2 - bb_size.width / 2, (i + 1) * individual_image_size + individual_image_size / 2 + bb_size.height / 2), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                font_scale,
                CV_RGB(0, 0, 0), //font color
                1);

            }

        
        }
    }


}

void calculate_histogram(cv::Mat& image, std::vector<cv::Mat>& histograms, int image_number){

    histograms.clear();

    cv::Mat image_8bit;

    image.convertTo(image_8bit,CV_8UC3);


    std::vector<cv::Mat> color_channels;
    cv::split(image_8bit,color_channels);


    int histogram_size = 256;

    float range[] = {0,256};

    const float* histogram_range[] = {range};

    bool uniform_histogram = true;
    bool accumulate_histogram = false;

    cv::Mat histogram_blue;
    cv::Mat histogram_green;
    cv::Mat histogram_red;

    //int channels[3] = {0,1,2};
    //int histogram_sizes[3] = {0,1,2};
    //const float* histogram_ranges[3] = {range,range,range};
    cv::Mat histogram_rgb;

    cv::calcHist(&color_channels[0],1,0,cv::Mat(),histogram_blue,1,&histogram_size,histogram_range,uniform_histogram,accumulate_histogram);
    cv::calcHist(&color_channels[1],1,0,cv::Mat(),histogram_green,1,&histogram_size,histogram_range,uniform_histogram,accumulate_histogram);
    cv::calcHist(&color_channels[2],1,0,cv::Mat(),histogram_red,1,&histogram_size,histogram_range,uniform_histogram,accumulate_histogram); 

    //cv::calcHist(image_8bit,1,channels,cv::Mat(),histogram_rgb,3,histogram_sizes,histogram_ranges,uniform_histogram,accumulate_histogram);

    //std::cout << histogram_blue.size << std::endl;

    //std::cout << histogram_blue.elemSize() << " " << histogram_blue.elemSize1() << std::endl;

    cv::normalize(histogram_blue,histogram_blue,0.0,1.0,cv::NORM_MINMAX,-1,cv::Mat());
    cv::normalize(histogram_green,histogram_green,0.0,1.0,cv::NORM_MINMAX,-1,cv::Mat());
    cv::normalize(histogram_red,histogram_red,0.0,1.0,cv::NORM_MINMAX,-1,cv::Mat());

    
    cv::Mat output_image;
    output_image = cv::Mat::zeros(256,256*3,CV_8UC3);

    cv::Point line_begin;
    cv::Point line_end;

    line_begin.y = 255; 

    for(int i = 0; i < 256;i++){

        if((float)(histogram_blue.at<float>(i,0) <= 0.0f)){
            (histogram_blue.at<float>(i,0)) = 0.0f;
        }

        if((float)(histogram_green.at<float>(i,0) <= 0.0f)){
            (histogram_green.at<float>(i,0)) = 0.0f;
        }

        if((float)(histogram_red.at<float>(i,0) <= 0.0f)){
            (histogram_red.at<float>(i,0)) = 0.0f;
        }

    }

    /*
    for(int i = 0; i < 256;i++){

        int output_base_i = i * 3;

        line_begin.x = output_base_i;

        line_end.x = output_base_i;

        if((float)(histogram_blue.at<float>(i,0) <= 0.0f)){
            (histogram_blue.at<float>(i,0)) = 0.0f;
        }

        line_end.y = 255 - ((float)255 * (float)(histogram_blue.at<float>(i,0)));

        cv::line(output_image,line_begin,line_end,cv::Scalar(255,0,0),1);

        line_begin.x = output_base_i + 1;

        line_end.x = output_base_i + 1;

        if((float)(histogram_green.at<float>(i,0) <= 0.0f)){
            (histogram_green.at<float>(i,0)) = 0.0f;
        }

        line_end.y = 255 - ((float)255 * (float)(histogram_green.at<float>(i,0)));

        

        //std::cout << line_begin << " " << line_end << std::endl;

        cv::line(output_image,line_begin,line_end,cv::Scalar(0,255,0),1);

        line_begin.x = output_base_i + 2;

        line_end.x = output_base_i + 2;

        if((float)(histogram_red.at<float>(i,0) <= 0.0f)){
            (histogram_red.at<float>(i,0)) = 0.0f;
        }

        line_end.y = 255 - ((float)255 * (float)(histogram_red.at<float>(i,0)));

        cv::line(output_image,line_begin,line_end,cv::Scalar(0,0,255),1);
    }

    std::cout << std::endl;

    std::string histogram_window_name = "histogram_" + std::to_string(image_number);

    cv::namedWindow(histogram_window_name, cv::WINDOW_KEEPRATIO );
    cv::imshow(histogram_window_name, output_image);
    */
    

    histograms.push_back(histogram_blue);
    histograms.push_back(histogram_green);
    histograms.push_back(histogram_red);

    cv::merge(histograms,histogram_rgb);

    histograms.push_back(histogram_rgb);
}

double compare_histograms(std::vector<cv::Mat>& histograms_1,std::vector<cv::Mat>& histograms_2){

    int comp_method = 0;
    comp_method = cv::HISTCMP_BHATTACHARYYA;


    double avg_compare = 0.0;

    double sqr_acc = 0;

    double sum_1_0 = 0;
    double sum_2_0 = 0;

    double sum_1_1 = 0;
    double sum_2_1 = 0;

    double sum_1_2 = 0;
    double sum_2_2 = 0;
    //std::cout << histograms_1[0].cols << " " << histograms_1[0].rows << std::endl;
    for(int  i = 0; i < histograms_1[0].rows;i++){
        sum_1_0 += histograms_1[0].at<float>(i,0);
        sum_2_0 += histograms_2[0].at<float>(i,0);

        sum_1_1 += histograms_1[1].at<float>(i,0);
        sum_2_1 += histograms_2[1].at<float>(i,0);

        sum_1_2 += histograms_1[2].at<float>(i,0);
        sum_2_2 += histograms_2[2].at<float>(i,0);
    }

    double total_sum_1 = sum_1_0 + sum_1_1 + sum_1_2;
    double total_sum_2 = sum_2_0 + sum_2_1 + sum_2_2;

    double check_1_0 = 0;
    double check_2_0 = 0;

    double check_1_1 = 0;
    double check_2_1 = 0;

    double check_1_2 = 0;
    double check_2_2 = 0;

    double check_total_1;
    double check_total_2;

    for(int  i = 0; i < histograms_1[0].rows;i++){
        double p_i_0 = histograms_1[0].at<float>(i,0) / total_sum_1;
        check_1_0 += p_i_0;
        double q_i_0 = histograms_2[0].at<float>(i,0) / total_sum_2; 
        check_2_0 += q_i_0;

        double p_i_1 = histograms_1[1].at<float>(i,0) / total_sum_1;
        check_1_1 += p_i_1;
        double q_i_1 = histograms_2[1].at<float>(i,0) / total_sum_2; 
        check_2_1 += q_i_1;

        double p_i_2 = histograms_1[2].at<float>(i,0) / total_sum_1;
        check_1_2 += p_i_2;
        double q_i_2 = histograms_2[2].at<float>(i,0) / total_sum_2; 
        check_2_2 += q_i_2;

        double pi_qi = sqrt(p_i_0) - sqrt(q_i_0);
        sqr_acc += pi_qi * pi_qi;

        pi_qi = sqrt(p_i_1) - sqrt(q_i_1);
        sqr_acc += pi_qi * pi_qi;

        pi_qi = sqrt(p_i_2) - sqrt(q_i_2);
        sqr_acc += pi_qi * pi_qi;

    }

    sqr_acc = 1.0/(sqrt(2.0)) * sqrt(sqr_acc);

    double cmp_1 = 1.0 - sqr_acc;

    //for(int comp_method = 0; comp_method < 4; comp_method++ ){
    double compare_blue = 1.0 - cv::compareHist(histograms_1[0],histograms_2[0],comp_method);
    double compare_green = 1.0 - cv::compareHist(histograms_1[1],histograms_2[1],comp_method);
    double compare_red = 1.0 - cv::compareHist(histograms_1[2],histograms_2[2],comp_method);

    double compare_rgb = 1.0 - cv::compareHist(histograms_1[3],histograms_2[3],comp_method);

    avg_compare = (compare_blue + compare_red + compare_green) / 3.0f;

    //std::cout << "Histogram comparision with method " << comp_method << " : " << compare_blue << " " << compare_green << " " << compare_red << " " << avg_compare << " " << compare_rgb << std::endl;
    //}

    //std::cout << std::endl;
    return compare_rgb; //avg_compare;

}

void print_square_organoid_images(std::vector<cv::Mat*>& all_square_org_img){

    int num_cols = 26;

    int num_rows = 5;//floor((float)all_square_org_img.size() / (float)num_cols);

    int indv_img_size = CLUSTERING_VIEW_INDIVIDUAL_SQUARE_SIZE;

    cv::Size output_img_size = cv::Size(indv_img_size * num_cols, indv_img_size * num_rows);
    cv::Mat output_img = cv::Mat::zeros(output_img_size,CV_16UC3);

    std::vector<cv::Mat*> shuffled_org_img = all_square_org_img;

    auto rng = std::default_random_engine{};
    std::shuffle(shuffled_org_img.begin(),shuffled_org_img.end(),rng);
    //std::shuffle(shuffled_org_img.begin(),shuffled_org_img.end(),rng);

    int elem_size = ((all_square_org_img)[0])->elemSize1();

    for(int row = 0; row < num_rows;row++){
        for(int col = 0; col < num_cols;col++){

            int img_index = row * num_cols + col;

            if(img_index >= all_square_org_img.size()){
                goto print_img;
            }

            cv::Mat* current_img = shuffled_org_img[row * num_cols + col]; 

            for(int local_row = 0; local_row < indv_img_size; local_row++){
                for(int local_col = 0; local_col < indv_img_size; local_col++){

                    int row_in_output_img = row * indv_img_size + local_row;
                    int col_in_output_img = col * indv_img_size + local_col;

                    if(elem_size == 1){
                        output_img.at<cv::Vec3w>(row_in_output_img,col_in_output_img) = cv::Vec3w(250,250,250).mul(current_img->at<cv::Vec3b>(local_row,local_col));
                    }else if(elem_size == 2){
                        output_img.at<cv::Vec3w>(row_in_output_img,col_in_output_img) = cv::Vec3w(2,2,2).mul(current_img->at<cv::Vec3w>(local_row,local_col));

                    }
                }   
            }
        
        }   
    }

    print_img:

    std::string output_name = "all_organoids_";
    output_name += std::to_string(num_cols);
    output_name += "_";
    output_name += std::to_string(num_rows);
    output_name += ".png"; 


}

struct Recall_Cuts_Compare{
    bool operator()(const TPR_FPR_Tuple& lhs, const TPR_FPR_Tuple& rhs){

        if(lhs.recall_cuts == rhs.recall_cuts){
            return lhs.threshold < rhs.threshold;
        }

        return lhs.recall_cuts < rhs.recall_cuts;
    }
}Recall_Cuts_Compare;

struct Recall_Joins_Compare{
    bool operator()(const TPR_FPR_Tuple& lhs, const TPR_FPR_Tuple& rhs){

        if(lhs.recall_joins == rhs.recall_joins){
            return lhs.threshold < rhs.threshold;
        }

        return lhs.recall_joins < rhs.recall_joins;
    }
}Recall_Joins_Compare;

void draw_grid(cv::Mat& img, int num_intervalls, int border_offset){

    int interior_size = img.cols - 2 * border_offset;

    float intervall_size = (float)interior_size / (float)num_intervalls;

    for(int i = 0; i <= num_intervalls;i++){
        cv::line(img, cv::Point2i(border_offset,i * intervall_size + border_offset), cv::Point2i(border_offset + interior_size,i * intervall_size + border_offset),cv::Vec3f(0.5f,0.5f,0.5f));
        cv::line(img, cv::Point2i(i * intervall_size + border_offset, border_offset), cv::Point2i(i * intervall_size + border_offset, border_offset + interior_size),cv::Vec3f(0.5f,0.5f,0.5f));

    }
}

void display_roc_curves(std::vector<TPR_FPR_Tuple>& all_measurements, std::string base_window_name){

    cv::Vec3f zero_thresh_color(1.0f,0.0f,0.0f);
    cv::Vec3f middle_thresh_color(1.0f,1.0f,1.0f);
    cv::Vec3f one_thresh_color(0.0f,0.0f,1.0f);

    int border_offset = 48;

    int drawable_area_size = 512;

    int roc_image_size = drawable_area_size + 2 * border_offset;

    cv::Mat cuts_prec_recall_img = cv::Mat::zeros(roc_image_size,roc_image_size,CV_32FC3);
    draw_grid(cuts_prec_recall_img,10,border_offset);

    int baseline = 0;
    double font_scale = 1.0;
    std::string y_axis_string = "precision cuts";
    cv::Size bb_size = cv::getTextSize(y_axis_string, cv::FONT_HERSHEY_DUPLEX, font_scale,1,&baseline);

    cv::putText(cuts_prec_recall_img, //target image
    y_axis_string, //text
    cv::Point(roc_image_size / 2 - bb_size.width/ 2,bb_size.height / 2 + border_offset / 2), //top-left position
    cv::FONT_HERSHEY_DUPLEX,
    font_scale,
    CV_RGB(255, 255, 255), //font color
    1);

    cv::Point2f pc(cuts_prec_recall_img.cols/2., cuts_prec_recall_img.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pc, 90, 1.0);

    cv::warpAffine(cuts_prec_recall_img, cuts_prec_recall_img, r, cuts_prec_recall_img.size());


    std::string x_axis_string = "recall cuts";
    bb_size = cv::getTextSize(x_axis_string, cv::FONT_HERSHEY_DUPLEX, font_scale,1,&baseline);

    cv::putText(cuts_prec_recall_img, //target image
    x_axis_string, //text
    cv::Point(roc_image_size / 2 - bb_size.width/ 2, border_offset + drawable_area_size + bb_size.height / 2 + border_offset / 2), //top-left position
    cv::FONT_HERSHEY_DUPLEX,
    font_scale,
    CV_RGB(255, 255, 255), //font color
    1,cv::LINE_AA);

    std::sort(all_measurements.begin(),all_measurements.end(),Recall_Cuts_Compare);

    cv::Point2i prev_point(border_offset,border_offset);

    float increment = (float)roc_image_size / (float)all_measurements.size(); 

    //std::cout << all_measurements.size() << " " << increment << std::endl;
 
    for(int i = 0; i < all_measurements.size();i++){

    
        TPR_FPR_Tuple current_measurement = all_measurements[i];

        cv::Vec3f line_color;

        if(current_measurement.threshold < 0.5f){
            float t = current_measurement.threshold * 2.0f;

            line_color = t * middle_thresh_color + (1.0f - t) * zero_thresh_color;
        }else{
            float t = (current_measurement.threshold - 0.5f) * 2.0f;

            line_color = t * one_thresh_color + (1.0f - t) * middle_thresh_color;
        }


        cv::Point2i current_point(current_measurement.recall_cuts * drawable_area_size + border_offset,drawable_area_size - drawable_area_size * current_measurement.prec_cuts + border_offset);

        //std::cout << current_measurement.recall_cuts << " " << current_measurement.prec_cuts << " " << current_measurement.threshold << " " << current_point << std::endl;

        cv::line(cuts_prec_recall_img,prev_point,current_point,line_color,3);

        prev_point = current_point;

    }


    cv::Mat joins_prec_recall_img = cv::Mat::zeros(roc_image_size,roc_image_size,CV_32FC3);

    y_axis_string = "precision joins";
    bb_size = cv::getTextSize(y_axis_string, cv::FONT_HERSHEY_DUPLEX, font_scale,1,&baseline);

    cv::putText(joins_prec_recall_img, //target image
    y_axis_string, //text
    cv::Point(roc_image_size / 2 - bb_size.width/ 2,bb_size.height / 2 + border_offset / 2), //top-left position
    cv::FONT_HERSHEY_DUPLEX,
    font_scale,
    CV_RGB(1, 1, 1), //font color
    1);

    cv::warpAffine(joins_prec_recall_img, joins_prec_recall_img, r, joins_prec_recall_img.size());

    x_axis_string = "recall joins";
    bb_size = cv::getTextSize(x_axis_string, cv::FONT_HERSHEY_DUPLEX, font_scale,1,&baseline);

    cv::putText(joins_prec_recall_img, //target image
    x_axis_string, //text
    cv::Point(roc_image_size / 2 - bb_size.width/ 2, border_offset + drawable_area_size + bb_size.height / 2 + border_offset / 2), //top-left position
    cv::FONT_HERSHEY_DUPLEX,
    font_scale,
    CV_RGB(1, 1, 1), //font color
    1);

    draw_grid(joins_prec_recall_img,10,border_offset);

    std::sort(all_measurements.begin(),all_measurements.end(),Recall_Joins_Compare);

    prev_point = cv::Point2i(border_offset,border_offset);

    for(int i = 0; i < all_measurements.size();i++){

        TPR_FPR_Tuple current_measurement = all_measurements[i];

        cv::Vec3f line_color;

        if(current_measurement.threshold < 0.5f){
            float t = current_measurement.threshold * 2.0f;

            line_color = t * middle_thresh_color + (1.0f - t) * zero_thresh_color;
        }else{
            float t = (current_measurement.threshold - 0.5f) * 2.0f;

            line_color = t * one_thresh_color + (1.0f - t) * middle_thresh_color;
        }


        cv::Point2i current_point(current_measurement.recall_joins * drawable_area_size + border_offset,drawable_area_size - drawable_area_size * current_measurement.prec_joins + border_offset);

        //std::cout << current_measurement.recall_joins << " " << current_measurement.prec_joins << " " << current_measurement.threshold << " " << current_point << std::endl;

        cv::line(joins_prec_recall_img,prev_point,current_point,line_color,3);

        prev_point = current_point;

    }

    /*
    for(int row = 0; row < roc_image_size; row++){
        for(int col = 0; col < roc_image_size; col++){

            cuts_prec_recall_img.at<cv::Vec3f>(col,row) = cv::Vec3f((float)row/(float)roc_image_size,1.0f,(float)col/(float)roc_image_size);
        
        }    
    }
    */

    std::string cuts_window_name = base_window_name;
    cuts_window_name += " Cuts Recall X Prec Y";

    cv::namedWindow(cuts_window_name,cv::WINDOW_KEEPRATIO);
    cv::imshow(cuts_window_name,cuts_prec_recall_img);


    std::string joins_window_name = base_window_name;
    joins_window_name += " Joins Recall X Prec Y";

    cv::namedWindow(joins_window_name,cv::WINDOW_KEEPRATIO);
    cv::imshow(joins_window_name,joins_prec_recall_img);
}