#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>
#include "utils.h"
#include "cost_definitions.h"
#include <opencv2/highgui.hpp>
#include "clustering.h"

#define MATCHING_MATRIX_COLORGRADIENT_BIT_INDEX 0
#define MATCHING_MATRIX_NUMBER_DISPLAY_BIT_INDEX 1
#define MATCHING_MATRIX_SYMMETRIZE_BIT_INDEX 2
#define MATCHING_MATRIX_WRITE_TO_FILE_BIT_INDEX 3

typedef struct Cluster Cluster;

typedef struct Single_Cluster_Layout{
    int index_in_cluster_vector;
    int num_members;
    int rows;
    int cols;
    int start_row_in_overall_layout;
    int start_col_in_overall_layout;
}Single_Cluster_Layout;

void calculate_histogram(cv::Mat& image, std::vector<cv::Mat>& histograms, int image_number = 1);

double compare_histograms(std::vector<cv::Mat>& histograms_1,std::vector<cv::Mat>& histograms_2);

void update_selected_cells_after_image_order_switch(bool use_image_order);

void show_clustering(std::vector<cv::Mat*>* square_organoid_images, std::vector<Cluster>* all_clusters, std::vector<Matching_Result>* all_matching_results,std::vector<Image_Features_Pair>* all_feature_vectors, int individual_image_size, std::vector<Single_Cluster_Layout>* cluster_layouts, All_Cost_Parameters& cost_params, int window_number = PRIMARY_CLUSTERING_WINDOW_IMG_ID, std::vector<Cluster_Representative_Pair>* selected_cluster_representatives = nullptr);

void show_matching_matrix(std::vector<cv::Mat*>* square_organoid_images, std::vector<Matching_Result>* all_matching_results,std::vector<Image_Features_Pair>* all_feature_vectors, int individual_image_size, float cost_threshold, Matching_Visualization_Type viz_type, uint32_t flags, std::vector<int>* image_order, bool use_image_order, All_Cost_Parameters& cost_params);

void show_assignment(std::vector<int>* assignment, std::vector<Feature_Point_Data>* features,std::vector<Feature_Point_Data>* assigned_candidates, cv::Mat* feature_image, cv::Mat* candidate_image, cv::Point2i center_feature_image, cv::Point2i center_candidate_image, cv::Point2i offset_feature_image, cv::Point2i offset_candidate_image, std::filesystem::path path_to_feature_image, std::filesystem::path path_to_candidate_image);

cv::Point2i extract_features_from_single_organoid_image(std::filesystem::path single_organoid_img_path, std::vector<Feature_Point_Data>& output_feature_points, std::vector<Feature_Point_Data>& output_candidate_points, cv::Mat* mat_to_store_image_in, bool read_features_from_file, std::filesystem::path data_file_path, int img_number_offset);

void subdivide_into_single_organoid_images(cv::Mat* organoids_img, const Organoid_Image_Header* header, bool is_single_organoid_image, unsigned int& number_offset, cv::Vec2i root_image_size, cv::Vec2i cluster_start_point);

//template<typename T> void adaptive_mask_upscale(cv::Mat* downscaled_mask, int downscale_factor);

template<typename T>
void get_local_peaks(cv::Mat& input, cv::Mat* mask_input, std::vector<Feature_Point_Data>& feature_points,std::vector<Feature_Point_Data>& candidates, int local_search_dim, double& global_mean, double& global_std_dev, double& global_mean_foreground_only, double& global_std_dev_foreground_only, int& center_row, int& center_col, Channel_Type channel, int num_features_in_other_channels);

template<typename T>
void finalize_organoid_images(cv::Mat* input, cv::Mat& output, unsigned int num_contours, cv::Mat* original_img, const Organoid_Image_Header* header, unsigned int& number_offset, bool was_already_subdivided, cv::Vec2i root_image_size, cv::Vec2i cluster_start_point);

void reset_clustering_visualization_selection();

void mouse_call_back_function_matching_matrix_window(int event, int x, int y, int flags, void* userdata);

void mouse_call_back_function_clustering_window(int event, int x, int y, int flags, void* userdata);

void print_square_organoid_images(std::vector<cv::Mat*>& all_square_org_img);

void display_roc_curves(std::vector<TPR_FPR_Tuple>& all_measurements, std::string base_window_name);

template<typename T>
void adaptive_mask_upscale(cv::Mat* downscaled_mask, int downscale_factor){

    cv::namedWindow("Original Mask", cv::WINDOW_KEEPRATIO );
    cv::imshow("Original Mask", *downscaled_mask);

    //*downscaled_mask *= 1500;

    cv::waitKey(0);

    cv::Mat output = downscaled_mask->clone();

    cv::namedWindow("Upscaled Mask", cv::WINDOW_KEEPRATIO );
    cv::imshow("Upscaled Mask", output);


    std::cout << downscaled_mask->elemSize() << " " << downscaled_mask->elemSize1() << std::endl;
    std::cout << output.elemSize() << " " << output.elemSize1() << std::endl;

    cv::waitKey(0);

    std::cout << "started upscaling" << std::endl;
    for(int row = 0; row < downscaled_mask->rows; row++){
        for(int col = 0; col < downscaled_mask->cols; col++){

            T mask_value = downscaled_mask->at<T>(row,col);

            if(mask_value != 0){
                continue;
            }

            int intersection_in_row = 0;
            int intersection_in_col = 0;

            for(int i = 1; i < downscale_factor; i++){
                int local_row_up = std::max<int>(0,row - i);
                int local_row_down = std::min<int>(downscaled_mask->rows-1,row + i);
                int local_col_left = std::max<int>(0,col - i);
                int local_col_right = std::min<int>(downscaled_mask->cols-1,col + i);

                if(intersection_in_row == 0){
                    T value_up = downscaled_mask->at<T>(local_row_up,col);
                    T value_down = downscaled_mask->at<T>(local_row_down,col);

                    if(value_down != 0){
                        intersection_in_row = i;
                    }

                    if(value_up != 0){
                        intersection_in_row = i;
                    }

                }



                if(intersection_in_col == 0){
                    T value_left = downscaled_mask->at<T>(row,local_col_left);
                    T value_right = downscaled_mask->at<T>(row,local_col_right);

                    if(value_left != 0){
                        intersection_in_col = i;
                    }

                    if(value_right != 0){
                        intersection_in_col = i;
                    }

                }

                if(intersection_in_col != 0 && intersection_in_row != 0){
                    break;
                }
            }

            if((intersection_in_col + intersection_in_row) <= downscale_factor && intersection_in_col > 0 && intersection_in_row > 0){
                    output.at<T>(row,col) = T(-1);
                    std::cout << "write at: " << row << " " << col << std::endl;
            }
        
        }
    }

    std::cout << "finished upscaling" << std::endl;

    cv::namedWindow("Upscaled Mask", cv::WINDOW_KEEPRATIO );
    cv::imshow("Upscaled Mask", output);

    cv::waitKey(0);

}

typedef enum Corner_Types{
    UPPER_LEFT_CORNER,
    UPPER_RIGHT_CORNER,
    LOWER_LEFT_CORNER,
    LOWER_RIGHT_CORNER,
    NO_CORNER
}Corner_Types;

template<typename T>
cv::Mat adaptive_mask_upscale_edge_tracing(cv::Mat* downscaled_mask){
    /*
    *downscaled_mask *= 1500;

    cv::namedWindow("Original Mask", cv::WINDOW_KEEPRATIO );
    cv::imshow("Original Mask", *downscaled_mask);

    cv::waitKey(0);
    */

    cv::Mat original_mask;

    downscaled_mask->convertTo(original_mask,CV_8U);
    
    cv::cvtColor(original_mask,original_mask,cv::COLOR_GRAY2BGR);
    original_mask *= 50;
    cv::imwrite("mask.png",original_mask);

    cv::Mat output;

    assert(downscaled_mask->elemSize1() == sizeof(T));

    if(downscaled_mask->elemSize1() == 1){
        output = cv::Mat::zeros(downscaled_mask->size(),CV_8U);
    } else if(downscaled_mask->elemSize1() == 2){
        output = cv::Mat::zeros(downscaled_mask->size(),CV_16U);
    }
    //downscaled_mask->convertTo(output,CV_16U);

    //output *= 500;
    /*
    cv::namedWindow("Upscaled Mask", cv::WINDOW_KEEPRATIO );
    cv::imshow("Upscaled Mask", output);


    std::cout << downscaled_mask->elemSize() << " " << downscaled_mask->elemSize1() << std::endl;
    std::cout << output.elemSize() << " " << output.elemSize1() << std::endl;

    cv::waitKey(0);
    std::cout << "started upscaling " << std::endl;
    */
    for(int row = 0; row < downscaled_mask->rows; row++){
        for(int col = 0; col < downscaled_mask->cols; col++){

            T mask_value = downscaled_mask->at<T>(row,col);

            T corner_value = T(0);

            if(mask_value != 0 && output.at<T>(row,col) == 0){
                output.at<T>(row,col) = mask_value;
                //std::cout << mask_value << std::endl;
                //continue;
            }


            int local_row_up = std::max<int>(0,row - 1);
            int local_row_down = std::min<int>(downscaled_mask->rows-1,row + 1);
            int local_col_left = std::max<int>(0,col - 1);
            int local_col_right = std::min<int>(downscaled_mask->cols-1,col + 1);

            T upper_left = downscaled_mask->at<T>(local_row_up,local_col_left);// == target_mask_value;
            T upper_middle = downscaled_mask->at<T>(local_row_up,col);// == target_mask_value;
            T upper_right = downscaled_mask->at<T>(local_row_up,local_col_right);// == target_mask_value;

            T center_left = downscaled_mask->at<T>(row,local_col_left);// == target_mask_value;
            T center_right = downscaled_mask->at<T>(row,local_col_right);// == target_mask_value;

            T lower_left = downscaled_mask->at<T>(local_row_down,local_col_left);// == target_mask_value;
            T lower_middle = downscaled_mask->at<T>(local_row_down,col);// == target_mask_value;
            T lower_right = downscaled_mask->at<T>(local_row_down,local_col_right);// == target_mask_value;

            Corner_Types corner_type = NO_CORNER;

            if(upper_left && center_left == upper_left && upper_middle == upper_left && upper_left != mask_value){
                corner_type = UPPER_LEFT_CORNER;
                corner_value = upper_left;//downscaled_mask->at<T>(local_row_up,local_col_left);
            } else if(upper_right && center_right == upper_right && upper_middle == upper_right && upper_right != mask_value){
                corner_type = UPPER_RIGHT_CORNER;
                corner_value = upper_right;//downscaled_mask->at<T>(local_row_up,local_col_right);
                //std::cout << corner_value << std::endl;
            } else if(lower_left && center_left == lower_left && lower_middle == lower_left && lower_left != mask_value){
                corner_type = LOWER_LEFT_CORNER;
                corner_value = lower_left;//downscaled_mask->at<T>(local_row_down,local_col_left);
            } else if(lower_right && center_right == lower_right && lower_middle == lower_right && lower_right != mask_value){
                corner_type = LOWER_RIGHT_CORNER;
                corner_value = lower_right;//downscaled_mask->at<T>(local_row_down,local_col_right);
            }

            /*
            if(&& corner_value != mask_value){
                corner_type = NO_CORNER;
            }
            if(corner_value == 65535){
                continue;
            }
            */

            int trace_direction_row = 0;
            int trace_direction_col = 0;

            /*
            int max_value = (1 << (8 * 2))-1;
            std::cout << (int)max_value << std::endl;
            cv::Vec3s debug_color = cv::Vec3s(max_value,max_value,max_value);;
            */

           bool is_at_interface = false;

            switch(corner_type){

                case UPPER_LEFT_CORNER:
                    //debug_color = cv::Vec3s(65000,0,0);
                    trace_direction_col = 1;
                    trace_direction_row = 1;
                    break;

                case UPPER_RIGHT_CORNER:
                    //debug_color = cv::Vec3s(0,65000,0);
                    trace_direction_col = -1;
                    trace_direction_row = 1;
                    break;

                case LOWER_LEFT_CORNER:
                    //debug_color = cv::Vec3s(0,0,65000);
                    trace_direction_col = 1;
                    trace_direction_row = -1;
                    //std::cout << "lower left corner" << std::endl;
                    break;

                case LOWER_RIGHT_CORNER:
                    //debug_color = cv::Vec3s(65000,65000,0);
                    trace_direction_col = -1;
                    trace_direction_row = -1;

                    break;

                case NO_CORNER:
                    break;

            }
            if(corner_type != NO_CORNER){
                original_mask.at<cv::Vec3b>(row,col) = cv::Vec3b(0,0,250);

                if(mask_value != 0){
                    is_at_interface = true;
                }

                int col_start = col - trace_direction_col;
                int row_start = row - trace_direction_row;

                int col_line_pos = 1;
                int row_line_pos = 1;

                while(col_line_pos < downscaled_mask->cols){
                    int local_col_index = col + col_line_pos * trace_direction_col;

                    if(local_col_index < 0 || local_col_index >= downscaled_mask->cols){
                        break;
                    }

                    T mask_value_inner = downscaled_mask->at<T>(row_start,local_col_index);
                    T mask_value_outer = downscaled_mask->at<T>(row,local_col_index);

                    if(mask_value_inner == corner_value && mask_value_outer == mask_value){
                        //output.at<cv::Vec3s>(row,local_col_index) = debug_color;
                        //original_mask.at<cv::Vec3b>(row,local_col_index) = cv::Vec3b(250,0,0);
                        col_line_pos++;
                    }else{
                        break;
                    }
                }

                while(row_line_pos < downscaled_mask->rows){
                    int local_row_index = row + row_line_pos * trace_direction_row;

                    if(local_row_index < 0 || local_row_index >= downscaled_mask->rows){
                        break;
                    }

                    T mask_value_inner = downscaled_mask->at<T>(local_row_index,col_start);
                    T mask_value_outer = downscaled_mask->at<T>(local_row_index,col);


                    if(mask_value_inner == corner_value && mask_value_outer == mask_value){
                        //output.at<cv::Vec3s>(local_row_index,col) = debug_color;
                        //original_mask.at<cv::Vec3b>(local_row_index,col) = cv::Vec3b(0,250,0);
                        row_line_pos++;
                    }else{
                        break;
                    }
                }

                //std::vector<std::vector<cv::Point2i>> triangle_vector;

                //std::vector<cv::Point2i> triangle;

                cv::Vec2f dividing_line_start(row,col + col_line_pos * trace_direction_col);
                cv::Vec2f dividing_line_end(row + row_line_pos * trace_direction_row,col);
                cv::Vec2f origin(row,col);

                cv::Vec2f dividing_line_dir = cv::normalize(dividing_line_end - dividing_line_start);
                cv::Vec2f dividing_line_normal(dividing_line_dir[1],-dividing_line_dir[0]);

                cv::Vec2f start_to_origin = origin - dividing_line_start;
                int origin_side = get_sign(dividing_line_normal.dot(start_to_origin));

                //std::cout << origin_side << std::endl;

                //triangle.push_back(dividing_line_end);
                //triangle.push_back(origin);
                //triangle.push_back(dividing_line_start);

                //triangle_vector.push_back(triangle);

                if(is_at_interface){
                    dividing_line_start[1] = col + (col_line_pos / 2) * trace_direction_col;
                    //is we are at an interface between two regions we want the dividing line not to start at the end of the traced edge, but in the middle of the traced edge
                }

                for(int local_region_row = 0; local_region_row < row_line_pos;local_region_row++){
                    for(int local_region_col = 0; local_region_col < col_line_pos;local_region_col++){
                        cv::Vec2f current_pixel_pos(row + local_region_row * trace_direction_row,col + local_region_col * trace_direction_col);

                        cv::Vec2f start_to_current_pixel = current_pixel_pos - dividing_line_start;
                        int current_sign = get_sign(dividing_line_normal.dot(start_to_current_pixel));
                        if(current_sign == origin_side){
                            output.at<T>(row + local_region_row * trace_direction_row,col + local_region_col * trace_direction_col) = corner_value;
                            original_mask.at<cv::Vec3b>(row + local_region_row * trace_direction_row,col + local_region_col * trace_direction_col) = cv::Vec3b(0,250,0);
                        }else if(is_at_interface){
                            //output.at<T>(row + local_region_row * trace_direction_row,col + local_region_col * trace_direction_col) = mask_value;
                        }
                    
                    }
                }
                
                //cv::drawContours(output,triangle_vector,-1,cv::Scalar(65000,65000,65000));

                //cv::line(output,dividing_line_start,dividing_line_end,cv::Scalar(65000,65000,65000));
                //for(int i = 1; i < 5;i++){
                    //output.at<cv::Vec3s>(row,col + i * trace_direction_col) = debug_color;
                    //output.at<cv::Vec3s>(row + i * trace_direction_row,col) = debug_color;

                //}
            }


        
        }
    }
    /*
    std::cout << "finished upscaling" << std::endl;

    //output *= 1500;

    cv::namedWindow("Upscaled Mask", cv::WINDOW_KEEPRATIO );
    cv::imshow("Upscaled Mask", output);

    cv::waitKey(0);
    */

    cv::imwrite("mask_with_added_sections.png",original_mask);

    cv::Mat smoothed_out = output.clone();
    smoothed_out *= 1500;

    cv::imwrite("smoothed_mask.png",smoothed_out);

    return output;

}