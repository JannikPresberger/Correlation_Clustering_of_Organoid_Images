#pragma once

#include "utils.h"
#include <vector>
#include <opencv2/opencv.hpp>

#define IMAGES_PER_WELL_IN_Z 4
#define NUM_DIFFERENT_CHANNELS 4

typedef struct Experiment_Well Experiment_Well;

typedef struct Organoid_Image{
    Experiment_Well* well_index;
    unsigned short z_pos;
    std::filesystem::path img_file_path;
    Channel_Type channel;
    //cv::Mat image_data;
}Organoid_Image;

typedef struct Experiment_Well{
    char row;
    unsigned short col;
    Organoid_Image images_in_z[IMAGES_PER_WELL_IN_Z][NUM_DIFFERENT_CHANNELS];
    unsigned short found_channels_in_z[IMAGES_PER_WELL_IN_Z];
    cv::Mat combined_images[IMAGES_PER_WELL_IN_Z];
}Experiment_Well;

typedef struct Experiment_Identifier{
    unsigned short year;
    unsigned short month;
    unsigned short day;
    Experiment_Replicate replicate;

}Experiment_Identifier;



typedef struct Experiment_Plate{
    unsigned short plate_number;
    Experiment_Identifier* parent_experiment;
    std::vector<Experiment_Well*> wells;
}Experiment_Plate;

typedef struct Organoid_Image_Experiment{
    Experiment_Identifier identifier;
    std::vector<Experiment_Plate*> plates;

}Organoid_Image_Experiment;

typedef struct Experiment_Data_Handler{
    std::vector<Organoid_Image_Experiment*> experiments;
}Experiment_Data_Handler; 

Experiment_Data_Handler* init_experiment_data_handler();

void destroy_experiment_data_handler(Experiment_Data_Handler** handler);