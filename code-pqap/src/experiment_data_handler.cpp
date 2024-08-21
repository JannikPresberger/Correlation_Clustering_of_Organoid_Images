#include "experiment_data_handler.h"
#include "image_processing.h"

void load_experiment_data(Experiment_Data_Handler* exp_data_handler);
void process_single_organoid_image(std::filesystem::path img_path,Experiment_Data_Handler* exp_data_handler);
bool compare_experiment_identifiers(const Experiment_Identifier* exp_a, const Experiment_Identifier* exp_b);
bool compare_experiment_identifier_to_header(const Experiment_Identifier* exp, const Organoid_Image_Header* header);
Experiment_Replicate get_experiment_replicate_from_char(char replicate);
Organoid_Image_Experiment* init_experiment_data_from_header(const Organoid_Image_Header* header);
Experiment_Well* init_new_experiment_well_from_header(const Organoid_Image_Header* header);
void fill_plate_data_from_header(Organoid_Image_Experiment* exp, const Organoid_Image_Header* header);
void fill_well_data_from_header(Experiment_Plate* plate, const Organoid_Image_Header* header);
int get_channel_index_from_type(Channel_Type type, const Experiment_Well* well, unsigned int z_pos);
void load_and_combine_images(Experiment_Well* well, unsigned int z_pos);

Experiment_Data_Handler* init_experiment_data_handler(){

    Experiment_Data_Handler* new_exp_data_handler = new Experiment_Data_Handler;//(Experiment_Data_Handler*)malloc(sizeof(Experiment_Data_Handler));

    if(new_exp_data_handler == NULL){
        std::cerr << "Could not allocate enough memory for the experiment data handler" << std::endl;
        return NULL;
    }

    new_exp_data_handler->experiments.clear();

    load_experiment_data(new_exp_data_handler);


    return new_exp_data_handler;
}

void destroy_experiment_data_handler(Experiment_Data_Handler** handler){

    for(int i = 0; i < (*handler)->experiments.size();i++){

        for(int j = 0; j < (*handler)->experiments[i]->plates.size();j++){

            std::cout << "Platenumber: " << (*handler)->experiments[i]->plates[j]->plate_number << " deleted" << std::endl;

            for(int k = 0; k < (*handler)->experiments[i]->plates[j]->wells.size();k++){

                std::cout << (*handler)->experiments[i]->plates[j]->wells[k]->row << (*handler)->experiments[i]->plates[j]->wells[k]->col << " deleted" << std::endl;

                delete((*handler)->experiments[i]->plates[j]->wells[k]);
            }

            delete((*handler)->experiments[i]->plates[j]);

        }

        delete((*handler)->experiments[i]);

    }


    delete(*handler);

    *handler = nullptr;

}

void load_experiment_data(Experiment_Data_Handler* exp_data_handler){

    std::filesystem::path cur_path = get_data_folder_path();//std::filesystem::current_path(); 

    cur_path.append("images");
    cur_path.append("microscopy_images");

    for (const auto & entry : std::filesystem::directory_iterator(cur_path)){
        if(!entry.is_directory()){
            if(entry.path().filename().string().compare(".gitkeep") == 0){
                continue;
            }

            process_single_organoid_image(entry, exp_data_handler);
        }     
    }

    //merge_images_from_multiple_folders_into_single_folder(base_folder);

}

void process_single_organoid_image(std::filesystem::path img_path, Experiment_Data_Handler* exp_data_handler){


    //cv::Mat organoids_img = cv::imread(img_path);

    Organoid_Image_Header header = parse_org_img_filename(img_path);

    bool found_exiting_experiment_data = false;
    Organoid_Image_Experiment* experiment = nullptr;

    //experiment = (Organoid_Image_Experiment*)malloc(sizeof(Organoid_Image_Experiment));
    //exp_data_handler->experiments.push_back(experiment);

    for(int i = 0; i < exp_data_handler->experiments.size();i++){
        if(compare_experiment_identifier_to_header(&(exp_data_handler->experiments[i]->identifier),&header)){
            found_exiting_experiment_data = true;
            experiment = exp_data_handler->experiments[i];
            break;
        }

    }

    if(!found_exiting_experiment_data){
        experiment = init_experiment_data_from_header(&header);
        exp_data_handler->experiments.push_back(experiment);
    }

    fill_plate_data_from_header(experiment,&header);

}

bool compare_well_row_and_cols_to_header(const Experiment_Well* well_a, const Organoid_Image_Header* header){

    if(well_a->col != header->well_col_number){return false;}
    if(well_a->row != header->well_row_char){return false;}

    return true;

}

bool compare_well_row_and_cols(const Experiment_Well* well_a, const Experiment_Well* well_b){

    if(well_a->col != well_b->col){return false;}
    if(well_a->row != well_b->row){return false;}

    return true;

}

bool compare_experiment_identifiers(const Experiment_Identifier* exp_a, const Experiment_Identifier* exp_b){
    if(exp_a->year != exp_b->year){ return false;}
    if(exp_a->month != exp_b->month){ return false;}
    if(exp_a->day != exp_b->day){ return false;}
    if(exp_a->replicate != exp_b->replicate){ return false;}

    return true;
}

bool compare_experiment_identifier_to_header(const Experiment_Identifier* exp, const Organoid_Image_Header* header){
    if(exp->year != header->year_of_experiment){ return false;}
    if(exp->month != header->month_of_experiment){ return false;}
    if(exp->day != header->day_of_experiment){ return false;}
    if(exp->replicate != header->replicate){ return false;}

    return true;

}

Organoid_Image_Experiment* init_experiment_data_from_header(const Organoid_Image_Header* header){
    Organoid_Image_Experiment* exp = new Organoid_Image_Experiment;//(Organoid_Image_Experiment*)malloc(sizeof(Organoid_Image_Experiment));

    exp->identifier.day = header->day_of_experiment;
    exp->identifier.month = header->month_of_experiment;
    exp->identifier.year = header->year_of_experiment;
    exp->identifier.replicate = header->replicate;

    exp->plates.clear();

    return exp;

}

void fill_plate_data_from_header(Organoid_Image_Experiment* exp, const Organoid_Image_Header* header){

    bool found_existing_plate = false;
    Experiment_Plate* exp_plate = nullptr;

    for(int i = 0; i < exp->plates.size(); i++){
        if(header->plate_number == exp->plates[i]->plate_number){
            exp_plate = exp->plates[i];
            found_existing_plate = true;
            break;
        }

    }

    if(!found_existing_plate){
        exp_plate = new Experiment_Plate;//(Experiment_Plate*)malloc(sizeof(Experiment_Plate));
        exp_plate->plate_number = header->plate_number;
        exp_plate->wells.clear();
        exp->plates.push_back(exp_plate);
    }

    fill_well_data_from_header(exp_plate,header);

}

void fill_well_data_from_header(Experiment_Plate* plate, const Organoid_Image_Header* header){

    bool found_existing_well = false;
    Experiment_Well* exp_well = nullptr;

    for(int i = 0; i < plate->wells.size(); i++){

        if(compare_well_row_and_cols_to_header(plate->wells[i],header)){
            found_existing_well = true;
            exp_well = plate->wells[i];
            break;
        }

    }

    if(!found_existing_well){
        exp_well = init_new_experiment_well_from_header(header);
        plate->wells.push_back(exp_well);
    }

    int offset_z_pos = header->z_position - 1;

    if(offset_z_pos < IMAGES_PER_WELL_IN_Z && offset_z_pos >= 0){
        int channel_index = get_channel_index_from_type(header->channel,exp_well,offset_z_pos);

        exp_well->images_in_z[offset_z_pos][channel_index].img_file_path = header->full_file_path;
        exp_well->found_channels_in_z[offset_z_pos]++;

        if(exp_well->found_channels_in_z[offset_z_pos] == 4){
            load_and_combine_images(exp_well,offset_z_pos);

            unsigned int number_offset = 0;
            cv::Vec2i upper_left(0,0);

            subdivide_into_single_organoid_images(&(exp_well->combined_images[offset_z_pos]),header,false,number_offset,upper_left,upper_left);
        }

    }else{
        std::cerr << "Invalid Z Position of Image: " << offset_z_pos << std::endl;
    }

}

Experiment_Well* init_new_experiment_well_from_header(const Organoid_Image_Header* header){
    Experiment_Well* new_exp_well = new Experiment_Well;

    new_exp_well->col = header->well_col_number;
    new_exp_well->row = header->well_row_char;

    for(int i = 0; i < IMAGES_PER_WELL_IN_Z; i++){
        for(int k = 0; k < NUM_DIFFERENT_CHANNELS; k++){
            new_exp_well->images_in_z[i][k].well_index = new_exp_well;
            new_exp_well->images_in_z[i][k].z_pos = i;
            new_exp_well->images_in_z[i][k].img_file_path.clear();
            new_exp_well->images_in_z[i][k].channel = get_channel_type_from_int(k+1);
        }
        new_exp_well->found_channels_in_z[i] = 0;
    }

    return new_exp_well;
}

int get_channel_index_from_type(Channel_Type type, const Experiment_Well* well, unsigned int z_pos){

    if(z_pos >= IMAGES_PER_WELL_IN_Z){
        std::cerr << "Invalid z position: " << z_pos << " in function get_channel_index_from_type" << std::endl;
    }

    for(int i = 0; i < NUM_DIFFERENT_CHANNELS; i++){

        Channel_Type channel = well->images_in_z[z_pos][i].channel; 

        if(channel == type){
            return i;
        }
    }

    std::cerr << "Could not find matching type in function get_channel_index_from_type, " << type << std::endl;
    return -1;
}

void load_and_combine_images(Experiment_Well* well, unsigned int z_pos){

    std::filesystem::path blue_channel_path = well->images_in_z[z_pos][get_channel_index_from_type(DAPI_CHANNEL,well,z_pos)].img_file_path;
    std::filesystem::path green_channel_path = well->images_in_z[z_pos][get_channel_index_from_type(PDX1_GFP_CHANNEL,well,z_pos)].img_file_path;
    std::filesystem::path red_rfp_channel_path = well->images_in_z[z_pos][get_channel_index_from_type(NEUROG3_RFP_CHANNEL,well,z_pos)].img_file_path;
    std::filesystem::path red_af647_channel_path = well->images_in_z[z_pos][get_channel_index_from_type(Phalloidon_AF647_CHANNEL,well,z_pos)].img_file_path;


    cv::Mat channels[4];

    cv::Mat test;

    cv::Mat thres_test;
    cv::Mat thres_blue;

    cv::Mat sub_channel[2];


    channels[0] = cv::imread(blue_channel_path.string(),cv::IMREAD_ANYDEPTH);
    channels[1] = cv::imread(green_channel_path.string(),cv::IMREAD_ANYDEPTH);
    channels[2] = cv::imread(red_af647_channel_path.string(),cv::IMREAD_ANYDEPTH);
    channels[3] = cv::imread(red_rfp_channel_path.string(),cv::IMREAD_ANYDEPTH);


    cv::add(channels[2],channels[3],channels[2]);

    cv::Mat combined_no_rescale;
    combined_no_rescale.create(channels[0].rows, channels[0].cols,CV_16UC3);
    cv::merge(channels,3,combined_no_rescale);

    double global_min;

    double global_green_max;
    double global_blue_max;
    double global_red_max;
    double global_max;

    cv::Scalar blue_mean = cv::mean(channels[0]);
    cv::Scalar green_mean = cv::mean(channels[1]);
    cv::Scalar red_mean = cv::mean(channels[2]);
    
    cv::minMaxLoc(channels[0],&global_min,&global_blue_max);

    cv::minMaxLoc(channels[1],&global_min,&global_green_max);

    cv::minMaxLoc(channels[2],&global_min,&global_red_max);
    

    global_max = blue_mean.val[0];

    if(global_max < green_mean.val[0]){
        global_max = green_mean.val[0];
    }

    if(global_max < red_mean.val[0]){
        global_max = red_mean.val[0];
    }

    double blue_rescale_factor = global_max / blue_mean.val[0];
    double green_rescale_factor = global_max / green_mean.val[0];
    double red_rescale_factor = global_max / red_mean.val[0]; 

    channels[0] *= blue_rescale_factor;
    channels[1] *= green_rescale_factor;
    channels[2] *= red_rescale_factor;

    well->combined_images[z_pos].create(channels[0].rows, channels[0].cols,CV_8UC3);

    cv::merge(channels,3,well->combined_images[z_pos]);
}