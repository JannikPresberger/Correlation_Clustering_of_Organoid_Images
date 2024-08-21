#include <iostream>
#include "image_processing.h"
#include "experiment_data_handler.h"
#include "python_handler.h"
//#include "cost_definitions.h"
#include "task_handler.h"

#ifdef _WIN64
#include <Windows.h>
#include <winsock.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#ifdef __unix__
#include <unistd.h>
#endif

//#include "gurobi_c.h"
#include "solvers.h"
#include "memory_manager.h"
#include <chrono>
//using namespace cv;
Python_Handler* global_python_handler;

bool global_activate_debug_prints = false;

int main(int argc, char** argv) {
    //std::cout << "Hello, world!\n";

    #ifdef _WIN64
        SetConsoleOutputCP(CP_UTF8);
        setvbuf(stdout, nullptr, _IOFBF, 1000);

    #endif // DEBUG

    Input_Arguments input_arguments;

    All_Cost_Parameters initial_cost_params{COST_DOUBLE_ASSIGNMENT,
                                    COST_MISMATCHING_CHANNELS,
                                    DEFAULT_COLOR_COST_OFFSET,
                                    DEFAULT_DIST_COST_OFFSET,
                                    DEFAULT_ANGLE_COST_OFFSET,
                                    DEFAULT_COLOR_TO_DIST_WEIGHT,
                                    DEFAULT_UNARY_TO_QUADR_WEIGHT};

    read_input_argument(input_arguments,argc,argv, initial_cost_params);


    if(input_arguments.perform_dry_run || input_arguments.print_help_info){
        return 0;
    }

    std::filesystem::path cur_path = std::filesystem::current_path(); 
    std::filesystem::path base_folder = cur_path.parent_path();

    std::filesystem::path python_scripts_folder_path = base_folder;

    python_scripts_folder_path = find_python_scripts_folder(base_folder);
    global_python_handler = init_python_handler(python_scripts_folder_path);


    if(input_arguments.segment_organoids){
        Experiment_Data_Handler* exp_data_handler = init_experiment_data_handler();
        destroy_experiment_data_handler(&exp_data_handler);

    }else{
        std::filesystem::path img_dataset_path = get_data_folder_path();
        img_dataset_path.append("images");
        img_dataset_path.append("image_sets");
        img_dataset_path.append(input_arguments.image_set_name);

        evaluate_all_possible_matchings(img_dataset_path, input_arguments);
    }


    delete(input_arguments.search_order);

    destroy_python_handler(&global_python_handler);

    return 0;
}
