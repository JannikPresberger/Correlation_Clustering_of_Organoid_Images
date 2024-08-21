#pragma once

#include "global_data_types.h"

#ifdef _WIN64
#include <winsock.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#ifdef __unix__
#include <unistd.h>
#endif

void init_mpi(int argc, char** argv);

int get_num_mpi_processes();

int get_mpi_process_rank();

void mpi_distribute_cost_parameters(All_Cost_Parameters* cost_params, bool& learning_loop_is_finished);

void mpi_distribute_task_vectors(std::vector<Task_Vector>& all_task_vectors, Task_Vector& active_task_vector_of_process, All_Cost_Parameters* cost_params);

void mpi_gather_matching_results(std::vector<Matching_Result>& all_matching_results, std::vector<Op_Numbers_And_Runtime>& execution_times, Additional_Viz_Data_Input* viz_input = nullptr);

void finalize_and_quit_mpi();