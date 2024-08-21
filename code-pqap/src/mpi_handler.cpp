#include "mpi_handler.h"

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
//
#include "mpi.h"

typedef struct MPI_Info_Of_Process{
    int size;
    int rank;
}MPI_Info_Of_Process;

MPI_Datatype mpi_cost_parameters;
MPI_Datatype mpi_matching_task;
MPI_Datatype mpi_matching_result;
MPI_Datatype mpi_execution_time_measurement;

MPI_Info_Of_Process mpi_info_of_current_process; 

void init_mpi(int argc, char** argv){

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_info_of_current_process.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_info_of_current_process.rank);

    //const int num_elems_in_mpi_com_matching_task = NUM_ELEMS_MPI_COMM_MATCHING_TASK;
    int block_lengths_matching_task[NUM_ELEMS_MPI_COMM_MATCHING_TASK] = {1,1,1};
    MPI_Datatype types_matching_task[NUM_ELEMS_MPI_COMM_MATCHING_TASK] = {MPI_INT,MPI_INT,MPI_DOUBLE};

    MPI_Aint offsets_matching_task[NUM_ELEMS_MPI_COMM_MATCHING_TASK];
    offsets_matching_task[0] = offsetof(MPI_Comm_Matching_Task,id_1);
    offsets_matching_task[1] = offsetof(MPI_Comm_Matching_Task,id_2);
    offsets_matching_task[2] = offsetof(MPI_Comm_Matching_Task,runtime_estimation);

    MPI_Type_create_struct(NUM_ELEMS_MPI_COMM_MATCHING_TASK,block_lengths_matching_task,offsets_matching_task,types_matching_task,&mpi_matching_task);
    MPI_Type_commit(&mpi_matching_task);


    int block_lengths_matching_result[NUM_ELEMS_MPI_COMM_MATCHING_RESULT] = {1,1,1,1,1,1,1,1,1,1};
    MPI_Datatype types_matching_result[NUM_ELEMS_MPI_COMM_MATCHING_RESULT] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_INT,MPI_INT,MPI_INT,MPI_INT,MPI_INT,MPI_INT};

    MPI_Aint offsets_matching_result[NUM_ELEMS_MPI_COMM_MATCHING_RESULT];
    offsets_matching_result[0] = offsetof(MPI_Comm_Matching_Result,rel_quadr_cost_optimal);
    offsets_matching_result[1] = offsetof(MPI_Comm_Matching_Result,rel_quadr_cost);
    offsets_matching_result[2] = offsetof(MPI_Comm_Matching_Result,linear_cost_per_feature);
    offsets_matching_result[3] = offsetof(MPI_Comm_Matching_Result,linear_cost_per_candidate);
    offsets_matching_result[4] = offsetof(MPI_Comm_Matching_Result,id_1);
    offsets_matching_result[5] = offsetof(MPI_Comm_Matching_Result,id_2);
    offsets_matching_result[6] = offsetof(MPI_Comm_Matching_Result,set_id_1);
    offsets_matching_result[7] = offsetof(MPI_Comm_Matching_Result,set_id_2);
    offsets_matching_result[8] = offsetof(MPI_Comm_Matching_Result,num_elems_in_assignment);
    offsets_matching_result[9] = offsetof(MPI_Comm_Matching_Result,offset_into_assignment_array);

    MPI_Type_create_struct(NUM_ELEMS_MPI_COMM_MATCHING_RESULT,block_lengths_matching_result,offsets_matching_result,types_matching_result,&mpi_matching_result);
    MPI_Type_commit(&mpi_matching_result);


    int block_lengths_execution_time_measurement[NUM_ELEMS_MPI_COMM_EXECUTION_TIME_MEASUREMENT] = {1,1,1};
    MPI_Datatype types_execution_time_measurement[NUM_ELEMS_MPI_COMM_EXECUTION_TIME_MEASUREMENT] = {MPI_INT,MPI_INT,MPI_DOUBLE};

    MPI_Aint offsets_execution_time_measurement[NUM_ELEMS_MPI_COMM_EXECUTION_TIME_MEASUREMENT];

    offsets_execution_time_measurement[0] = offsetof(MPI_Comm_Execution_Time_Measurement,num_features);
    offsets_execution_time_measurement[1] = offsetof(MPI_Comm_Execution_Time_Measurement,num_candidates);
    offsets_execution_time_measurement[2] = offsetof(MPI_Comm_Execution_Time_Measurement,execution_time);

    MPI_Type_create_struct(NUM_ELEMS_MPI_COMM_EXECUTION_TIME_MEASUREMENT,block_lengths_execution_time_measurement,offsets_execution_time_measurement,types_execution_time_measurement,&mpi_execution_time_measurement);
    MPI_Type_commit(&mpi_execution_time_measurement);


    int block_lengths_cost_parameters[NUM_ELEMS_MPI_COMM_COST_PARAMETERS] = {1,1,1,1,1,1};
    MPI_Datatype types_cost_parameters[NUM_ELEMS_MPI_COMM_COST_PARAMETERS] = {MPI_INT,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};

    MPI_Aint offsets_cost_parameters[NUM_ELEMS_MPI_COMM_COST_PARAMETERS];
    offsets_cost_parameters[0] = offsetof(MPI_Comm_Cost_Parameters,learning_loop_is_finished);
    offsets_cost_parameters[1] = offsetof(MPI_Comm_Cost_Parameters,color_cost_offset);
    offsets_cost_parameters[2] = offsetof(MPI_Comm_Cost_Parameters,dist_cost_offset);
    offsets_cost_parameters[3] = offsetof(MPI_Comm_Cost_Parameters,angle_cost_offset);
    offsets_cost_parameters[4] = offsetof(MPI_Comm_Cost_Parameters,color_to_dist_weight);
    offsets_cost_parameters[5] = offsetof(MPI_Comm_Cost_Parameters,unary_to_quadr_weight);

    MPI_Type_create_struct(NUM_ELEMS_MPI_COMM_COST_PARAMETERS,block_lengths_cost_parameters,offsets_cost_parameters,types_cost_parameters,&mpi_cost_parameters);
    MPI_Type_commit(&mpi_cost_parameters);


}

void finalize_and_quit_mpi(){

    MPI_Finalize();

    if(mpi_info_of_current_process.rank != 0){
        return;
    }
}

int get_num_mpi_processes(){

    return mpi_info_of_current_process.size;
}

int get_mpi_process_rank(){

    return mpi_info_of_current_process.rank;
}

void mpi_distribute_task_vectors(std::vector<Task_Vector>& all_task_vectors, Task_Vector& active_task_vector_of_process, All_Cost_Parameters* cost_params){

    active_task_vector_of_process.tasks->clear();
    active_task_vector_of_process.total_runtime_estimation = 0.0;

    int i,rank, size, tag=99;
    char machine_name[256];
    MPI_Status status;

    gethostname(machine_name, 255);

    rank = mpi_info_of_current_process.rank;
    size = mpi_info_of_current_process.size;

    if(rank == 0) {
        //printf ("master process %d running on %s\n",rank,machine_name);
        std::vector<std::vector<MPI_Comm_Matching_Result>*> all_matching_results;

        for (i = 1; i < size; i++) {
            int num_tasks_to_process = all_task_vectors[i].tasks->size();
            MPI_Send(&num_tasks_to_process, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
        }


        for (i = 1; i < size; i++) {
            std::vector<MPI_Comm_Matching_Task> matching_tasks;

            for(int j = 0; j < all_task_vectors[i].tasks->size();j++){
                Matching_Calculation_Task current_task = (*(all_task_vectors[i].tasks))[j];

                MPI_Comm_Matching_Task new_matching_task;

                new_matching_task.id_1 = current_task.id_1;
                new_matching_task.id_2 = current_task.id_2;
                new_matching_task.runtime_estimation = current_task.runtime_estimation;

                matching_tasks.push_back(new_matching_task);
            }

            MPI_Send(matching_tasks.data(), matching_tasks.size(), mpi_matching_task, i, tag, MPI_COMM_WORLD);  
        }

        *(active_task_vector_of_process.tasks) = *(all_task_vectors[0].tasks);
        active_task_vector_of_process.total_runtime_estimation = all_task_vectors[0].total_runtime_estimation;

    } else {
        //sprintf(message, "Hello world from process %d running on %s",rank,machine_name);
        int tasks_to_process = 0;
        MPI_Recv(&tasks_to_process, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        //printf("process = %d on %s should process: %d tasks\n", rank,machine_name ,tasks_to_process);


        std::vector<MPI_Comm_Matching_Task> received_tasks;
        received_tasks.resize(tasks_to_process);

        //MPI_Comm_Matching_Task new_matching_task;
        MPI_Recv(received_tasks.data(), tasks_to_process, mpi_matching_task, 0, tag, MPI_COMM_WORLD, &status);


        for(int i = 0; i < tasks_to_process;i++){
            MPI_Comm_Matching_Task current_task = received_tasks[i];

            Matching_Calculation_Task new_matching_calc_task;

            new_matching_calc_task.id_1 = current_task.id_1;
            new_matching_calc_task.id_2 = current_task.id_2;
            new_matching_calc_task.runtime_estimation = current_task.runtime_estimation;
            new_matching_calc_task.viz_data = nullptr;
            new_matching_calc_task.index_features_in_viz_data = -1;
            new_matching_calc_task.index_candidates_in_viz_data = -1;

            new_matching_calc_task.cost_params = cost_params;

            active_task_vector_of_process.tasks->push_back(new_matching_calc_task);
            active_task_vector_of_process.total_runtime_estimation += current_task.runtime_estimation;

            //printf("process = %d matching_task: %d %d %f\n", rank, current_task.id_1,current_task.id_2,current_task.runtime_estimation);


        }
        //printf("process = %d finished receiving tasks\n", rank);

    }

    //printf ("process %d running on %s has %d tasks to process\n",rank,machine_name,active_task_vector_of_process.tasks->size());

}

void mpi_distribute_cost_parameters(All_Cost_Parameters* cost_params, bool& learning_loop_is_finished){

    int i,rank, size, tag=99;
    char machine_name[256];
    MPI_Status status;

    gethostname(machine_name, 255);

    rank = mpi_info_of_current_process.rank;
    size = mpi_info_of_current_process.size;

    //printf ("process %d running on %s begins mpi_distribute_cost_parameters\n",rank,machine_name);

    if(rank == 0){
        MPI_Comm_Cost_Parameters params_to_send;
        params_to_send.learning_loop_is_finished = learning_loop_is_finished;
        //std::cout << "master is sending: " << params_to_send.learning_loop_is_finished << " original: " << learning_loop_is_finished << std::endl;

        params_to_send.color_cost_offset = cost_params->color_offset;
        params_to_send.dist_cost_offset = cost_params->dist_offset;
        params_to_send.angle_cost_offset = cost_params->angle_offset;
        params_to_send.color_to_dist_weight = cost_params->color_to_dist_weight;
        params_to_send.unary_to_quadr_weight = cost_params->unary_to_to_quadr_weight;

        for (i = 1; i < size; i++) {
            MPI_Send(&params_to_send,1,mpi_cost_parameters,i,tag,MPI_COMM_WORLD);
        }


    }else{

        MPI_Comm_Cost_Parameters received_params;

        MPI_Recv(&received_params,1,mpi_cost_parameters,0,tag,MPI_COMM_WORLD,&status);

        cost_params->color_offset = received_params.color_cost_offset;
        cost_params->dist_offset = received_params.dist_cost_offset;
        cost_params->angle_offset = received_params.angle_cost_offset;
        cost_params->color_to_dist_weight = received_params.color_to_dist_weight;
        cost_params->unary_to_to_quadr_weight = received_params.unary_to_quadr_weight;
        cost_params->cost_double_assignment = COST_DOUBLE_ASSIGNMENT;
        cost_params->cost_mismatching_channels = COST_MISMATCHING_CHANNELS;


        learning_loop_is_finished = received_params.learning_loop_is_finished;
        //std::cout << rank << " received: " << received_params.learning_loop_is_finished << " new: " << learning_loop_is_finished << std::endl;

    }

}

void mpi_gather_matching_results(std::vector<Matching_Result>& all_matching_results, std::vector<Op_Numbers_And_Runtime>& execution_times, Additional_Viz_Data_Input* viz_input){

    int i = 0,rank, size, tag=99;
    char machine_name[256];
    MPI_Status status;

    gethostname(machine_name, 255);

    rank = mpi_info_of_current_process.rank;
    size = mpi_info_of_current_process.size;

    if(rank == 0) {
        std::vector<int> num_matching_results_per_process;

        for (i = 1; i < size; i++) {
            int num_of_matching_results = 0;

            MPI_Recv(&num_of_matching_results, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
            //std::cout << rank << " will receive: " << num_of_matching_results << " matching results from process: " << i << std::endl;

            num_matching_results_per_process.push_back(num_of_matching_results); 
        }

        std::vector<std::vector<MPI_Comm_Matching_Result>*> mpi_all_matching_results;

        for (i = 1; i < size; i++) {

            std::vector<MPI_Comm_Matching_Result>* new_matching_result_vector = new std::vector<MPI_Comm_Matching_Result>;
            new_matching_result_vector->resize(num_matching_results_per_process[i-1]);

            //std::cout << "allocated vector with size: " << num_matching_results_per_process[i-1] << " for process: " << i << std::endl; 
            //new_matching_result_vector->resize(i);
            mpi_all_matching_results.push_back(new_matching_result_vector);  
        }

        for (i = 1; i < size; i++) {
            //std::cout << "master expects to receive " << num_matching_results_per_process[i-1] << " results from process " << i << std::endl; 

            int assignment_flat_array_size = 0;
            MPI_Recv(&assignment_flat_array_size, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);

            std::vector<int> all_assignments;
            all_assignments.resize(assignment_flat_array_size);

            MPI_Recv(all_assignments.data(),assignment_flat_array_size,MPI_INT,i,tag,MPI_COMM_WORLD,&status);

            //std::cout << "received all assignment data from: " << i << std::endl;

            MPI_Recv(mpi_all_matching_results[i-1]->data(), num_matching_results_per_process[i-1], mpi_matching_result, i, tag, MPI_COMM_WORLD, &status);

            //std::cout << "received all matching data from: " << i << std::endl;

            for(int j = 0; j < num_matching_results_per_process[i-1]; j++){
                MPI_Comm_Matching_Result cur_mr = mpi_all_matching_results[i-1]->at(j);
                //printf("master received from process %d matching result: %f %f %f %f %d %d %d %d \n", i, cur_mr.rel_quadr_cost_optimal,cur_mr.rel_quadr_cost,cur_mr.linear_cost_per_feature,cur_mr.linear_cost_per_candidate,cur_mr.id_1,cur_mr.id_2,cur_mr.num_elems_in_assignment,cur_mr.offset_into_assignment_array);
                //std::cout << "master received from process: " << i << " matching result: " << cur_mr.rel_quadr_cost_optimal << " " << cur_mr.rel_quadr_cost << " " << cur_mr.linear_cost_per_feature << " " << cur_mr.linear_cost_per_candidate  << " " << cur_mr.id_1  << " " << cur_mr.id_2 << " " << cur_mr.num_elems_in_assignment << " " << cur_mr.offset_into_assignment_array << std::endl;
                
                Matching_Result new_mr;
                new_mr.id_1 = cur_mr.id_1;
                new_mr.id_2 = cur_mr.id_2;
                new_mr.set_id_1 = cur_mr.set_id_1;
                new_mr.set_id_2 = cur_mr.set_id_2;
                new_mr.linear_cost_per_candidate = cur_mr.linear_cost_per_candidate;
                new_mr.linear_cost_per_feature = cur_mr.linear_cost_per_feature;
                new_mr.rel_quadr_cost = cur_mr.rel_quadr_cost;
                new_mr.rel_quadr_cost_optimal = cur_mr.rel_quadr_cost_optimal;
                new_mr.additional_viz_data_id1_to_id2 = nullptr;
                new_mr.additional_viz_data_id2_to_id1 = nullptr;

                new_mr.assignment = new std::vector<int>;

                //std::cout << "before push assignment" << std::endl;
                for(int k = 0; k < cur_mr.num_elems_in_assignment;k++){
                    new_mr.assignment->push_back(all_assignments[k + cur_mr.offset_into_assignment_array]);
                    //std::cout << all_assignments[k + cur_mr.offset_into_assignment_array] << " ";
            
                }
                //std::cout << "after push assignment"<< std::endl;

                if(viz_input != nullptr){
                    int index_of_image_1 = 0;
                    int index_of_image_2 = 0;

                    //std::cout << "viz_input was not null" << std::endl;  

                    for(int j = 0; j < (*(viz_input->all_feature_vectors)).size();j++){
                        Image_Features_Pair current_ifp = (*(viz_input->all_feature_vectors))[j];
                        

                        if(current_ifp.image_number == new_mr.id_1){
                            index_of_image_1 = j;
                        }

                        if(current_ifp.image_number == new_mr.id_2){
                            index_of_image_2 = j;
                        }
                    }

                    new_mr.additional_viz_data_id1_to_id2 = new Matching_Result_Additional_Viz_Data;
                    new_mr.additional_viz_data_id2_to_id1 = new Matching_Result_Additional_Viz_Data;

                    new_mr.additional_viz_data_id1_to_id2->center_feature_image = (*(viz_input->sorted_image_centers))[index_of_image_1];
                    new_mr.additional_viz_data_id1_to_id2->center_candidate_image = (*(viz_input->sorted_image_centers))[index_of_image_2];

                    new_mr.additional_viz_data_id1_to_id2->offset_feature_image = (*(viz_input->sorted_offsets))[index_of_image_1];
                    new_mr.additional_viz_data_id1_to_id2->offset_candidate_image = (*(viz_input->sorted_offsets))[index_of_image_2];

                    new_mr.additional_viz_data_id1_to_id2->path_to_feature_image = (*(viz_input->sorted_image_paths))[index_of_image_1];
                    new_mr.additional_viz_data_id1_to_id2->path_to_candidate_image = (*(viz_input->sorted_image_paths))[index_of_image_2];

                    new_mr.additional_viz_data_id1_to_id2->feature_image = nullptr;//new cv::Mat;
                    //*new_mr.additional_viz_data_id1_to_id2->feature_image = (*(viz_input->sorted_square_organoid_images))[index_of_image_1]->clone();

                    new_mr.additional_viz_data_id1_to_id2->candidate_image = nullptr;//new cv::Mat;
                    //*new_mr.additional_viz_data_id1_to_id2->candidate_image = (*(viz_input->sorted_square_organoid_images))[index_of_image_2]->clone();

                    new_mr.additional_viz_data_id1_to_id2->features = (*(viz_input->all_feature_vectors))[index_of_image_1].features;
                    new_mr.additional_viz_data_id1_to_id2->assigned_candidates = (*(viz_input->all_feature_vectors))[index_of_image_2].candidates;


                    new_mr.additional_viz_data_id2_to_id1->center_feature_image = (*(viz_input->sorted_image_centers))[index_of_image_2];
                    new_mr.additional_viz_data_id2_to_id1->center_candidate_image = (*(viz_input->sorted_image_centers))[index_of_image_1];

                    new_mr.additional_viz_data_id2_to_id1->offset_feature_image = (*(viz_input->sorted_offsets))[index_of_image_2];
                    new_mr.additional_viz_data_id2_to_id1->offset_candidate_image = (*(viz_input->sorted_offsets))[index_of_image_1];

                    new_mr.additional_viz_data_id2_to_id1->path_to_feature_image = (*(viz_input->sorted_image_paths))[index_of_image_2];
                    new_mr.additional_viz_data_id2_to_id1->path_to_candidate_image = (*(viz_input->sorted_image_paths))[index_of_image_1];

                    new_mr.additional_viz_data_id2_to_id1->feature_image = nullptr;//new cv::Mat;
                    //*new_mr.additional_viz_data_id2_to_id1->feature_image = (*(viz_input->sorted_square_organoid_images))[index_of_image_2]->clone();

                    new_mr.additional_viz_data_id2_to_id1->candidate_image = nullptr;//new cv::Mat;
                    //*new_mr.additional_viz_data_id2_to_id1->candidate_image = (*(viz_input->sorted_square_organoid_images))[index_of_image_1]->clone();

                    new_mr.additional_viz_data_id2_to_id1->features = (*(viz_input->all_feature_vectors))[index_of_image_2].features;
                    new_mr.additional_viz_data_id2_to_id1->assigned_candidates = (*(viz_input->all_feature_vectors))[index_of_image_1].candidates;
                }
                //std::cout << "before push mr" << std::endl;
                all_matching_results.push_back(new_mr);
                //std::cout << "after push mr" << std::endl;
            }

            //std::cout << "transcribed all matching results from: " << i << std::endl;

        }

        for (i = 0; i < mpi_all_matching_results.size(); i++) {

            std::vector<MPI_Comm_Matching_Result>* new_matching_result_vector = mpi_all_matching_results[i];
            delete(new_matching_result_vector);  
        }

        std::vector<int> num_execution_time_measurements_per_process;
        
        for (i = 1; i < size; i++) {
            int num_execution_time_measurements = 0;

            MPI_Recv(&num_execution_time_measurements, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
            //std::cout << rank << " will receive: " << num_execution_time_measurements << " execution time measurements from process: " << i << std::endl;

            num_execution_time_measurements_per_process.push_back(num_execution_time_measurements); 
        }

        std::vector<std::vector<MPI_Comm_Execution_Time_Measurement>*> mpi_all_execution_time_measurements;

        for (i = 1; i < size; i++) {

            std::vector<MPI_Comm_Execution_Time_Measurement>* new_execution_time_measurement_vector = new std::vector<MPI_Comm_Execution_Time_Measurement>;
            new_execution_time_measurement_vector->resize(num_execution_time_measurements_per_process[i-1]);

            mpi_all_execution_time_measurements.push_back(new_execution_time_measurement_vector);  
        }

        for (i = 1; i < size; i++) {
            MPI_Recv(mpi_all_execution_time_measurements[i-1]->data(), num_execution_time_measurements_per_process[i-1], mpi_execution_time_measurement, i, tag, MPI_COMM_WORLD, &status);

            for(int j = 0; j < num_execution_time_measurements_per_process[i-1];j++){

                Op_Numbers_And_Runtime new_exe_meausure;

                MPI_Comm_Execution_Time_Measurement received_measurement = mpi_all_execution_time_measurements[i-1]->at(j); 

                new_exe_meausure.num_features = received_measurement.num_features;
                new_exe_meausure.num_candiates = received_measurement.num_candidates;
                new_exe_meausure.total_runtime = received_measurement.execution_time;

                execution_times.push_back(new_exe_meausure);
            }
        }

        for (i = 0; i < mpi_all_execution_time_measurements.size(); i++) {

            std::vector<MPI_Comm_Execution_Time_Measurement>* new_execution_time_measurement_vector = mpi_all_execution_time_measurements[i];
            delete(new_execution_time_measurement_vector);  
        }
        

    } else {
        //sprintf(message, "Hello world from process %d running on %s",rank,machine_name);
        std::vector<MPI_Comm_Matching_Result> obtained_matching_results;
        obtained_matching_results.reserve(all_matching_results.size());

        int num_of_obtained_matching_results = all_matching_results.size();
        MPI_Send(&num_of_obtained_matching_results, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);


        std::vector<int> all_assignments_flat_array;
        all_assignments_flat_array.clear();

        for(int i = 0; i < all_matching_results.size();i++){
            MPI_Comm_Matching_Result current_matching_result;//{(double)rank,(double)rank,(double)rank,(double)rank,rank,rank,rank,rank};
            current_matching_result.id_1 = all_matching_results[i].id_1;
            current_matching_result.id_2 = all_matching_results[i].id_2;
            current_matching_result.set_id_1 = all_matching_results[i].set_id_1;
            current_matching_result.set_id_2 = all_matching_results[i].set_id_2;
            current_matching_result.linear_cost_per_candidate = all_matching_results[i].linear_cost_per_candidate;
            current_matching_result.linear_cost_per_feature = all_matching_results[i].linear_cost_per_feature;
            current_matching_result.rel_quadr_cost = all_matching_results[i].rel_quadr_cost;
            current_matching_result.rel_quadr_cost_optimal = all_matching_results[i].rel_quadr_cost_optimal;

            current_matching_result.offset_into_assignment_array = all_assignments_flat_array.size();

            for(int j = 0; j < all_matching_results[i].assignment->size();j++){
                //std::cout << (*(all_matching_results[i].assignment))[j] << " ";
                all_assignments_flat_array.push_back((*(all_matching_results[i].assignment))[j]);
            }

            //std::cout << std::endl;

            current_matching_result.num_elems_in_assignment = all_assignments_flat_array.size() - current_matching_result.offset_into_assignment_array;

            obtained_matching_results.push_back(current_matching_result);
        }


        int all_assignments_flat_array_size = all_assignments_flat_array.size();
        MPI_Send(&all_assignments_flat_array_size, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);

        MPI_Send(all_assignments_flat_array.data(),all_assignments_flat_array.size(),MPI_INT,0,tag,MPI_COMM_WORLD);
        
        MPI_Send(obtained_matching_results.data(), num_of_obtained_matching_results, mpi_matching_result, 0, tag, MPI_COMM_WORLD);

        int num_execution_time_measurements = execution_times.size();
        //std::cout << "process " << rank << " will send: " << num_execution_time_measurements << std::endl;
        MPI_Send(&num_execution_time_measurements,1,MPI_INT,0,tag,MPI_COMM_WORLD);

        std::vector<MPI_Comm_Execution_Time_Measurement> obtained_execution_time_measurements;

        for(int i = 0; i < execution_times.size();i++){
            MPI_Comm_Execution_Time_Measurement new_exe_time_measure;
            new_exe_time_measure.num_features = execution_times[i].num_features;
            new_exe_time_measure.num_candidates = execution_times[i].num_candiates;
            new_exe_time_measure.execution_time = execution_times[i].total_runtime;

            obtained_execution_time_measurements.push_back(new_exe_time_measure);
        }

        MPI_Send(obtained_execution_time_measurements.data(),num_execution_time_measurements,mpi_execution_time_measurement,0,tag,MPI_COMM_WORLD);

    }


    return;
}