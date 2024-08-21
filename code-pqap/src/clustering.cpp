#include "clustering.h"
#include "gurobi_c++.h"
#include <thread>


#include <andres/graph/complete-graph.hxx>
#include <andres/graph/multicut/greedy-additive.hxx>
#include <andres/graph/multicut/kernighan-lin.hxx>
#include <andres/graph/components.hxx>
#include "andres/graph/multicut/ilp.hxx"
#include "andres/ilp/gurobi.hxx"

#define CLUSTERING_WITH_LAZY_CONSTRAINTS_HARD_TIMELIMIT 6000
#define CLUSTERING_WITH_LAZY_CONSTRAINTS_SOFT_TIMELIMIT 2000
#define CLUSTERING_WITH_LAZY_CONSTRAINTS_GAP_MINIMUM 0.01



typedef struct Three_Elementary_Subset{
    int elem_1;
    int elem_2;
    int elem_3;

}Three_Elementary_Subset;

typedef struct Cost_Change_Move{
    double cost_change_cluster_1;
    double cost_change_cluster_2;
    double total_cost_change;
}Cost_Change_Move;

typedef struct Move{
    Cost_Change_Move cost_of_move;
    int cluster_1_id;
    int cluster_2_id;
    int member_in_c1_to_move;
}Move;


float sum_similarities_to_cluster_members(std::vector<Matching_Result>& all_matching_results, Cluster& current_cluster, int base_member_id);

void delete_empty_clusters(std::vector<Cluster>& all_clusters);

void calculate_all_three_elementary_subsets(std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Three_Elementary_Subset>& all_subsets);

void print_all_clusters(std::vector<Cluster>& all_clusters);

double calculate_cost_of_single_cluster(std::vector<Cluster>& all_clusters, int cluster_id, std::vector<Matching_Result>& all_matching_results, int member_to_exclude = -1);

double calculate_cost_of_clustering(std::vector<Cluster>& all_clusters);

double calculate_cost_of_clustering_from_matching_data(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results);

double calculate_cost_for_joining_two_clusters(std::vector<Cluster>& all_clusters, int cluster_id1, int cluster_id2, std::vector<Matching_Result>& all_matching_results);

Cost_Change_Move calculate_cost_of_move(std::vector<Cluster>& all_clusters, int cluster_id1, int cluster_id2, int member_of_cluster_1_to_move ,std::vector<Matching_Result>& all_matching_results);

void join_two_clusters(std::vector<Cluster>& all_clusters, int cluster_id1, int cluster_id2, double cost_of_join, std::vector<int>* valid_cluster_indices = nullptr, double* cost_matrix = nullptr, std::vector<Matching_Result>* all_matching_results = nullptr);

void execute_move(std::vector<Cluster>& all_clusters, int cluster_id1, int cluster_id2, int member_in_c1_to_move, Cost_Change_Move& cost_of_move, Move* costs_of_best_moves_matrix = nullptr, std::vector<int>* currently_active_cluster_ids = nullptr, std::vector<int>* currently_inactive_cluster_ids = nullptr, std::vector<Matching_Result>* all_matching_results = nullptr);

void greedy_join_clusters(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results);

void greedy_join_clusters_optimized(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results);

void greedy_move_clusters(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results);

void greedy_move_clusters_optimized(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results);

int get_index_into_cost_matrix(int cluster_id_1, int cluster_id_2, int total_num_initial_clusters);

void write_to_move_cost_matrix_thread_safe(int cluster_id_1, int cluster_id_2, int total_num_initial_clusters, std::vector<std::mutex*>* lock_matrix, Move& best_move_of_cluster, Move* best_moves_matrix);

void print_cost_matrix(double* cost_matrix, int total_num_initial_clusters);

void update_move_cost_matrix_for_single_cluster(int cluster_id,std::vector<int>* currently_active_cluster_ids, Move* move_cost_matrix, std::vector<Cluster>& all_clusters, std::vector<Matching_Result>* all_matching_results);

void update_move_cost_matrix_for_single_cluster_in_active_id_range(int first_index_to_process, int last_index_to_process, int cluster_id,std::vector<int>* currently_active_cluster_ids, Move* move_cost_matrix, std::vector<Cluster>* all_clusters, std::vector<Matching_Result>* all_matching_results);

void calculate_move_cost_matrix_for_active_id_range(int thread_id, int total_num_threads, std::vector<int>* currently_active_cluster_ids,std::vector<Cluster>* all_clusters, std::vector<Matching_Result>* all_matching_results, Move* best_moves_per_cluster_matrix, std::vector<std::mutex*>* lock_matrix);

void setup_new_clusters(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters);

void setup_new_clusters(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters){

    int num_organoids_to_cluster = 0;

    std::vector<int> unique_image_ids_in_matching_results;

    for(int i = 0; i < all_matching_results.size();i++){

        Matching_Result current_mr = all_matching_results[i];

        bool found_first_image_id = false;
        bool found_second_image_id = false;

        for(int j = 0; j < unique_image_ids_in_matching_results.size();j++){
            if(current_mr.id_1 == unique_image_ids_in_matching_results[j]){
                found_first_image_id = true;
            }

            if(current_mr.id_2 == unique_image_ids_in_matching_results[j]){
                found_second_image_id = true;
            }

            if(found_first_image_id && found_second_image_id){
                break;
            }
        }

        if(!found_first_image_id){
            unique_image_ids_in_matching_results.push_back(current_mr.id_1);
        }

        if(!found_second_image_id){
            unique_image_ids_in_matching_results.push_back(current_mr.id_2);
        }
    }

    num_organoids_to_cluster =  unique_image_ids_in_matching_results.size();

    for(int i = 0; i < num_organoids_to_cluster; i++){
        Cluster new_cluster;

        new_cluster.cost_of_cluster = 0.0;

        new_cluster.members = new std::vector<int>;
        new_cluster.members->push_back(unique_image_ids_in_matching_results[i]);

        all_clusters.push_back(new_cluster);
    }

}

template<class T>
void findLocalSearchEdgeLabels(
        andres::graph::CompleteGraph<> const & graph,
        std::vector<T> const & edgeCosts,
        std::vector<size_t> & edgeLabels
        ){
    std::cout << "Applying Additive Edge Contraction..." << std::endl;
    /*
    andres::graph::multicut::greedyAdditiveEdgeContractionCompleteGraph(
            graph,
            edgeCosts,
            edgeLabels
    );
    */

   andres::graph::multicut::greedyAdditiveEdgeContraction(
            graph,
            edgeCosts,
            edgeLabels
    );

    std::cout << "Applying Kernighan Lin..." << std::endl;
    andres::graph::multicut::kernighanLin(
            graph,
            edgeCosts,
            edgeLabels,
            edgeLabels
    );
    std::cout << "Applying Kernighan Lin done..." << std::endl;
}

void calculate_clustering_using_graph_implementation(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters){

    if(all_matching_results.size() == 0){
        std::cout << "size of all_matching_results was 0 in calculate_clustering_using_graph_implementation." << std::endl;
        return;
    }

    size_t numberOfElements = all_feature_vectors.size();

    andres::graph::CompleteGraph<> graph(numberOfElements);

    std::vector<double> edgeCosts(graph.numberOfEdges(), 0);

    setup_new_clusters(all_matching_results,all_feature_vectors,all_clusters);

    for (size_t i = 0; i < numberOfElements; ++i){
        for (size_t j = 0; j < i ; ++j){



            int first_img_id = all_feature_vectors[i].image_number;
            int second_img_id = all_feature_vectors[j].image_number;

            double new_edge_cost = 0.0;

            for(int k = 0; k < all_matching_results.size();k++){

                if( (first_img_id == all_matching_results[k].id_1 && second_img_id == all_matching_results[k].id_2) || (first_img_id == all_matching_results[k].id_2 && second_img_id == all_matching_results[k].id_1) ){
                    new_edge_cost = all_matching_results[k].rel_quadr_cost;
                    break;
                }
                
            }



            std::pair<bool , size_t > graphEdgeIndex = graph.findEdge(j, i);
            if (!graphEdgeIndex.first){
                std::cout << j << " - " << i << std::endl;
                throw std::runtime_error("Graph Edge does not exist.");
            }
            edgeCosts[graphEdgeIndex.second] = new_edge_cost;

        }
    }

    std::vector<std::size_t> localSearchEdgeLabels(graph.numberOfEdges(), 1);

    findLocalSearchEdgeLabels<double>(graph, edgeCosts, localSearchEdgeLabels);


    for (size_t i = 0; i < numberOfElements; ++i){
        for (size_t j = 0; j < i ; ++j){


            int first_img_id = all_feature_vectors[i].image_number;
            int second_img_id = all_feature_vectors[j].image_number;

            std::pair<bool , size_t > graphEdgeIndex = graph.findEdge(j, i);
            if (!graphEdgeIndex.first){
                std::cout << j << " - " << i << std::endl;
                throw std::runtime_error("Graph Edge does not exist.");
            }

            size_t edge_label = localSearchEdgeLabels[graphEdgeIndex.second];

            int img_id1 = all_feature_vectors[i].image_number;
            int img_id2 = all_feature_vectors[j].image_number;


            //std::cout << var_index << " " << current_var.get(GRB_DoubleAttr_X) << std::endl;
            int cluster_id1 = -1;
            int cluster_id2 = -1;

            if(!edge_label){
                
                for(int k = 0; k < all_clusters.size();k++){

                    Cluster current_cluster = all_clusters[k];

                    for(int l = 0; l < current_cluster.members->size();l++){
                        int current_cluster_member = (*(current_cluster.members))[l];

                        if(img_id1 == current_cluster_member){
                            cluster_id1 = k;
                        }

                        if(img_id2 == current_cluster_member){
                            cluster_id2 = k;
                        }

                        if(cluster_id1 != -1 && cluster_id2 != -1){
                            goto loop_end;
                        }

                    }
                }

                loop_end:

                if(cluster_id1 != cluster_id2){
                    float cost_of_join = calculate_cost_for_joining_two_clusters(all_clusters,cluster_id1,cluster_id2,all_matching_results);
                    join_two_clusters(all_clusters,cluster_id1,cluster_id2,cost_of_join);
                }

            
            }
        }
    }

}

void calculate_clustering(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters, bool use_optimized){

    std::cout << std::endl;
    std::cout << "Begin Greedy Clustering Calculation" << std::endl;

    int num_organoids_to_cluster = all_feature_vectors.size();

    /*
    for(int i = 0; i < all_matching_results.size();i++){
        Matching_Result current_mr = all_matching_results[i];

        std::cout << current_mr.id_1 << " " << current_mr.id_2 << " " << current_mr.set_id_1 << " " << current_mr.set_id_2 << std::endl;
    }
    */

    setup_new_clusters(all_matching_results,all_feature_vectors,all_clusters);

    /*
    for(int i = 0; i < num_organoids_to_cluster; i++){
        Cluster new_cluster;

        new_cluster.cost_of_cluster = 0.0;

        new_cluster.members = new std::vector<int>;
        new_cluster.members->push_back(all_feature_vectors[i].image_number);

        all_clusters.push_back(new_cluster);
    }
    */

    //print_all_clusters(all_clusters);


    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if(use_optimized){
        std::cout << "begin optimized greedy joining" << std::endl;
        greedy_join_clusters_optimized(all_clusters,all_matching_results);
    }else{
        greedy_join_clusters(all_clusters,all_matching_results);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();


    std::cout << "Greedy join cost: " << calculate_cost_of_clustering(all_clusters) << " time: " << time << " optimized: " << use_optimized << std::endl;


    std::cout << "Finished joining clusters" << std::endl;

    //print_all_clusters(all_clusters);

    //print_all_clusters(all_clusters);



    begin = std::chrono::steady_clock::now();
    if(use_optimized){
        std::cout << "begin optimized greedy moving" << std::endl;
        //greedy_move_clusters_optimized(all_clusters,all_matching_results);
    }else{
        //greedy_move_clusters(all_clusters,all_matching_results);
    }
    end = std::chrono::steady_clock::now();

    time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "Greedy move cost: " << calculate_cost_of_clustering(all_clusters) << " time: " << time << " optimized: " << use_optimized << std::endl;

    //std::cout << "costs of clustering:" << std::endl;
    //std::cout << calculate_cost_of_clustering(all_clusters) << std::endl;
    //std::cout << calculate_cost_of_clustering_from_matching_data(all_clusters,all_matching_results,threshold) << std::endl;


    delete_empty_clusters(all_clusters);

    //print_all_clusters(all_clusters);

}

void greedy_move_clusters(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results){

    Cost_Change_Move best_move_cost_change;
    best_move_cost_change.total_cost_change = -1.0;

    int id_of_c1_in_move = -1;
    int id_of_c2_in_move = -1;

    int id_of_c1_member_to_move = -1;

    bool found_move = true;

    while(found_move){

        found_move = false;

        for(int i = 0; i < all_clusters.size();i++){    
            for(int j = 0; j < all_clusters.size();j++){

                if(i == j){
                    continue;
                }

                for(int member_index = 0; member_index < all_clusters[i].members->size();member_index++){

                    Cost_Change_Move current_cost_of_move = calculate_cost_of_move(all_clusters,i,j,member_index,all_matching_results);

                    if(current_cost_of_move.total_cost_change > best_move_cost_change.total_cost_change){
                        best_move_cost_change = current_cost_of_move;

                        id_of_c1_in_move = i;
                        id_of_c2_in_move = j;

                        id_of_c1_member_to_move = member_index;
                    }
                }


            }
        }

        if(best_move_cost_change.total_cost_change > 0.0 && id_of_c1_in_move != -1 && id_of_c2_in_move != -1 && id_of_c1_member_to_move != -1){

            //print_all_clusters(all_clusters);

            //std::cout << "cost of clustering before move: " << calculate_cost_of_clustering_from_matching_data(all_clusters,all_matching_results,threshold) << std::endl;

            execute_move(all_clusters,id_of_c1_in_move,id_of_c2_in_move,id_of_c1_member_to_move,best_move_cost_change);

            best_move_cost_change.total_cost_change = -1.0;
            id_of_c1_in_move = -1;
            id_of_c2_in_move = -1;
            id_of_c1_member_to_move = -1;

            found_move = true;

            //std::cout << "cost of clustering after move: " << calculate_cost_of_clustering_from_matching_data(all_clusters,all_matching_results,threshold) << std::endl;

            //print_all_clusters(all_clusters);
        }
    }
}

void greedy_move_clusters_optimized(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results){

    Cost_Change_Move best_move_cost_change;
    best_move_cost_change.total_cost_change = -1.0;

    int initial_num_clusters = all_clusters.size();


    // in the currently active_cluster_ids we save all the ids of the clusters which have atleast one member and one cluster which has no members  
    std::vector<int> currently_active_cluster_ids;

    // the currently_inactive_cluster_ids contain all the cluster ids which have no members exept for the one which we have put into the currently_active_cluster_ids
    std::vector<int> currently_inactive_cluster_ids;

    Move* best_moves_per_cluster_matrix = (Move*)malloc(sizeof(Move) * initial_num_clusters * initial_num_clusters);

    std::vector<std::mutex*> lock_matrix;

    for(int i = 0; i < initial_num_clusters * initial_num_clusters; i++){
        std::mutex* new_lock = new std::mutex;
        lock_matrix.push_back(new_lock);
    }

    int id_of_c1_in_move = -1;
    int id_of_c2_in_move = -1;

    int id_of_c1_member_to_move = -1;

    for(int i = 0; i < all_clusters.size(); i++){
        Cluster current_cluster = all_clusters[i];

        if(current_cluster.members->size() == 0){
            currently_inactive_cluster_ids.push_back(i);
        }else{
            currently_active_cluster_ids.push_back(i);
        }
    }

    //add one of the empty clusters to the active_cluster_ids;

    if(currently_inactive_cluster_ids.size() > 0){
        currently_active_cluster_ids.push_back(currently_inactive_cluster_ids[currently_inactive_cluster_ids.size() - 1]);
        currently_inactive_cluster_ids.pop_back();
    }


    //we need to initialize the best move matrix
    for(int i = 0; i < currently_active_cluster_ids.size();i++){    
        int cluster_id_1 = currently_active_cluster_ids[i];

        for(int j = 0; j < currently_active_cluster_ids.size();j++){

            if(i == j){
                continue;
            }

            int cluster_id_2 = currently_active_cluster_ids[j];

            Move best_move_of_cluster;
            best_move_of_cluster.cost_of_move.total_cost_change = -10000000000000.0;

            best_moves_per_cluster_matrix[get_index_into_cost_matrix(cluster_id_1,cluster_id_2,initial_num_clusters)] = best_move_of_cluster;
        }
    }

    
    unsigned int num_hardware_threads = std::thread::hardware_concurrency();

    std::vector<std::thread*> additional_threads;
    additional_threads.resize(num_hardware_threads - 1);

    int ids_per_thread = currently_active_cluster_ids.size() / num_hardware_threads;

    for(int thread = 0; thread < num_hardware_threads - 1; thread++){

        int first_id_for_thread_to_process = ids_per_thread * thread;
        int last_id_for_thread_to_process = ids_per_thread * (thread + 1);

        //std::cout << "ids to process for thread " << thread << ": " << first_id_for_thread_to_process << " to " << last_id_for_thread_to_process << std::endl;
        
        additional_threads[thread] = new std::thread(calculate_move_cost_matrix_for_active_id_range,thread,num_hardware_threads,&currently_active_cluster_ids,&all_clusters,&all_matching_results,best_moves_per_cluster_matrix,&lock_matrix);
    }

    int first_id_for_masterthread_to_process = ids_per_thread * (num_hardware_threads - 1);
    int last_id_for_masterthread_to_process = currently_active_cluster_ids.size();

    //std::cout << "ids to process for masterthread: " << first_id_for_masterthread_to_process << " to " << last_id_for_masterthread_to_process << std::endl;

    calculate_move_cost_matrix_for_active_id_range(num_hardware_threads - 1,num_hardware_threads,&currently_active_cluster_ids,&all_clusters,&all_matching_results,best_moves_per_cluster_matrix,&lock_matrix);

    for(int thread = 0; thread < num_hardware_threads - 1; thread++){
        additional_threads[thread]->join();
    }
    

    /*
    for(int i = 0; i < currently_active_cluster_ids.size();i++){    
        int cluster_id_1 = currently_active_cluster_ids[i];

        for(int j = 0; j < currently_active_cluster_ids.size();j++){

            if(i == j){
                continue;
            }

            int cluster_id_2 = currently_active_cluster_ids[j];


            Move best_move_of_cluster;
            best_move_of_cluster.cost_of_move.total_cost_change = -10000000000000.0;

            for(int member_index = 0; member_index < all_clusters[cluster_id_1].members->size();member_index++){

                Cost_Change_Move current_cost_of_move = calculate_cost_of_move(all_clusters,cluster_id_1,cluster_id_2,member_index,all_matching_results);

                if(current_cost_of_move.total_cost_change > best_move_of_cluster.cost_of_move.total_cost_change){
                    best_move_of_cluster.cost_of_move = current_cost_of_move;

                    best_move_of_cluster.cluster_1_id = cluster_id_1;
                    best_move_of_cluster.cluster_2_id = cluster_id_2;

                    best_move_of_cluster.member_in_c1_to_move = member_index;
                }
            }

            best_moves_per_cluster_matrix[get_index_into_cost_matrix(cluster_id_1,cluster_id_2,initial_num_clusters)] = best_move_of_cluster;
        }
    }
    */
    
    bool found_move = true;

    while(found_move){

        found_move = false;

        for(int i = 0; i < currently_active_cluster_ids.size();i++){    

            int cluster_id_1 = currently_active_cluster_ids[i];
            for(int j = 0; j < currently_active_cluster_ids.size();j++){

                if(i == j){
                    continue;
                }

                int cluster_id_2 = currently_active_cluster_ids[j];


                Move best_move_of_cluster = best_moves_per_cluster_matrix[get_index_into_cost_matrix(cluster_id_1,cluster_id_2,initial_num_clusters)];

                Cost_Change_Move current_cost_of_move = best_move_of_cluster.cost_of_move;

                if(current_cost_of_move.total_cost_change > best_move_cost_change.total_cost_change){
                    best_move_cost_change = current_cost_of_move;

                    id_of_c1_in_move = best_move_of_cluster.cluster_1_id;
                    id_of_c2_in_move = best_move_of_cluster.cluster_2_id;

                    id_of_c1_member_to_move = best_move_of_cluster.member_in_c1_to_move;
                }
            }
        }

        if(best_move_cost_change.total_cost_change > 0.0 && id_of_c1_in_move != -1 && id_of_c2_in_move != -1 && id_of_c1_member_to_move != -1){


            execute_move(all_clusters,id_of_c1_in_move,id_of_c2_in_move,id_of_c1_member_to_move,best_move_cost_change,best_moves_per_cluster_matrix,&currently_active_cluster_ids,&currently_inactive_cluster_ids,&all_matching_results);

            best_move_cost_change.total_cost_change = -1.0;
            id_of_c1_in_move = -1;
            id_of_c2_in_move = -1;
            id_of_c1_member_to_move = -1;

            found_move = true;

            //std::cout << "cost of clustering after move: " << calculate_cost_of_clustering_from_matching_data(all_clusters,all_matching_results,threshold) << std::endl;

            //print_all_clusters(all_clusters);
        }
    }

    for(int i = 0; i < lock_matrix.size(); i++){
        delete(lock_matrix[i]);
    }
}

void greedy_join_clusters(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results){

    int num_initial_clusters = all_clusters.size();

    double cost_of_best_join = -1.0;

    int id_of_c1_to_join = -1;
    int id_of_c2_to_join = -1;

    bool found_cluster_to_join = true;

    //double* cost_matrix = (double*)malloc(sizeof(double) * num_initial_clusters * num_initial_clusters);

    while(found_cluster_to_join){
        found_cluster_to_join = false;

        for(int i = 0; i < all_clusters.size();i++){


            for(int j = i + 1; j < all_clusters.size();j++){

                double cost_of_current_join = calculate_cost_for_joining_two_clusters(all_clusters,i,j,all_matching_results);

                //int index_cost_matrix = i * num_initial_clusters + j;
                //int transposed_index_cost_matrix = j * num_initial_clusters + i;

                //cost_matrix[index_cost_matrix] = cost_of_current_join;
                //cost_matrix[transposed_index_cost_matrix] = cost_of_current_join;

                //

                if(cost_of_current_join > cost_of_best_join){
                    cost_of_best_join = cost_of_current_join;

                    id_of_c1_to_join = i;
                    id_of_c2_to_join = j;
                }
                

            }
        }

        //std::cout << "cost matrix of current iteration" << std::endl;
        //print_cost_matrix(cost_matrix,num_initial_clusters);

        if(cost_of_best_join > 0.0 && id_of_c1_to_join != -1 && id_of_c2_to_join != -1){
            join_two_clusters(all_clusters,id_of_c1_to_join,id_of_c2_to_join,cost_of_best_join);
            //std::cout << cost_of_best_join << " from " << id_of_c1_to_join << " and " << id_of_c2_to_join  << std::endl;
            //std::cout << calculate_cost_of_clustering(all_clusters) << std::endl;
            //std::cout << calculate_cost_of_clustering_from_matching_data(all_clusters,all_matching_results,threshold) << std::endl;
            //print_all_clusters(all_clusters);

            cost_of_best_join = -1.0;
            id_of_c1_to_join = -1;
            id_of_c2_to_join = -1;

            found_cluster_to_join = true;

        }

    }

}

void greedy_join_clusters_optimized(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results){

    int num_initial_clusters = all_clusters.size();

    double* cost_matrix = (double*)malloc(sizeof(double) * num_initial_clusters * num_initial_clusters); 

    std::vector<int> valid_cluster_ids;

    for(int i = 0; i < all_clusters.size();i++){
        valid_cluster_ids.push_back(i);
    }

    for(int i = 0; i < valid_cluster_ids.size();i++){

        int cluster_id_1 = valid_cluster_ids[i];

        for(int j = i + 1; j < valid_cluster_ids.size();j++){

            int cluster_id_2 = valid_cluster_ids[j];

            double cost_of_current_join = calculate_cost_for_joining_two_clusters(all_clusters,cluster_id_1,cluster_id_2,all_matching_results);

            int index_cost_matrix = cluster_id_1 * num_initial_clusters + cluster_id_2;
            int transposed_index_cost_matrix = cluster_id_2 * num_initial_clusters + cluster_id_1;

            cost_matrix[index_cost_matrix] = cost_of_current_join;
            cost_matrix[transposed_index_cost_matrix] = cost_of_current_join;
        }
    }


    double cost_of_best_join = -1.0;

    int id_of_c1_to_join = -1;
    int id_of_c2_to_join = -1;

    bool found_cluster_to_join = true;

    while(found_cluster_to_join){
        //std::cout << "start of iteration" << std::endl;
        //print_cost_matrix(cost_matrix, num_initial_clusters);
        found_cluster_to_join = false;

        for(int i = 0; i < valid_cluster_ids.size();i++){

            int cluster_id_1 = valid_cluster_ids[i];

            for(int j = i + 1; j < valid_cluster_ids.size();j++){

                int cluster_id_2 = valid_cluster_ids[j];

                int index_into_cost_matrix = get_index_into_cost_matrix(cluster_id_1,cluster_id_2,num_initial_clusters);

                double cost_of_current_join = cost_matrix[index_into_cost_matrix];//calculate_cost_for_joining_two_clusters(all_clusters,cluster_id_1,cluster_id_2,all_matching_results,threshold);

                if(cost_of_current_join > cost_of_best_join){
                    cost_of_best_join = cost_of_current_join;

                    id_of_c1_to_join = cluster_id_1;
                    id_of_c2_to_join = cluster_id_2;
                }
            }
        }

        if(cost_of_best_join > 0.0 && id_of_c1_to_join != -1 && id_of_c2_to_join != -1){
            join_two_clusters(all_clusters,id_of_c1_to_join,id_of_c2_to_join,cost_of_best_join,&valid_cluster_ids,cost_matrix,&all_matching_results);
            //std::cout << cost_of_best_join << " from " << id_of_c1_to_join << " and " << id_of_c2_to_join  << std::endl;
            //std::cout << calculate_cost_of_clustering(all_clusters) << std::endl;
            //std::cout << calculate_cost_of_clustering_from_matching_data(all_clusters,all_matching_results,threshold) << std::endl;
            //print_all_clusters(all_clusters);

            cost_of_best_join = -1.0;
            id_of_c1_to_join = -1;
            id_of_c2_to_join = -1;

            found_cluster_to_join = true;

        }

    }

    free(cost_matrix);

}

int get_index_into_cost_matrix(int cluster_id_1, int cluster_id_2, int total_num_initial_clusters){
    return  cluster_id_1 * total_num_initial_clusters + cluster_id_2;
}

void print_cost_matrix(double* cost_matrix, int total_num_initial_clusters){
    for(int i = 0; i < total_num_initial_clusters;i++){
        for(int j = 0; j < total_num_initial_clusters;j++){

            std::cout << cost_matrix[get_index_into_cost_matrix(i,j,total_num_initial_clusters)] << "  ";
        
        }   
        std::cout << std::endl; 
    }

}

double calculate_cost_of_clustering(std::vector<Cluster>& all_clusters){

    double total_cost = 0.0;

    for(int i = 0; i < all_clusters.size();i++){

        //std::cout << "cluster " << i << " with " << all_clusters[i].members->size() << " and cost: " << all_clusters[i].cost_of_cluster << std::endl;
        total_cost += all_clusters[i].cost_of_cluster;
    }

    return total_cost;
}

double calculate_cost_of_clustering_from_matching_data(std::vector<Cluster>& all_clusters, std::vector<Matching_Result>& all_matching_results){

    double total_cost_of_clustering = 0.0;

    for(int cluster_id = 0; cluster_id < all_clusters.size(); cluster_id++){

        double cost_of_cluster = calculate_cost_of_single_cluster(all_clusters,cluster_id,all_matching_results);

        total_cost_of_clustering += cost_of_cluster;
    }

    return total_cost_of_clustering;

}

double calculate_cost_of_single_cluster(std::vector<Cluster>& all_clusters, int cluster_id, std::vector<Matching_Result>& all_matching_results, int member_to_exclude){

    Cluster current_cluster = all_clusters[cluster_id];

    double cost_of_cluster = 0.0;

    for(int i = 0; i < current_cluster.members->size();i++){
        if(i == member_to_exclude){
            continue;
        }   

        int id1 = (*(current_cluster.members))[i];

        for(int j = i+1; j < current_cluster.members->size();j++){
            if(j == member_to_exclude){
                continue;
            }

            int id2 = (*(current_cluster.members))[j];

            for(int k = 0; k < all_matching_results.size();k++){

                Matching_Result current_mr = all_matching_results[k];

                if((current_mr.id_1 == id1 && current_mr.id_2 == id2) || (current_mr.id_1 == id2 && current_mr.id_2 == id1)){

                    cost_of_cluster += current_mr.rel_quadr_cost; //- threshold;

                }
            }

        
        }    
    }

    return cost_of_cluster;

}

Cost_Change_Move calculate_cost_of_move(std::vector<Cluster>& all_clusters, int cluster_id1, int cluster_id2, int member_of_cluster_1_to_move ,std::vector<Matching_Result>& all_matching_results){

    double cost_of_cluster_1_after_moving = calculate_cost_of_single_cluster(all_clusters,cluster_id1,all_matching_results,member_of_cluster_1_to_move);

    all_clusters[cluster_id2].members->push_back((*(all_clusters[cluster_id1].members))[member_of_cluster_1_to_move]);

    double cost_of_cluster_2_after_moving = calculate_cost_of_single_cluster(all_clusters,cluster_id2,all_matching_results);

    all_clusters[cluster_id2].members->pop_back();

    double cost_change_in_cluster_1 = cost_of_cluster_1_after_moving - all_clusters[cluster_id1].cost_of_cluster;
    double cost_change_in_cluster_2 = cost_of_cluster_2_after_moving - all_clusters[cluster_id2].cost_of_cluster;

    Cost_Change_Move cost_change;
    cost_change.cost_change_cluster_1 = cost_change_in_cluster_1;
    cost_change.cost_change_cluster_2 = cost_change_in_cluster_2;
    cost_change.total_cost_change = cost_change_in_cluster_1 + cost_change_in_cluster_2;

    return cost_change;

}

double calculate_cost_for_joining_two_clusters(std::vector<Cluster>& all_clusters, int cluster_id1, int cluster_id2, std::vector<Matching_Result>& all_matching_results){
    Cluster cluster_1 = all_clusters[cluster_id1];
    Cluster cluster_2 = all_clusters[cluster_id2];

    double total_cost_of_join = 0.0;

    for(int i = 0; i < cluster_1.members->size();i++){
        int c1_memb_id = (*(cluster_1.members))[i];

        for(int j = 0; j < cluster_2.members->size();j++){
            int c2_memb_id = (*(cluster_2.members))[j];

            for(int k = 0; k < all_matching_results.size();k++){
                Matching_Result m_result = all_matching_results[k];

                if((m_result.id_1 == c1_memb_id && m_result.id_2 == c2_memb_id) || (m_result.id_1 == c2_memb_id && m_result.id_2 == c1_memb_id)){

                    total_cost_of_join += m_result.rel_quadr_cost;// - threshold; 
                }
            }
        }
    }

    return total_cost_of_join;
}

void calculate_move_cost_matrix_for_active_id_range(int thread_id, int total_num_threads, std::vector<int>* currently_active_cluster_ids,std::vector<Cluster>* all_clusters, std::vector<Matching_Result>* all_matching_results, Move* best_moves_per_cluster_matrix, std::vector<std::mutex*>* lock_matrix){

    std::vector<Cluster> clusters_local_copy;

    for(int i = 0; i < all_clusters->size();i++){

        Cluster current_cluster = (*all_clusters)[i];

        Cluster local_copy_of_cluster;

        local_copy_of_cluster.cost_of_cluster = current_cluster.cost_of_cluster;

        local_copy_of_cluster.members = new std::vector<int>;

        for(int l = 0; l < current_cluster.members->size();l++){
            local_copy_of_cluster.members->push_back((*current_cluster.members)[l]);
        }

        clusters_local_copy.push_back(local_copy_of_cluster);

    } 

    for(int i = 0; i < currently_active_cluster_ids->size();i++){    
        int cluster_id_1 = (*currently_active_cluster_ids)[i];

        for(int j = 0; j < currently_active_cluster_ids->size();j++){

            if(i == j){
                continue;
            }

            int cluster_id_2 = (*currently_active_cluster_ids)[j];


            Move best_move_of_cluster;
            best_move_of_cluster.cost_of_move.total_cost_change = -10000000000000.0;

            int total_num_members = clusters_local_copy[cluster_id_1].members->size();

            int members_per_thread = total_num_members / total_num_threads;

            int first_member_index = std::min(thread_id * members_per_thread,total_num_members);
            int last_member_index = std::min((thread_id + 1) * members_per_thread,total_num_members);

            if(thread_id == total_num_threads - 1){
                last_member_index = total_num_members;
            }

            for(int member_index = first_member_index; member_index < last_member_index;member_index++){

                Cost_Change_Move current_cost_of_move = calculate_cost_of_move(clusters_local_copy,cluster_id_1,cluster_id_2,member_index,*all_matching_results);

                if(current_cost_of_move.total_cost_change > best_move_of_cluster.cost_of_move.total_cost_change){
                    best_move_of_cluster.cost_of_move = current_cost_of_move;

                    best_move_of_cluster.cluster_1_id = cluster_id_1;
                    best_move_of_cluster.cluster_2_id = cluster_id_2;

                    best_move_of_cluster.member_in_c1_to_move = member_index;
                }
            }

            write_to_move_cost_matrix_thread_safe(cluster_id_1,cluster_id_2,all_clusters->size(),lock_matrix,best_move_of_cluster,best_moves_per_cluster_matrix);
            // we need to make this write thread safe and 
            //best_moves_per_cluster_matrix[get_index_into_cost_matrix(cluster_id_1,cluster_id_2,all_clusters->size())] = best_move_of_cluster;
        }
    }

}

void update_move_cost_matrix_for_single_cluster_in_active_id_range(int first_index_to_process, int last_index_to_process, int cluster_id,std::vector<int>* currently_active_cluster_ids, Move* move_cost_matrix, std::vector<Cluster>* all_clusters, std::vector<Matching_Result>* all_matching_results){

    std::vector<Cluster> all_clusters_local_copy;

    for(int i = 0; i < all_clusters->size();i++){

        Cluster current_cluster = (*all_clusters)[i];

        Cluster local_copy_of_cluster;

        local_copy_of_cluster.cost_of_cluster = current_cluster.cost_of_cluster;

        local_copy_of_cluster.members = new std::vector<int>;

        for(int l = 0; l < current_cluster.members->size();l++){
            local_copy_of_cluster.members->push_back((*current_cluster.members)[l]);
        }

        all_clusters_local_copy.push_back(local_copy_of_cluster);

    }

    for(int j = first_index_to_process; j < last_index_to_process;j++){

        if(cluster_id == j){
            continue;
        }

        int cluster_id_2 = (*currently_active_cluster_ids)[j];

        // find the best move of one element from cluster 1 into cluster 2
        Move best_move_of_cluster;
        best_move_of_cluster.cost_of_move.total_cost_change = -10000000000000.0;

        for(int member_index = 0; member_index < all_clusters_local_copy[cluster_id].members->size();member_index++){

            Cost_Change_Move current_cost_of_move = calculate_cost_of_move(all_clusters_local_copy,cluster_id,cluster_id_2,member_index,*all_matching_results);

            if(current_cost_of_move.total_cost_change > best_move_of_cluster.cost_of_move.total_cost_change){
                best_move_of_cluster.cost_of_move = current_cost_of_move;

                best_move_of_cluster.cluster_1_id = cluster_id;
                best_move_of_cluster.cluster_2_id = cluster_id_2;

                best_move_of_cluster.member_in_c1_to_move = member_index;
            }
        }

        move_cost_matrix[get_index_into_cost_matrix(cluster_id,cluster_id_2,all_clusters_local_copy.size())] = best_move_of_cluster;


        // now_to the reverse and find the best move from cluster 2 to cluster 1
        best_move_of_cluster.cost_of_move.total_cost_change = -10000000000000.0;

        for(int member_index = 0; member_index < all_clusters_local_copy[cluster_id_2].members->size();member_index++){

            Cost_Change_Move current_cost_of_move = calculate_cost_of_move(all_clusters_local_copy,cluster_id_2,cluster_id,member_index,*all_matching_results);

            if(current_cost_of_move.total_cost_change > best_move_of_cluster.cost_of_move.total_cost_change){
                best_move_of_cluster.cost_of_move = current_cost_of_move;

                best_move_of_cluster.cluster_1_id = cluster_id_2;
                best_move_of_cluster.cluster_2_id = cluster_id;

                best_move_of_cluster.member_in_c1_to_move = member_index;
            }
        }

        move_cost_matrix[get_index_into_cost_matrix(cluster_id_2,cluster_id,all_clusters_local_copy.size())] = best_move_of_cluster;
    }

    for(int i = 0; i < all_clusters_local_copy.size();i++){

        Cluster local_copy_of_cluster = all_clusters_local_copy[i];

        delete(local_copy_of_cluster.members);

    }

    //std::cout << "finished update_move_cost_matrix_for_single_cluster_in_active_id_range" << std::endl;

}

void update_move_cost_matrix_for_single_cluster(int cluster_id,std::vector<int>* currently_active_cluster_ids, Move* move_cost_matrix, std::vector<Cluster>& all_clusters, std::vector<Matching_Result>* all_matching_results){

    unsigned int num_hardware_threads = std::thread::hardware_concurrency();

    std::vector<std::thread*> additional_threads;
    additional_threads.resize(num_hardware_threads - 1);

    int ids_per_thread = currently_active_cluster_ids->size() / num_hardware_threads;

    for(int thread = 0; thread < num_hardware_threads - 1; thread++){

        int first_id_for_thread_to_process = ids_per_thread * thread;
        int last_id_for_thread_to_process = ids_per_thread * (thread + 1);
        
        additional_threads[thread] = new std::thread(update_move_cost_matrix_for_single_cluster_in_active_id_range,first_id_for_thread_to_process,last_id_for_thread_to_process,cluster_id,currently_active_cluster_ids,move_cost_matrix,&all_clusters,all_matching_results);
    }

    int first_id_for_masterthread_to_process = ids_per_thread * (num_hardware_threads - 1);
    int last_id_for_masterthread_to_process = currently_active_cluster_ids->size();

    update_move_cost_matrix_for_single_cluster_in_active_id_range(first_id_for_masterthread_to_process,last_id_for_masterthread_to_process,cluster_id,currently_active_cluster_ids,move_cost_matrix,&all_clusters,all_matching_results);

    for(int thread = 0; thread < num_hardware_threads - 1; thread++){
        additional_threads[thread]->join();
    }

}

void execute_move(std::vector<Cluster>& all_clusters, int cluster_id1, int cluster_id2, int member_in_c1_to_move, Cost_Change_Move& cost_of_move, Move* costs_of_best_moves_matrix, std::vector<int>* currently_active_cluster_ids, std::vector<int>* currently_inactive_cluster_ids, std::vector<Matching_Result>* all_matching_results){
    std::cout << "execute move: member: " << member_in_c1_to_move << " from cluster: " << cluster_id1 << " to cluster: " << cluster_id2 << " with total cost change: " << cost_of_move.total_cost_change << std::endl; 

    all_clusters[cluster_id2].members->push_back((*(all_clusters[cluster_id1].members))[member_in_c1_to_move]);

    all_clusters[cluster_id1].members->erase(all_clusters[cluster_id1].members->begin() + member_in_c1_to_move);

    all_clusters[cluster_id1].cost_of_cluster += cost_of_move.cost_change_cluster_1;
    all_clusters[cluster_id2].cost_of_cluster += cost_of_move.cost_change_cluster_2;

    if(costs_of_best_moves_matrix != nullptr && currently_active_cluster_ids != nullptr && currently_inactive_cluster_ids != nullptr && all_matching_results != nullptr){
        // if cluster 2 was previously empty we have to add one of the currently inactive cluster ids to the active ids and calculate all the costs for this newly activated cluster

        if(all_clusters[cluster_id2].members->size() == 1){

            std::cout << "Cluster previously empty" << std::endl;

            int id_of_newly_activated_cluster = (*currently_inactive_cluster_ids)[currently_inactive_cluster_ids->size() - 1];

            currently_active_cluster_ids->push_back(id_of_newly_activated_cluster);
            currently_inactive_cluster_ids->pop_back();
    
            update_move_cost_matrix_for_single_cluster(id_of_newly_activated_cluster,currently_active_cluster_ids,costs_of_best_moves_matrix,all_clusters,all_matching_results);
        }

        // we have moved a member from cluster 1 to cluster 2 so we now have to update the costs for cluster 2
        update_move_cost_matrix_for_single_cluster(cluster_id2,currently_active_cluster_ids,costs_of_best_moves_matrix,all_clusters,all_matching_results);

        // we also have to update the costs for cluster 1
        update_move_cost_matrix_for_single_cluster(cluster_id1,currently_active_cluster_ids,costs_of_best_moves_matrix,all_clusters,all_matching_results);
    }

    //std::cout << "finished updating costs after move" << std::endl;
}

void join_two_clusters(std::vector<Cluster>& all_clusters, int cluster_id1, int cluster_id2, double cost_of_join, std::vector<int>* valid_cluster_indices, double* cost_matrix, std::vector<Matching_Result>* all_matching_results){

    //std::cout << "perform join of: " << cluster_id1 << " and " << cluster_id2 << std::endl;

    all_clusters[cluster_id1].cost_of_cluster += cost_of_join + all_clusters[cluster_id2].cost_of_cluster;
    all_clusters[cluster_id2].cost_of_cluster = 0.0;

    for(int i = 0; i < all_clusters[cluster_id2].members->size();i++){
        all_clusters[cluster_id1].members->push_back((*(all_clusters[cluster_id2].members))[i]);
    }

    all_clusters[cluster_id2].members->clear();


    if(valid_cluster_indices != nullptr && cost_matrix != nullptr){

        int index_to_delete = -1;

        for(int i = 0; i < valid_cluster_indices->size();i++){
            int valid_cluster_index = (*valid_cluster_indices)[i];

            if(valid_cluster_index == cluster_id2){
                index_to_delete = i;
                continue;
            }

            double new_cost = calculate_cost_for_joining_two_clusters(all_clusters,cluster_id1,valid_cluster_index,*all_matching_results);

            int index_in_cost_matrix = get_index_into_cost_matrix(cluster_id1,valid_cluster_index,all_clusters.size());

            cost_matrix[index_in_cost_matrix] = new_cost;

            index_in_cost_matrix = get_index_into_cost_matrix(valid_cluster_index,cluster_id1,all_clusters.size());

            cost_matrix[index_in_cost_matrix] = new_cost;

        }

        valid_cluster_indices->erase(valid_cluster_indices->begin() + index_to_delete);
    }
}

void print_all_clusters(std::vector<Cluster>& all_clusters){

    double total_cost_of_clustering = 0.0;

    std::cout << "all clusters: " << std::endl;

    for(int i = 0; i < all_clusters.size();i++){

        if(all_clusters[i].members->size() > 0){
            std::cout << "cluster: " << i << " with total cost: " << all_clusters[i].cost_of_cluster << std::endl;
            total_cost_of_clustering += all_clusters[i].cost_of_cluster;
            std::cout << "members: ";

            for(int j = 0; j < all_clusters[i].members->size(); j++){
                std::cout << (*(all_clusters[i].members))[j] << " "; 
            }
            std::cout << std::endl; 
            std::cout << std::endl;

        }else{
            std::cout << "cluster: " << i << " is empty" << std::endl;
            std::cout << std::endl;
        }

    }

    std::cout << std::endl;
    std::cout << "total cost of clustering: " << total_cost_of_clustering << std::endl;

}

void calculate_all_three_elementary_subsets(std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Three_Elementary_Subset>& all_subsets){

    for(int i = 0; i < all_feature_vectors.size() - 2;i++){
        for(int j = i+1; j < all_feature_vectors.size() - 1;j++){
            for(int k = j+1; k < all_feature_vectors.size();k++){
                Three_Elementary_Subset new_subset;
                new_subset.elem_1 = i;
                new_subset.elem_2 = j;
                new_subset.elem_3 = k;
                //std::cout << "(" << new_subset.elem_1 << ","<< new_subset.elem_2 << ","<< new_subset.elem_3 << ")" << std::endl;
                all_subsets.push_back(new_subset);
            }
        }   
    }

}

int calculate_var_index(int x, int y, int num_variables){
    int var_index = 0;

    for (int i = 0; i < x; i++)
    {
        var_index += (num_variables-1) - i;
    }

    var_index += (y-1) - x;
    
    return var_index;
    //std::cout << x << " " << y << " " << var_index << std::endl;

}

std::vector<int>* calculate_clustering_using_gurobi(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters, double threshold){

    //std::cout << "Begin clustering using Gurobi" << std::endl;

    int num_objects = all_feature_vectors.size();

    int num_matching_results = all_matching_results.size();

    int num_required_matching_results = ((num_objects * num_objects) - num_objects) / 2;

    std::vector<Three_Elementary_Subset> all_subsets;

    std::vector<int>* solution = new std::vector<int>; 

    setup_new_clusters(all_matching_results,all_feature_vectors,all_clusters);

    /*
    if(num_required_matching_results != num_matching_results){
        std::cout << "ERROR: the number of matching results: " << num_matching_results << " did not match the expected amount of: " << num_required_matching_results << " given the " << num_objects << " objects!" << std::endl;
        //return solution;
    }
    */

    calculate_all_three_elementary_subsets(all_feature_vectors, all_subsets);

    //std::cout << "num_subsets: " << all_subsets.size() << std::endl;

    //print_all_clusters(all_clusters);

    try {

        // Create an environment
        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag, 1);
        env.set("LogFile", "mip1.log");
        env.start();

        // Create an empty model
        GRBModel model = GRBModel(env);

        //model.getEnv().set(GRB_DoubleParam_TimeLimit, GUROBI_TIMELIMIT_SECONDS);

        GRBVar* all_model_variables = (GRBVar*)malloc(sizeof(GRBVar) * all_matching_results.size());
        double* variable_costs = (double*)malloc(sizeof(double) * all_matching_results.size());

        GRBLinExpr model_objective;

        for(int i = 0; i < num_objects; i++){
            for(int j = i+1; j < num_objects; j++){

                int var_index = calculate_var_index(i,j,num_objects);

                int img_id1 = all_feature_vectors[i].image_number;
                int img_id2 = all_feature_vectors[j].image_number;

                std::string new_variable_name = "x" + std::to_string(i) + "_" + std::to_string(j);
                all_model_variables[var_index] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, new_variable_name);

                float matching_cost = -10000.0;

                for(int k = 0; k < all_matching_results.size();k++){

                    if((img_id1 == all_matching_results[k].id_1 && img_id2 == all_matching_results[k].id_2) ||(img_id1 == all_matching_results[k].id_2 && img_id2 == all_matching_results[k].id_1)){

                        matching_cost = all_matching_results[k].rel_quadr_cost;
                        break;

                    }
                }

                //std::cout << "added new variable: " << new_variable_name << " at index: " << var_index << " with matching: " << img_id1 << " " << img_id2 << " " << matching_cost << std::endl;
                
                variable_costs[var_index] = matching_cost;         
            }

        }

        model.update();

        int num_quadr_expr = 0;

        for(int i = 0; i < all_subsets.size();i++){

            Three_Elementary_Subset current_subset = all_subsets[i];

            int index_of_first_var = calculate_var_index(current_subset.elem_1,current_subset.elem_2,num_objects);
            int index_of_second_var = calculate_var_index(current_subset.elem_1,current_subset.elem_3,num_objects);
            int index_of_third_var = calculate_var_index(current_subset.elem_2,current_subset.elem_3,num_objects);

            model.addConstr(all_model_variables[index_of_first_var] >= all_model_variables[index_of_second_var] + all_model_variables[index_of_third_var] - 1);
            model.addConstr(all_model_variables[index_of_second_var] >= all_model_variables[index_of_first_var] + all_model_variables[index_of_third_var] - 1);
            model.addConstr(all_model_variables[index_of_third_var] >= all_model_variables[index_of_first_var] + all_model_variables[index_of_second_var] - 1);

        }


        model_objective.addTerms(variable_costs,all_model_variables,all_matching_results.size());

        model.setObjective(model_objective,GRB_MAXIMIZE);

        model.update();

        //grb_print_quadratic_expr(&model_objective);
        

        model.optimize();

        std::cout << "Gurobi Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
        //std::cout << "Solution with full constraints" << std::endl;

        for(int i = 0; i < num_objects; i++){
            for(int j = i+1; j < num_objects; j++){

                int var_index = calculate_var_index(i,j,num_objects);

                int img_id1 = all_feature_vectors[i].image_number;
                int img_id2 = all_feature_vectors[j].image_number;

               
                GRBVar current_var = all_model_variables[var_index];

                if(current_var.get(GRB_DoubleAttr_X) < 0.5){
                    solution->push_back(0);
                }else{
                    solution->push_back(1);
                }

                //std::cout << var_index << " " << current_var.get(GRB_DoubleAttr_X) << std::endl;
                int cluster_id1 = -1;
                int cluster_id2 = -1;

                if(current_var.get(GRB_DoubleAttr_X) > 0.5){
                    
                    for(int k = 0; k < all_clusters.size();k++){

                        Cluster current_cluster = all_clusters[k];

                        for(int l = 0; l < current_cluster.members->size();l++){
                            int current_cluster_member = (*(current_cluster.members))[l];

                            if(img_id1 == current_cluster_member){
                                cluster_id1 = k;
                            }

                            if(img_id2 == current_cluster_member){
                                cluster_id2 = k;
                            }

                            if(cluster_id1 != -1 && cluster_id2 != -1){
                                goto loop_end;
                            }

                        }
                    }

                    loop_end:

                    if(cluster_id1 != cluster_id2){
                        float cost_of_join = calculate_cost_for_joining_two_clusters(all_clusters,cluster_id1,cluster_id2,all_matching_results);
                        join_two_clusters(all_clusters,cluster_id1,cluster_id2,cost_of_join);
                    }

                    /*
                    if(cluster_id1 != cluster_id2 && cluster_id1 != -1 && cluster_id2 != -1){
                        std::cout << "Detected cluster assignment conflict for images: " << img_id1 << " and " << img_id2 << std::endl;
                    }else if(cluster_id1 == -1 && cluster_id2 == -1){
                        Cluster new_cluster;
                        new_cluster.members = new std::vector<int>;

                        new_cluster.members->push_back(img_id1);
                        new_cluster.members->push_back(img_id2);

                        all_clusters.push_back(new_cluster);
                    }else if(cluster_id1 == -1){
                        all_clusters[cluster_id2].members->push_back(img_id1);
                    }else if(cluster_id2 == -1){
                        all_clusters[cluster_id1].members->push_back(img_id2);
                    }
                    */
                    //print_all_clusters(all_clusters);

                }


                //std::cout << img_id1 << " " << img_id2 << " " << same_cluster << std::endl;
                
        
            }

        }

        delete_empty_clusters(all_clusters);

        //print_all_clusters(all_clusters);

        free(all_model_variables);
        free(variable_costs);

        //free(quadratic_expressions_var1);
        //free(quadratic_expressions_var2);
        //free(quadratic_expressions_costs);


    }
    catch (GRBException e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch (...) {
        std::cout << "Exception during optimization" << std::endl;
    }

    //std::cout << "Gurobi : " << calculate_cost_of_clustering(all_clusters) << std::endl;

    return solution;

}

class Violated_Inequalites_Callback: public GRBCallback
{
    public:
    GRBVar* vars;
    int n_variables;
    int n_objects;
    std::vector<Three_Elementary_Subset>* all_three_elementary_subsets;
    Violated_Inequalites_Callback(GRBVar* _vars, int _n_vars, int _n_objs,  std::vector<Three_Elementary_Subset>* subsets) {
        vars = _vars;
        n_variables = _n_vars;
        n_objects = _n_objs;
        all_three_elementary_subsets = subsets;
    }

    protected:
        void callback() {
            try {
                if (where == GRB_CB_MIPSOL) {

                    // Found an integer feasible solution - does it visit every node?
                    double *x = new double[n_variables];
                    int i, j, len;

                    for (i = 0; i < n_variables; i++){
                        x[i] = getSolution(vars[i]);
                        //std::cout << i << " " << x[i] << std::endl;
                    }

                    bool found_violated_inequality = false;

                    for(int i = 0; i < all_three_elementary_subsets->size();i++){

                        Three_Elementary_Subset current_subset = (*all_three_elementary_subsets)[i];

                        int index_of_first_var = calculate_var_index(current_subset.elem_1,current_subset.elem_2,n_objects);
                        int index_of_second_var = calculate_var_index(current_subset.elem_1,current_subset.elem_3,n_objects);
                        int index_of_third_var = calculate_var_index(current_subset.elem_2,current_subset.elem_3,n_objects);

                        double u = x[index_of_first_var];
                        double v = x[index_of_second_var];
                        double w = x[index_of_third_var];


                        if(u < v + w - 1){
                            //std::cout << "found violated inequality" << std::endl;
                            found_violated_inequality = true;
                            addLazy(vars[index_of_first_var] >= vars[index_of_second_var] + vars[index_of_third_var] - 1);
                        }

                        if(v < u + w - 1){
                            //std::cout << "found violated inequality" << std::endl
                            found_violated_inequality = true;
                            addLazy(vars[index_of_second_var] >= vars[index_of_first_var] + vars[index_of_third_var] - 1);
                        }

                        if(w < u + v - 1){
                            //std::cout << "found violated inequality" << std::endl;
                            found_violated_inequality = true;
                            addLazy(vars[index_of_third_var] >= vars[index_of_first_var] + vars[index_of_second_var] - 1);
                        }

                    }

                    delete[] x;

                    if(found_violated_inequality){
                        //std::cout << "found violated inequality" << std::endl;
                    }else{
                        double best_objective = getDoubleInfo(GRB_CB_MIPSOL_OBJBST);
                        double best_bound = getDoubleInfo(GRB_CB_MIPSOL_OBJBND);

                        double gap = fabs((best_objective - best_bound) / best_objective);

                        double runtime = getDoubleInfo(GRB_CB_RUNTIME);
                        
                        //std::cout << "runtime: " << runtime << " gap: " << gap << std::endl;

                        if(runtime > CLUSTERING_WITH_LAZY_CONSTRAINTS_SOFT_TIMELIMIT && gap < CLUSTERING_WITH_LAZY_CONSTRAINTS_GAP_MINIMUM){
                            std::cout << "reached termination criteria" << std::endl;
                            abort();
                        }

                    }
                }
            } catch (GRBException e) {
                std::cout << "Error number: " << e.getErrorCode() << std::endl;
                std::cout << e.getMessage() << std::endl;
            } catch (...) {
                std::cout << "Error during callback" << std::endl;
            }
        }
};

std::vector<int>* calculate_clustering_using_gurobi_with_lazy_constraints(std::vector<Matching_Result>& all_matching_results, std::vector<Image_Features_Pair>& all_feature_vectors, std::vector<Cluster>& all_clusters, double threshold){

    //std::cout << std::endl;
    //std::cout << "Begin clustering using Gurobi with lazy constraints" << std::endl;

    int num_objects = all_feature_vectors.size();

    int num_matching_results = all_matching_results.size();

    int num_required_matching_results = ((num_objects * num_objects) - num_objects) / 2;

    std::vector<Three_Elementary_Subset> all_subsets;

    std::vector<int>* solution = new std::vector<int>; 

    setup_new_clusters(all_matching_results,all_feature_vectors,all_clusters);
     

    if(num_required_matching_results != num_matching_results){
        std::cout << "ERROR: the number of matching results: " << num_matching_results << " did not match the expected amount of: " << num_required_matching_results << " given the " << num_objects << " objects!" << std::endl;
        return solution;
    }

    calculate_all_three_elementary_subsets(all_feature_vectors, all_subsets);

    //print_all_clusters(all_clusters);

    try {

        // Create an environment
        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag, 0);
        env.set("LogFile", "mip1.log");
        env.start();

        // Create an empty model
        GRBModel model = GRBModel(env);

        //model.getEnv().set(GRB_DoubleParam_TimeLimit, CLUSTERING_WITH_LAZY_CONSTRAINTS_HARD_TIMELIMIT);
        model.set(GRB_IntParam_LazyConstraints,1);

        GRBVar* all_model_variables = (GRBVar*)malloc(sizeof(GRBVar) * all_matching_results.size());
        double* variable_costs = (double*)malloc(sizeof(double) * all_matching_results.size());

        GRBLinExpr model_objective;

        for(int i = 0; i < num_objects; i++){
            for(int j = i+1; j < num_objects; j++){

                int var_index = calculate_var_index(i,j,num_objects);

                int img_id1 = all_feature_vectors[i].image_number;
                int img_id2 = all_feature_vectors[j].image_number;

                std::string new_variable_name = "x" + std::to_string(i) + "_" + std::to_string(j);
                all_model_variables[var_index] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, new_variable_name);

                float matching_cost = -10000.0;

                for(int k = 0; k < all_matching_results.size();k++){

                    if((img_id1 == all_matching_results[k].id_1 && img_id2 == all_matching_results[k].id_2) ||(img_id1 == all_matching_results[k].id_2 && img_id2 == all_matching_results[k].id_1)){

                        matching_cost = all_matching_results[k].rel_quadr_cost;
                        break;

                    }
                }

                //std::cout << "added new variable: " << new_variable_name << " at index: " << var_index << " with matching: " << img_id1 << " " << img_id2 << " " << matching_cost << std::endl;
                
                variable_costs[var_index] = matching_cost;         
            }

        }

        model.update();

        /*
        for(int i = 0; i < all_subsets.size();i++){

            Three_Elementary_Subset current_subset = all_subsets[i];

            int index_of_first_var = calculate_var_index(current_subset.elem_1,current_subset.elem_2,num_objects);
            int index_of_second_var = calculate_var_index(current_subset.elem_1,current_subset.elem_3,num_objects);
            int index_of_third_var = calculate_var_index(current_subset.elem_2,current_subset.elem_3,num_objects);

            model.addConstr(all_model_variables[index_of_first_var] >= all_model_variables[index_of_second_var] + all_model_variables[index_of_third_var] - 1);
            model.addConstr(all_model_variables[index_of_second_var] >= all_model_variables[index_of_first_var] + all_model_variables[index_of_third_var] - 1);
            model.addConstr(all_model_variables[index_of_third_var] >= all_model_variables[index_of_first_var] + all_model_variables[index_of_second_var] - 1);

        }
        */
        


        model_objective.addTerms(variable_costs,all_model_variables,all_matching_results.size());

        Violated_Inequalites_Callback cb = Violated_Inequalites_Callback(all_model_variables,num_matching_results,num_objects,&all_subsets);
        model.setCallback(&cb);          

        model.setObjective(model_objective,GRB_MAXIMIZE);

        model.update();

        //grb_print_quadratic_expr(&model_objective);
      

        model.optimize();

        //std::cout << "Gurobi Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

        std::string solution_file_name = "gurobi_solution_" + std::to_string(threshold) + ".sol";

        model.write(solution_file_name);

        //std::cout << "Solution with lazy constraints" << std::endl;

        for(int i = 0; i < num_objects; i++){
            for(int j = i+1; j < num_objects; j++){

                int var_index = calculate_var_index(i,j,num_objects);

                int img_id1 = all_feature_vectors[i].image_number;
                int img_id2 = all_feature_vectors[j].image_number;

               
                GRBVar current_var = all_model_variables[var_index];

                //std::cout << var_index << " " << current_var.get(GRB_DoubleAttr_X) << std::endl;  

                if(current_var.get(GRB_DoubleAttr_X) < 0.5){
                    solution->push_back(0);
                }else{
                    solution->push_back(1);
                }
        

                int cluster_id1 = -1;
                int cluster_id2 = -1;

                if(current_var.get(GRB_DoubleAttr_X) > 0.5){
                    
                    for(int k = 0; k < all_clusters.size();k++){

                        Cluster current_cluster = all_clusters[k];

                        for(int l = 0; l < current_cluster.members->size();l++){
                            int current_cluster_member = (*(current_cluster.members))[l];

                            if(img_id1 == current_cluster_member){
                                cluster_id1 = k;
                            }

                            if(img_id2 == current_cluster_member){
                                cluster_id2 = k;
                            }

                            if(cluster_id1 != -1 && cluster_id2 != -1){
                                goto loop_end;
                            }

                        }
                    }

                    loop_end:

                    if(cluster_id1 != cluster_id2){
                        float cost_of_join = calculate_cost_for_joining_two_clusters(all_clusters,cluster_id1,cluster_id2,all_matching_results);
                        join_two_clusters(all_clusters,cluster_id1,cluster_id2,cost_of_join);
                    }

                }
     
            }

        }

        delete_empty_clusters(all_clusters);

        //print_all_clusters(all_clusters);

        free(all_model_variables);
        free(variable_costs);

        //free(quadratic_expressions_var1);
        //free(quadratic_expressions_var2);
        //free(quadratic_expressions_costs);


    }
    catch (GRBException e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch (...) {
        std::cout << "Exception during optimization" << std::endl;
    }

    //std::cout << "Gurobi with lazy constraints: " << calculate_cost_of_clustering(all_clusters) << std::endl;


    return solution;
}

void delete_empty_clusters(std::vector<Cluster>& all_clusters){

     for(int i = 0; i < all_clusters.size(); i++){

        if(all_clusters[i].members->size() == 0){
            delete(all_clusters[i].members);
            
            all_clusters.erase(all_clusters.begin() + i);
            i--;
        } 

    }
}

void write_to_move_cost_matrix_thread_safe(int cluster_id_1, int cluster_id_2, int total_num_initial_clusters, std::vector<std::mutex*>* lock_matrix, Move& best_move_of_cluster, Move* best_moves_matrix){

    int index_into_matrices = get_index_into_cost_matrix(cluster_id_1,cluster_id_2,total_num_initial_clusters);

    std::lock_guard<std::mutex> lock(*((*lock_matrix)[index_into_matrices]));

    if(best_moves_matrix[index_into_matrices].cost_of_move.total_cost_change < best_move_of_cluster.cost_of_move.total_cost_change){
        best_moves_matrix[index_into_matrices] = best_move_of_cluster;
    }

}

void find_organoid_cluster_representative(std::vector<Matching_Result>& all_matching_results, std::vector<Cluster>& all_clusters, std::vector<Cluster_Representative_Pair>& selected_cluster_representatives){

    std::cout << "List of Cluster Representatives:" << std::endl;

    for(int i = 0; i < all_clusters.size();i++){
        Cluster current_cluster = all_clusters[i];

        int index_of_best_cluster_representative = -1;
        float total_similarity_to_cluster_mems_of_rep = -1.0f;

        for(int j = 0; j < current_cluster.members->size();j++){
            float similarity_of_current_mem = sum_similarities_to_cluster_members(all_matching_results,current_cluster,j);

            if(similarity_of_current_mem > total_similarity_to_cluster_mems_of_rep){
                total_similarity_to_cluster_mems_of_rep = similarity_of_current_mem;
                index_of_best_cluster_representative = j;
            }
        }

        Cluster_Representative_Pair new_crp;
        new_crp.cluster_index = i;
        new_crp.representative_img_number = (*(current_cluster.members))[index_of_best_cluster_representative];

        selected_cluster_representatives.push_back(new_crp);
        //std::cout << (*(current_cluster.members))[index_of_best_cluster_representative] << std::endl;
    }


}

float sum_similarities_to_cluster_members(std::vector<Matching_Result>& all_matching_results, Cluster& current_cluster, int base_member_id){

    int base_img_id = (*(current_cluster.members))[base_member_id];

    float sum_of_similarities = 0.0f;

    for(int i = 0 ; i < current_cluster.members->size();i++){
        if(i == base_member_id){
            continue;
        }

        int current_img_id = (*(current_cluster.members))[i];

        Matching_Result mr_with_base = get_matching_result_by_image_ids(all_matching_results,base_img_id,current_img_id,false);

        sum_of_similarities += mr_with_base.rel_quadr_cost;
    }

    return sum_of_similarities;

}

