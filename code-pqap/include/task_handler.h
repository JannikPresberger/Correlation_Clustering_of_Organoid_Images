#pragma once
#include <queue>
#include <mutex>
#include <thread>

#include "global_data_types.h"

void sleep_and_print_input_function(void* input);

typedef struct Thread_Task{

    void* thread_task_data;
    void (*thread_task_function)(void*);
    double thread_task_priority;
}Thread_Task;

typedef struct Thread_Task_Compare{

    bool operator()(const Thread_Task& lhs, const Thread_Task& rhs){

        return lhs.thread_task_priority < rhs.thread_task_priority;
    }
}Thread_Task_Compare;

typedef struct Task_Handler{
    std::priority_queue<Thread_Task,std::vector<Thread_Task>,Thread_Task_Compare>* task_queue;

    std::mutex* task_queue_mutex;

}Task_Handler;

Task_Handler* init_new_task_handler(Task_Vector& tasks_to_complete);

void delete_task_handler(Task_Handler** thread_handler);

Thread_Task request_task_from_task_handler(Task_Handler* thread_handler);

void process_task_queue_thread_wrapper(Task_Handler* thread_handler, std::vector<int>* output);