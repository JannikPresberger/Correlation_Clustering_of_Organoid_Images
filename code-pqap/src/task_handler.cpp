#include "task_handler.h"
#include <chrono>
#include <iostream>

void sleep_and_print_input_function(void* input){

    std::this_thread::sleep_for(std::chrono::microseconds(1));

}

Task_Handler* init_new_task_handler(Task_Vector& tasks_to_complete){
    Task_Handler* new_task_handler = new Task_Handler;

    new_task_handler->task_queue = new std::priority_queue<Thread_Task,std::vector<Thread_Task>,Thread_Task_Compare>;
    new_task_handler->task_queue_mutex = new std::mutex;


    for(int i = 0; i < tasks_to_complete.tasks->size();i++){

        Matching_Calculation_Task current_task = (*(tasks_to_complete.tasks))[i];

        Matching_Calculation_Task* persistent_matching_task = new Matching_Calculation_Task;
        *persistent_matching_task = current_task;

        Thread_Task new_thread_task;
        new_thread_task.thread_task_data = persistent_matching_task;
        new_thread_task.thread_task_function = nullptr;
        new_thread_task.thread_task_priority = current_task.runtime_estimation;

        new_task_handler->task_queue->push(new_thread_task);
    }

    return new_task_handler;
}

void delete_task_handler(Task_Handler** task_handler){
    delete((*task_handler)->task_queue);
    delete((*task_handler)->task_queue_mutex);

    delete(*task_handler);

}

Thread_Task request_task_from_task_handler(Task_Handler* task_handler){

    std::lock_guard<std::mutex> lock(*(task_handler->task_queue_mutex));

    Thread_Task task_to_hand_over;
    task_to_hand_over.thread_task_data = nullptr;
    task_to_hand_over.thread_task_function = nullptr;
    task_to_hand_over.thread_task_priority = -1;

    if(!task_handler->task_queue->empty()){

        task_to_hand_over = task_handler->task_queue->top();
        task_handler->task_queue->pop();

    }

    return task_to_hand_over;

}

void process_task_queue_thread_wrapper(Task_Handler* task_handler, std::vector<int>* output){

    while(!task_handler->task_queue->empty()){

        Thread_Task next_task = request_task_from_task_handler(task_handler);

        output->push_back(*((int*)next_task.thread_task_data));

        next_task.thread_task_function(next_task.thread_task_data);

        delete((int*)next_task.thread_task_data);
    }


}