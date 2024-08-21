#pragma once
#include "utils.h"

#define ENTRIES_PER_BITMAP_WORD 64

#define MEM_ELEM_ACCESS(mem_elem,data_type,data_index) ((data_type*)(mem_elem.memory_pointer))[data_index]
#define MEM_ELEM_WRITE(mem_elem,data_type,data_index,new_value) (((data_type*)(mem_elem.memory_pointer))[data_index])=new_value

enum Hierarchical_Bitmap_Node_Types{
    BITMAP_LEAF_NODE,
    BITMAP_INTERIOR_NODE
};

typedef struct Memory_Element{
    char* memory_pointer;
    uint32_t bit_index;
    uint32_t leaf_index;
}Memory_Element;

typedef struct Memory_Chunk{
    char* memory;
    size_t memory_chunk_size;
    uint32_t first_leaf_index_in_chunk; 
}Memory_Chunk;

typedef struct Hierarchical_Bitmap_Interior_Node{
    uint32_t parent_index;
    uint32_t own_index;
    uint64_t bitmap;
    uint64_t initialization_bitmap;
    uint32_t child_indices[ENTRIES_PER_BITMAP_WORD];
    uint8_t index_in_parent_bitmap;
    uint8_t level_in_hierarchy;
}Hierarchical_Bitmap_Interior_Node;

typedef struct Hierarchical_Bitmap_Leaf_Node{
    uint32_t parent_index;
    uint32_t own_index;
    uint64_t bitmap;
    uint8_t index_in_parent_bitmap;
    uint8_t memory_chunk_index;
}Hierarchical_Bitmap_Leaf_Node;


typedef struct Memory_Manager_Fixed_Size{
    //Hierarchical_Bitmap_Interior_Node* memory_bitmap_root;

    std::vector<Hierarchical_Bitmap_Interior_Node>* all_interior_nodes;
    std::vector<Hierarchical_Bitmap_Leaf_Node>* all_leaf_nodes;
    uint32_t root_index;

    uint32_t index_of_currently_used_leaf_node;

    uint32_t num_created_leaf_nodes;
    uint32_t num_created_interior_nodes;

    std::vector<Memory_Chunk>* memory_chunks;

    size_t size_of_single_element;

    uint64_t (*mark_next_free_spot_in_bitmap_word_function)(uint64_t*,int&);
    int (*get_index_of_next_free_spot_in_bitmap_word_function)(uint64_t*);
}Memory_Manager_Fixed_Size;

Memory_Element allocate_new_memory_element(Memory_Manager_Fixed_Size* memory_manager);
char* allocate_new_memory_slot(Memory_Manager_Fixed_Size* memory_manager);

void free_memory_element(Memory_Manager_Fixed_Size* memory_manager, Memory_Element* mem_elem);
void free_memory_slot(Memory_Manager_Fixed_Size* memory_manager, char* mem_slot);

void change_single_elem_size_and_reset_allocation(Memory_Manager_Fixed_Size* memory_manager, size_t new_single_elem_size);

void free_all_memory_elements(Memory_Manager_Fixed_Size* memory_manager);

Memory_Manager_Fixed_Size* init_fixed_size_memory_manager(size_t size_of_single_element, size_t initial_number_off_elements);

void destroy_fixed_size_memory_manager(Memory_Manager_Fixed_Size** memory_manager);