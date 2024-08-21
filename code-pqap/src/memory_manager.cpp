#include "memory_manager.h"
#include <math.h>

extern bool global_activate_debug_prints;

#ifdef _WIN64
#include <intrin.h>
#include <immintrin.h>
#endif

#define ENTRIES_PER_BITMAP_WORD 64
#define ENTRIES_PER_BITMAP_WORD_POWER_OF_TWO 6

#define BITMAP64_ALL_ONES 0xFFFFFFFFFFFFFFFF

void set_single_bit_to_zero(uint64_t* word, unsigned int bit_index);
void set_single_bit_to_one(uint64_t* word, unsigned int bit_index);

uint64_t mark_next_free_spot_in_bitmap_word_unix(uint64_t* memory_bitmap, int& index);
uint64_t mark_next_free_spot_in_bitmap_word_windows(uint64_t* memory_bitmap, int& index);

int get_index_of_next_free_spot_in_bitmap_word_unix(uint64_t* memory_bitmap);
int get_index_of_next_free_spot_in_bitmap_word_windows(uint64_t* memory_bitmap);


void propagate_zeros_up(uint32_t node_index, uint8_t index_in_bitmap, Memory_Manager_Fixed_Size* memory_manager);
void propagate_ones_up(uint32_t node_index, uint8_t index_in_bitmap, Memory_Manager_Fixed_Size* memory_manager);

void add_range_of_new_nodes_to_root(Memory_Manager_Fixed_Size* memory_manager, int start, int end);

void increase_total_allocated_memory(Memory_Manager_Fixed_Size* memory_manager);

void recursively_initialize_bitmap_node(uint32_t node_index, Memory_Manager_Fixed_Size* memory_manager);

bool find_and_mark_leaf_node_with_free_memory(Memory_Manager_Fixed_Size* memory_manager, uint32_t& output_leaf_node_index, uint32_t& output_bit_index);

char* get_memory_location_from_leaf_node_index_and_bit_index(Memory_Manager_Fixed_Size* memory_manager, uint32_t leaf_node_index, int bit_index);

Memory_Manager_Fixed_Size* init_fixed_size_memory_manager(size_t size_of_single_element, size_t initial_number_off_elements){

    Memory_Manager_Fixed_Size* new_memory_manager = (Memory_Manager_Fixed_Size*)malloc(sizeof(Memory_Manager_Fixed_Size));

    new_memory_manager->memory_chunks = new std::vector<Memory_Chunk>;
    new_memory_manager->all_interior_nodes = new std::vector<Hierarchical_Bitmap_Interior_Node>;
    new_memory_manager->all_leaf_nodes = new std::vector<Hierarchical_Bitmap_Leaf_Node>;

    new_memory_manager->mark_next_free_spot_in_bitmap_word_function = nullptr;

    #ifdef __unix__
        new_memory_manager->mark_next_free_spot_in_bitmap_word_function = mark_next_free_spot_in_bitmap_word_unix;
        new_memory_manager->get_index_of_next_free_spot_in_bitmap_word_function = get_index_of_next_free_spot_in_bitmap_word_unix;
    #endif

    #ifdef _WIN64
        new_memory_manager->mark_next_free_spot_in_bitmap_word_function = mark_next_free_spot_in_bitmap_word_windows;
        new_memory_manager->get_index_of_next_free_spot_in_bitmap_word_function = get_index_of_next_free_spot_in_bitmap_word_windows;
    #endif

    new_memory_manager->num_created_leaf_nodes = 0;
    new_memory_manager->num_created_interior_nodes = 0;  


    initial_number_off_elements = get_next_bigger_power_of_two(initial_number_off_elements);

    //std::cout << "initial num _elements: " << initial_number_off_elements << std::endl;



    new_memory_manager->size_of_single_element = size_of_single_element;

    size_t num_needed_levels_in_bitmap_hierarchy = 1;

    while(pow(ENTRIES_PER_BITMAP_WORD, num_needed_levels_in_bitmap_hierarchy) < initial_number_off_elements){
        //std::cout << (ENTRIES_PER_BITMAP_WORD << (ENTRIES_PER_BITMAP_WORD_POWER_OF_TWO * num_needed_levels_in_bitmap_hierarchy)) << std::endl;
        num_needed_levels_in_bitmap_hierarchy++;
    }

    //std::cout << "num needed levels: " << num_needed_levels_in_bitmap_hierarchy << std::endl;

    unsigned int num_entries_per_next_sublevel_node = pow(ENTRIES_PER_BITMAP_WORD, num_needed_levels_in_bitmap_hierarchy - 1);

    unsigned int num_needed_entries_in_root = initial_number_off_elements / num_entries_per_next_sublevel_node;

    if(!check_if_uint_is_power_of_two(num_needed_entries_in_root)){
        // we always want their to be a power of 2 num entries in the root 
        // this simplifies the allocation of new memory since we only have two cases:
        // either there are still uninitialized entries in the root avilable in which case we make new entries until we have doubled the number of entries
        // or if the root has no more space for additional entries we make a new root node at the old root node to its children and allocate and additional entry in the new root
        // besides the old root which again doubles the number of leaf nodes
        // in both cases we need to allocate a new memory chunk and copy the content of the old one into the new one. 
        num_needed_entries_in_root = get_next_bigger_power_of_two(num_needed_entries_in_root);
    }

    //std::cout << "num_entries_per_next_sublevel_node: " << num_entries_per_next_sublevel_node << std::endl;
    //std::cout << "num_needed_entries_in_root: " << num_needed_entries_in_root << std::endl;



    Hierarchical_Bitmap_Interior_Node root_node;
    root_node.bitmap = 0;
    root_node.initialization_bitmap = 0;

    root_node.index_in_parent_bitmap = 0;
    root_node.own_index = 0;
    root_node.parent_index = 0;

    root_node.level_in_hierarchy = num_needed_levels_in_bitmap_hierarchy - 1;

    if(root_node.level_in_hierarchy < 1){
        root_node.level_in_hierarchy = 1;
    }

    new_memory_manager->all_interior_nodes->push_back(root_node);
    new_memory_manager->num_created_interior_nodes++;

    new_memory_manager->root_index = 0;

    if(root_node.level_in_hierarchy == 1){
        ((*(new_memory_manager->all_interior_nodes))[root_node.own_index]).bitmap = BITMAP64_ALL_ONES;
        ((*(new_memory_manager->all_interior_nodes))[root_node.own_index]).initialization_bitmap = BITMAP64_ALL_ONES;

        //recursively_initialize_bitmap_node(&((*(new_memory_manager->all_interior_nodes))[root_node.own_index]),new_memory_manager);
        recursively_initialize_bitmap_node(root_node.own_index,new_memory_manager);
    }else{
        
        add_range_of_new_nodes_to_root(new_memory_manager,0,num_needed_entries_in_root);
    }

	//std::cout << "starting to allocate mem chunk" << std::endl;

    Memory_Chunk new_memory_chunk;

    new_memory_chunk.first_leaf_index_in_chunk = 0;
    new_memory_chunk.memory_chunk_size = new_memory_manager->num_created_leaf_nodes * ENTRIES_PER_BITMAP_WORD * new_memory_manager->size_of_single_element;
    new_memory_chunk.memory = (char*)malloc(new_memory_chunk.memory_chunk_size);

	//std::cout << "finished allocating mem_chunk" << std::endl;

    new_memory_manager->memory_chunks->push_back(new_memory_chunk);

    new_memory_manager->index_of_currently_used_leaf_node = 0;

    //std::cout << "Finished initializing the memory manager. Num leaf nodes: " << new_memory_manager->num_created_leaf_nodes << ". Num interior nodes: " << new_memory_manager->num_created_interior_nodes << std::endl;

    return new_memory_manager;
}

void add_range_of_new_nodes_to_root(Memory_Manager_Fixed_Size* memory_manager, int start, int end){
    //std::cout << "pointer to memory_manager: " << memory_manager << std::endl;
    //std::cout << "root node index: " << memory_manager->root_index << std::endl;

    Hierarchical_Bitmap_Interior_Node root_node = (*(memory_manager->all_interior_nodes))[memory_manager->root_index];


    //std::cout << "root node own index: " << root_node.own_index << std::endl;

    for(int i = start; i < end;i++){
		//std::cout << "child node of root " << i << std::endl;
            Hierarchical_Bitmap_Interior_Node new_interior_node;

            new_interior_node.bitmap = BITMAP64_ALL_ONES;
            new_interior_node.initialization_bitmap = BITMAP64_ALL_ONES;
            new_interior_node.index_in_parent_bitmap = i;
            new_interior_node.level_in_hierarchy = root_node.level_in_hierarchy - 1;

            set_single_bit_to_one(&((*(memory_manager->all_interior_nodes))[root_node.own_index].bitmap),i);
            set_single_bit_to_one(&((*(memory_manager->all_interior_nodes))[root_node.own_index].initialization_bitmap),i);
		//std::cout << "after set single bit" << std::endl;
            ((*(memory_manager->all_interior_nodes))[root_node.own_index]).child_indices[i] = memory_manager->num_created_interior_nodes;
            new_interior_node.own_index = memory_manager->num_created_interior_nodes;
		//std::cout << "SET new_interior_node.own_index to: " << new_interior_node.own_index << std::endl;
		//std::cout << "after child indices" << std::endl;
            new_interior_node.parent_index = root_node.own_index;

            memory_manager->all_interior_nodes->push_back(new_interior_node);
            memory_manager->num_created_interior_nodes++;
		//std::cout << "after push new root child to interior nodes" << std::endl;
            //recursively_initialize_bitmap_node(&((*(memory_manager->all_interior_nodes))[new_interior_node.own_index]),memory_manager);
            recursively_initialize_bitmap_node(new_interior_node.own_index,memory_manager);
        }

	//std::cout << "FINISHED add_nodes_to_rootsnode" << std::endl;
}

void free_all_memory_elements(Memory_Manager_Fixed_Size* memory_manager){

        // we will set all the bitmaps off all nodes to ALL_ONES indicating that they are now again free.

    for(int i = 0; i < memory_manager->all_interior_nodes->size();i++){

        (*(memory_manager->all_interior_nodes))[i].bitmap = BITMAP64_ALL_ONES;
    }

    for(int i = 0; i < memory_manager->all_leaf_nodes->size();i++){

        (*(memory_manager->all_leaf_nodes))[i].bitmap = BITMAP64_ALL_ONES;
    }

    (*(memory_manager->all_interior_nodes))[memory_manager->root_index].bitmap = (*(memory_manager->all_interior_nodes))[memory_manager->root_index].initialization_bitmap;
    memory_manager->index_of_currently_used_leaf_node = 0;


}

void change_single_elem_size_and_reset_allocation(Memory_Manager_Fixed_Size* memory_manager, size_t new_single_elem_size){

    free_all_memory_elements(memory_manager);

    // Then we reallocate the memory of the memory chunks s.t. we have enough memory to save the elements with the new single_elem_size

    for(int i = 0; i < memory_manager->memory_chunks->size();i++){
        free((*(memory_manager->memory_chunks))[i].memory);

        size_t new_memory_chunk_size = ((*(memory_manager->memory_chunks))[i].memory_chunk_size / memory_manager->size_of_single_element) * new_single_elem_size;
        (*(memory_manager->memory_chunks))[i].memory = (char*)malloc(new_memory_chunk_size);
        (*(memory_manager->memory_chunks))[i].memory_chunk_size = new_memory_chunk_size;
    }

    memory_manager->size_of_single_element = new_single_elem_size;
}

void increase_total_allocated_memory(Memory_Manager_Fixed_Size* memory_manager){

    uint32_t num_leaf_nodes_before_new_allocations = memory_manager->num_created_leaf_nodes; 

    uint64_t negated_initialization_bitmap = ~((*(memory_manager->all_interior_nodes))[memory_manager->root_index].initialization_bitmap);

    int next_free_slot = memory_manager->get_index_of_next_free_spot_in_bitmap_word_function(&negated_initialization_bitmap);

    if(next_free_slot != -1){
        std::cout << "Doubling memory available in root node" << std::endl;
        int doubled_index = next_free_slot << 1;
        
        add_range_of_new_nodes_to_root(memory_manager,next_free_slot,doubled_index);

    }else{

        Hierarchical_Bitmap_Interior_Node new_root_node;

        std::cout << "Allocating new root node" << std::endl;

        // the bit map of the new_root_node is zero because the old root node which will be the frist child of the new root node is already completly full
        // but ofcourse it is already properly initialized  and we can set int initialization_bitmap to 1 i.t set the first bit of the initialization_bitmap to 1
        new_root_node.bitmap = 0;
        new_root_node.initialization_bitmap = 1;

        new_root_node.index_in_parent_bitmap = 0;
        new_root_node.own_index = memory_manager->num_created_interior_nodes;
        // TODO: see if we can find a nice way of encoding that the root node has no parent instead of setting the parent index to its own index

        new_root_node.parent_index = memory_manager->num_created_interior_nodes;

        
        new_root_node.level_in_hierarchy = (*(memory_manager->all_interior_nodes))[memory_manager->root_index].level_in_hierarchy + 1;

        new_root_node.child_indices[0] = memory_manager->root_index;

        (*(memory_manager->all_interior_nodes))[memory_manager->root_index].index_in_parent_bitmap = 0;
        (*(memory_manager->all_interior_nodes))[memory_manager->root_index].parent_index = new_root_node.own_index;


        memory_manager->all_interior_nodes->push_back(new_root_node);
        memory_manager->num_created_interior_nodes++;

        memory_manager->root_index = new_root_node.own_index;


        // add one new node to the new root besides the old root the effectively double the available memory. 
        add_range_of_new_nodes_to_root(memory_manager,1,2);
    }

    Memory_Chunk new_memory_chunk;

    new_memory_chunk.first_leaf_index_in_chunk = num_leaf_nodes_before_new_allocations;
    new_memory_chunk.memory_chunk_size = (memory_manager->num_created_leaf_nodes - num_leaf_nodes_before_new_allocations)  * ENTRIES_PER_BITMAP_WORD * memory_manager->size_of_single_element;
    new_memory_chunk.memory = (char*)malloc(new_memory_chunk.memory_chunk_size);

    memory_manager->memory_chunks->push_back(new_memory_chunk);

}


bool find_and_mark_leaf_node_with_free_memory(Memory_Manager_Fixed_Size* memory_manager, uint32_t& output_leaf_node_index, uint32_t& output_bit_index){

    Hierarchical_Bitmap_Interior_Node current_node = (*(memory_manager->all_interior_nodes))[memory_manager->root_index];

    uint32_t leaf_node_index = memory_manager->index_of_currently_used_leaf_node;


    Hierarchical_Bitmap_Leaf_Node last_used_leaf_node = (*(memory_manager->all_leaf_nodes))[leaf_node_index];

    // Firstly we check if their is still memory available in the leaf_node we have used the last time we had to allocate memory
    // if that is the case we can immediatly jump to the end and calculate the memory addess
    if(last_used_leaf_node.bitmap == 0){
        //If the current leaf_node is full then maybe the next leaf node still has free memory

        uint32_t next_leaf_node_index = (leaf_node_index + 1) % memory_manager->num_created_leaf_nodes;
        Hierarchical_Bitmap_Leaf_Node next_after_last_used_leaf_node = (*(memory_manager->all_leaf_nodes))[next_leaf_node_index];
        if(next_after_last_used_leaf_node.bitmap != 0){
            leaf_node_index = next_leaf_node_index;
            memory_manager->index_of_currently_used_leaf_node = leaf_node_index;
        }else{
            // if this also is not the case then we need to traverse the hierarchy until we find a leaf node with free memory
            //std::cout << "needed to search" << std::endl;
            while(current_node.level_in_hierarchy > 1){
                int index = memory_manager->get_index_of_next_free_spot_in_bitmap_word_function(&current_node.bitmap);
                
                if(index == -1){
                    std::cout << "index in bitmask was -1" << std::endl;
                    return false; 
                }

                uint32_t child_node_index = current_node.child_indices[index];

                current_node = (*(memory_manager->all_interior_nodes))[child_node_index];
            }

            //get the leaf node
            int leaf_index = memory_manager->get_index_of_next_free_spot_in_bitmap_word_function(&current_node.bitmap);
                
            if(leaf_index == -1){
                std::cout << "leaf_index in bitmask was -1" << std::endl;
                return false; 
            }

            leaf_node_index = current_node.child_indices[leaf_index];

            memory_manager->index_of_currently_used_leaf_node = leaf_node_index;

        }

    }

    Hierarchical_Bitmap_Leaf_Node leaf_node = (*(memory_manager->all_leaf_nodes))[leaf_node_index];

    //find a spot in the bit map of the leaf node that is marked as 1 i.e still free
    int bit_index = -1;

    memory_manager->mark_next_free_spot_in_bitmap_word_function(&(*(memory_manager->all_leaf_nodes))[leaf_node_index].bitmap,bit_index);

    if(bit_index == -1){
        std::cout << "bit index in leaf node was -1" << std::endl;
        return false;
    }

    if((*(memory_manager->all_leaf_nodes))[leaf_node_index].bitmap == 0){
        //std::cout << "propagate zero" << std::endl; 
        propagate_zeros_up(leaf_node.parent_index,leaf_node.index_in_parent_bitmap,memory_manager);

    }


    output_leaf_node_index = leaf_node_index;
    output_bit_index = bit_index;

    return true;

}

Memory_Element allocate_new_memory_element(Memory_Manager_Fixed_Size* memory_manager){
    Memory_Element mem_elem;
    mem_elem.bit_index = 0;
    mem_elem.leaf_index = 0;
    mem_elem.memory_pointer = nullptr;

    bool success = find_and_mark_leaf_node_with_free_memory(memory_manager,mem_elem.leaf_index,mem_elem.bit_index);
    
    if(!success){
        std::cout << "ERROR in allocate_new_memory_element" << std::endl;
        return mem_elem;
    }

    mem_elem.memory_pointer = get_memory_location_from_leaf_node_index_and_bit_index(memory_manager, mem_elem.leaf_index, mem_elem.bit_index);

    return mem_elem;
}

char* allocate_new_memory_slot(Memory_Manager_Fixed_Size* memory_manager){
    Memory_Element mem_elem;
    mem_elem.bit_index = 0;
    mem_elem.leaf_index = 0;
    mem_elem.memory_pointer = nullptr;

    bool success = find_and_mark_leaf_node_with_free_memory(memory_manager,mem_elem.leaf_index,mem_elem.bit_index);
    
    if(!success){
        std::cout << "ERROR in allocate_new_memory_element" << std::endl;
        return nullptr;
    }

    return get_memory_location_from_leaf_node_index_and_bit_index(memory_manager, mem_elem.leaf_index, mem_elem.bit_index);
}

void free_memory_element(Memory_Manager_Fixed_Size* memory_manager, Memory_Element* mem_elem){


    uint64_t mem_bitmap = (*(memory_manager->all_leaf_nodes))[mem_elem->leaf_index].bitmap;
    /*
    if(global_activate_debug_prints){
        std::cout << "node_id: " << mem_elem->leaf_index << " bitmap: " << mem_bitmap ;
    }
    */
    set_single_bit_to_one(&((*(memory_manager->all_leaf_nodes))[mem_elem->leaf_index].bitmap),mem_elem->bit_index);

    /*
    if(global_activate_debug_prints){
        std::cout << "  index to set to one: " << (int)mem_elem->bit_index  << " resulting bitmap: " << (*(memory_manager->all_leaf_nodes))[mem_elem->leaf_index].bitmap << std::endl;
    }
    */
    if(mem_bitmap == 0){
        propagate_ones_up((*(memory_manager->all_leaf_nodes))[mem_elem->leaf_index].parent_index,(*(memory_manager->all_leaf_nodes))[mem_elem->leaf_index].index_in_parent_bitmap,memory_manager);
    }


}

void free_memory_slot(Memory_Manager_Fixed_Size* memory_manager, char* mem_slot){

    int offset_from_base = mem_slot - (*(memory_manager->memory_chunks))[0].memory;

    uint32_t leaf_id = offset_from_base / (ENTRIES_PER_BITMAP_WORD * memory_manager->size_of_single_element);

    int offset_from_leaf_base = mem_slot - ((*(memory_manager->memory_chunks))[0].memory + leaf_id * (ENTRIES_PER_BITMAP_WORD * memory_manager->size_of_single_element));

    uint32_t bit_id = offset_from_leaf_base / memory_manager->size_of_single_element;


    uint64_t mem_bitmap = (*(memory_manager->all_leaf_nodes))[leaf_id].bitmap;

    set_single_bit_to_one(&((*(memory_manager->all_leaf_nodes))[leaf_id].bitmap),bit_id);

    if(mem_bitmap == 0){
        propagate_ones_up((*(memory_manager->all_leaf_nodes))[leaf_id].parent_index,(*(memory_manager->all_leaf_nodes))[leaf_id].index_in_parent_bitmap,memory_manager);
    }

}

char* get_memory_location_from_leaf_node_index_and_bit_index(Memory_Manager_Fixed_Size* memory_manager, uint32_t leaf_node_index, int bit_index){
    //for each leaf node there are 64 (ENTRIES_PER_BITMAP_WORD) many entries and each is the size of the size_of_single_element variable

    uint8_t memory_chunk_index = (*(memory_manager->all_leaf_nodes))[leaf_node_index].memory_chunk_index;

    uint32_t leaf_node_index_in_memory_chunk = leaf_node_index - (*(memory_manager->memory_chunks))[memory_chunk_index].first_leaf_index_in_chunk;

    uint64_t offset_into_memory_chunk = leaf_node_index_in_memory_chunk * ENTRIES_PER_BITMAP_WORD * memory_manager->size_of_single_element + bit_index * memory_manager->size_of_single_element;

    return (*(memory_manager->memory_chunks))[memory_chunk_index].memory + offset_into_memory_chunk;

}

void destroy_fixed_size_memory_manager(Memory_Manager_Fixed_Size** memory_manager){

    //delete((*memory_manager)->memory_bitmap);
    //delete((*memory_manager)->memory);

    //std::cout << "Deleting fixed size memory manager" << std::endl;

    delete((*memory_manager)->all_interior_nodes);
    delete((*memory_manager)->all_leaf_nodes);


    for(int i = 0; i < (*memory_manager)->memory_chunks->size();i++){
        free((*((*memory_manager)->memory_chunks))[i].memory);
    }


    //delete((*memory_manager)->memory_chunks);

    free(*memory_manager);

    memory_manager = nullptr;

}

void recursively_initialize_bitmap_node(uint32_t node_index, Memory_Manager_Fixed_Size* memory_manager){
	
    Hierarchical_Bitmap_Interior_Node* node = &(*(memory_manager->all_interior_nodes))[node_index];

    if((*(memory_manager->all_interior_nodes))[node_index].level_in_hierarchy == 1){
        // we have reached the lowest level in the hierarchy and now need to initialize the leaf nodes;
        for(int i = 0; i < ENTRIES_PER_BITMAP_WORD;i++){
		//std::cout << i << std::endl;
            Hierarchical_Bitmap_Leaf_Node new_leaf_node;

            new_leaf_node.bitmap = BITMAP64_ALL_ONES;

            new_leaf_node.index_in_parent_bitmap = i;
            
            //set_single_bit_to_one(&((*(memory_manager->all_interior_nodes))[node->own_index].bitmap),i);
            //set_single_bit_to_one(&((*(memory_manager->all_interior_nodes))[node->own_index].initialization_bitmap),i);

            (*(memory_manager->all_interior_nodes))[node_index].child_indices[i] = memory_manager->num_created_leaf_nodes;
		//std::cout << "after set child indices of parent" << std::endl;
            new_leaf_node.parent_index = node_index;
            new_leaf_node.own_index = memory_manager->num_created_leaf_nodes;

            new_leaf_node.memory_chunk_index = memory_manager->memory_chunks->size();

            memory_manager->all_leaf_nodes->push_back(new_leaf_node);
		//std::cout << "after push" << std::endl;
            memory_manager->num_created_leaf_nodes++;
		//std::cout << "after increment lead nodes "<< std::endl;
        }
        return;

    }else{

        // we are still somewhere in the hierarchy and therefore need to initialize all the new child interior nodes
        for(int i = 0; i < ENTRIES_PER_BITMAP_WORD;i++){
            Hierarchical_Bitmap_Interior_Node new_interior_node;

            new_interior_node.bitmap = BITMAP64_ALL_ONES;
            new_interior_node.initialization_bitmap = BITMAP64_ALL_ONES;
            new_interior_node.index_in_parent_bitmap = i;
            new_interior_node.level_in_hierarchy = (*(memory_manager->all_interior_nodes))[node_index].level_in_hierarchy - 1;
	
            ((*(memory_manager->all_interior_nodes))[node_index]).child_indices[i] = memory_manager->num_created_interior_nodes;
            new_interior_node.own_index = memory_manager->num_created_interior_nodes;
            new_interior_node.parent_index = (*(memory_manager->all_interior_nodes))[node_index].own_index;

            memory_manager->all_interior_nodes->push_back(new_interior_node);
            memory_manager->num_created_interior_nodes++;

            //recursively_initialize_bitmap_node(&((*(memory_manager->all_interior_nodes))[new_interior_node.own_index]),memory_manager);
            recursively_initialize_bitmap_node(new_interior_node.own_index,memory_manager);
        }

    }

}

void propagate_zeros_up(uint32_t node_index, uint8_t index_in_bitmap, Memory_Manager_Fixed_Size* memory_manager){
    
    if(global_activate_debug_prints){
        std::cout << "bitmap: " << (*(memory_manager->all_interior_nodes))[node_index].bitmap ;
    }
    

    set_single_bit_to_zero(&(*(memory_manager->all_interior_nodes))[node_index].bitmap,index_in_bitmap);

    
    if(global_activate_debug_prints){
        std::cout << "  index to set to zero: " << (int)index_in_bitmap << " resulting bitmap: " << (*(memory_manager->all_interior_nodes))[node_index].bitmap << std::endl;
    }
    

    if((*(memory_manager->all_interior_nodes))[node_index].bitmap == 0){
        if(node_index == memory_manager->root_index){
            std::cout << "ALL MEMORY HAS BEEN USED UP!" << std::endl;
            increase_total_allocated_memory(memory_manager);
        }else{
            propagate_zeros_up((*(memory_manager->all_interior_nodes))[node_index].parent_index,(*(memory_manager->all_interior_nodes))[node_index].index_in_parent_bitmap,memory_manager);
        }

    }

}

void propagate_ones_up(uint32_t node_index, uint8_t index_in_bitmap, Memory_Manager_Fixed_Size* memory_manager){
    //we get the mem bitmap before we change it
    uint64_t mem_bitmap = (*(memory_manager->all_interior_nodes))[node_index].bitmap;

    /*
    if(global_activate_debug_prints){
        std::cout << "node_id: " << node_index << " bitmap: " << mem_bitmap ;
    }
    */
    // set the the bit in question
    set_single_bit_to_one(&((*(memory_manager->all_interior_nodes))[node_index].bitmap),index_in_bitmap);

    /*
    if(global_activate_debug_prints){
        std::cout << "  index to set to one: " << (int)index_in_bitmap << " resulting bitmap: " << (*(memory_manager->all_interior_nodes))[node_index].bitmap << std::endl;
    }
    */

    //if the bitmap was 0 before we set the bit then this means we had no memory available before and now after setting the bit we again have one element available.
    // hence we need to mark in the bitmap of the parent that this child again has memory available
    if(mem_bitmap == 0){
        if(node_index != memory_manager->root_index){
            propagate_ones_up((*(memory_manager->all_interior_nodes))[node_index].parent_index,(*(memory_manager->all_interior_nodes))[node_index].index_in_parent_bitmap,memory_manager);
        }
    }
}

void set_single_bit_to_zero(uint64_t* word, unsigned int bit_index){
    uint64_t mask = uint64_t(1) << bit_index;

    *word &= ~mask;

}

void set_single_bit_to_one(uint64_t* word, unsigned int bit_index){
    uint64_t mask = uint64_t(1) << bit_index;

    *word |= mask;
}

#ifdef __unix__
uint64_t mark_next_free_spot_in_bitmap_word_unix(uint64_t* memory_bitmap, int& index){

    int index_of_first_free_spot = ffsll(*memory_bitmap);

    index = index_of_first_free_spot - 1;

    uint64_t mask = uint64_t(1) << index;

    *memory_bitmap &= ~mask;

    return mask;
}

int get_index_of_next_free_spot_in_bitmap_word_unix(uint64_t* memory_bitmap){
    return ffsll(*memory_bitmap) - 1;
}
#endif

#ifdef _WIN64
uint64_t mark_next_free_spot_in_bitmap_word_windows(uint64_t* memory_bitmap, int& index) {
    //uint64_t mask = _blsi_u64(*memory_bitmap);

    //*memory_bitmap &= ~mask;

    //return mask;
    
    //std::cout << "mask: " << mask << std::endl;

    unsigned long pos = 0;
    int is_none_zero = _BitScanForward64(&pos, *memory_bitmap);

    if (is_none_zero) {
        uint64_t mask = uint64_t(1) << pos;

        index = pos;

        *memory_bitmap &= ~mask;

        return mask;

    }else {
        index = -1;
        return 0;
    }
    
}

int get_index_of_next_free_spot_in_bitmap_word_windows(uint64_t* memory_bitmap){
    unsigned long pos = 0;
    int is_none_zero = _BitScanForward64(&pos, *memory_bitmap);

    if (is_none_zero) {
        return pos;
    }else {
        return -1;
    }
}
#endif



