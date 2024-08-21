#include "python_handler.h"

//#include <chrono>

#define MIN_RELATIVE_PEAK_VALUE 0.05

#define SHOW_CPP_PYTHON_DEBUG false

#define CPP_PYTHON_DEBUG(expr) if(SHOW_CPP_PYTHON_DEBUG){expr}

void check_and_print_python_error() {
    if (PyErr_Occurred()) {
        PyErr_Print();
    }

}

Python_Handler* init_python_handler(std::filesystem::path python_scripts_folder_path) {
	Python_Handler* new_python_handler = (Python_Handler*)malloc(sizeof(Python_Handler));

	new_python_handler->nuclei_detection_function = nullptr;
    new_python_handler->organoid_detection_function = nullptr;
    new_python_handler->qap_solver_function = nullptr;

    const char* python_module_name = "image_segmentation_python_module";
    const char* python_nuclei_detection_function_name = "detect_nuclei_in_organoid_image";
    const char* python_organoid_detection_function_name = "detect_organoids_in_microscopy_image";
    //const char* python_path = "C:/Users/Jannik-Laptop/Documents/master_thesis_organoid/master_thesis_organoid/src/python_scripts";

    std::string python_module_path = python_scripts_folder_path.string();
    std::replace(python_module_path.begin(), python_module_path.end(), '\\', '/');
    //std::cout << python_module_path << std::endl;

    const char* qap_python_module_name = "qap_solver_python_module";
    const char* python_qap_solver_function_name = "solve_qap";

    std::filesystem::path qap_module_filepath = python_scripts_folder_path;
    qap_module_filepath.append("qap");

    std::string qap_python_module_path = qap_module_filepath.string();
    std::replace(qap_python_module_path.begin(), qap_python_module_path.end(), '\\', '/');


    Py_Initialize();

    PyObject* path_list = PySys_GetObject("path");
    if (!path_list) {
        std::cerr << "Invalid return value from PySys_GetObject(path)" << std::endl;
        return nullptr;
    }

    //PyObject* python_readable_path_name = PyUnicode_DecodeFSDefault(python_scripts_folder_path.string().c_str());
    PyObject* python_readable_path_name = PyUnicode_DecodeFSDefault(python_module_path.c_str());
    if (!python_readable_path_name) {
        std::cerr << "Could not decode path" << std::endl;
        return nullptr;
    }

    //std::cout << _PyUnicode_AsString(python_readable_path_name) << std::endl;

    int python_append_error = PyList_Append(path_list, python_readable_path_name);
    if (python_append_error) {
        std::cerr << "Could not append path name to path list" << std::endl;
        return nullptr;
    }

    python_readable_path_name = PyUnicode_DecodeFSDefault(qap_python_module_path.c_str());
    if (!python_readable_path_name) {
        std::cerr << "Could not decode path" << std::endl;
        return nullptr;
    }

    python_append_error = PyList_Append(path_list, python_readable_path_name);
    if (python_append_error) {
        std::cerr << "Could not append path name to path list" << std::endl;
        return nullptr;
    }


    Py_DECREF(python_readable_path_name);



    PyObject* python_readable_module_name = PyUnicode_DecodeFSDefault(python_module_name);
    if (!python_readable_module_name) {
        std::cerr << "Could not decode module name" << std::endl;
        return nullptr;
    }

    PyObject* python_module = PyImport_Import(python_readable_module_name);
    Py_DECREF(python_readable_module_name);
    if (!python_module) {
        std::cerr << "could not import python module: " << python_module_name << std::endl;
        check_and_print_python_error();
        return nullptr;
    }

    new_python_handler->nuclei_detection_function = PyObject_GetAttrString(python_module, python_nuclei_detection_function_name);
    new_python_handler->organoid_detection_function = PyObject_GetAttrString(python_module, python_organoid_detection_function_name);

    Py_DECREF(python_module);
    if (!new_python_handler->nuclei_detection_function) {
        std::cerr << "Could not find the function: " << python_nuclei_detection_function_name << " in module: " << python_module_name;
        return nullptr;
    }

    if (!PyCallable_Check(new_python_handler->nuclei_detection_function)) {
        std::cerr << "Py_Object: " << python_nuclei_detection_function_name << "is not callable" << std::endl;
        Py_DECREF(new_python_handler->nuclei_detection_function);
        return nullptr;
    }


    if (!new_python_handler->organoid_detection_function) {
        std::cerr << "Could not find the function: " << python_organoid_detection_function_name << " in module: " << python_module_name;
        return nullptr;
    }

    if (!PyCallable_Check(new_python_handler->organoid_detection_function)) {
        std::cerr << "Py_Object: " << python_organoid_detection_function_name << "is not callable" << std::endl;
        Py_DECREF(new_python_handler->organoid_detection_function);
        return nullptr;
    }



    PyObject* python_readable_qap_module_name = PyUnicode_DecodeFSDefault(qap_python_module_name);
    if (!python_readable_qap_module_name) {
        std::cerr << "Could not decode module name" << std::endl;
        return nullptr;
    }

    bool found_qap_solver_module = false;

    PyObject* python_qap_module = PyImport_Import(python_readable_qap_module_name);
    Py_DECREF(python_readable_qap_module_name);
    if (!python_qap_module) {
        //std::cerr << "could not import python module: " << qap_python_module_name << std::endl;
        check_and_print_python_error();
        //return nullptr;
    }else{
        found_qap_solver_module = true;
    }

    if(found_qap_solver_module){

        new_python_handler->qap_solver_function = PyObject_GetAttrString(python_qap_module, python_qap_solver_function_name);


        Py_DECREF(python_qap_module);
        if (!new_python_handler->qap_solver_function) {
            std::cerr << "Could not find the function: " << python_qap_solver_function_name << " in module: " << qap_python_module_name;
            return nullptr;
        }

        if (!PyCallable_Check(new_python_handler->qap_solver_function)) {
            std::cerr << "Py_Object: " << python_qap_solver_function_name << "is not callable" << std::endl;
            Py_DECREF(new_python_handler->qap_solver_function);
            return nullptr;
        }
    }


     

    //necessary call for the C/Python API
    import_array();

	return new_python_handler;

}

void python_extract_feature_points(Python_Handler* py_handler, cv::Mat* organoid_image, std::vector<Feature_Point_Data>& feature_points, double nms_thresh, Channel_Type channel) {

    //std::cout << organoid_image->rows << " " << organoid_image->cols << std::endl;
    //std::cout << "Org Elem size: " << organoid_image->elemSize() << " Elem size 1: " << organoid_image->elemSize1() << std::endl;

    //auto t_start = std::chrono::high_resolution_clock::now();

    if(py_handler == nullptr){
        std::cout << "Python Handler was NULL in python_extract_feature_points" << std::endl;
        return;
    }

    int elem_size = organoid_image->elemSize1();

    npy_intp dims[2] = { organoid_image->rows, organoid_image->cols };
    //npy_intp dims[2] = { 512, 512 };

    uint16_t* matrix_data_pointer = organoid_image->ptr<uint16_t>(0);

    cv::Mat output_label_img(organoid_image->size(), CV_8U);

    //cv::Mat test(cv::Size(512, 512), CV_64FC1, cv::Scalar(0));
    //matrix_data_pointer = test.ptr<uint16_t>(0);

    PyObject* py_array;

    int array_elem_type = NPY_UINT8;

    if (elem_size == 1) {
        array_elem_type = NPY_UINT8;
    }else if (elem_size == 2) {
        array_elem_type = NPY_UINT16;
    }else {
        std::cerr << "unsupported element size while constructing function arguments in C++ to pass to Python function. Element size was: " << elem_size << std::endl;
        return;
    }

    py_array = PyArray_SimpleNewFromData(2, dims, array_elem_type, matrix_data_pointer);

    if (!py_array) {
        std::cerr << "Could not create python array from matrix data" << std::endl;
        return;
    }

    PyObject* python_function_argument_nms_thresh = PyFloat_FromDouble(nms_thresh);
    //PyTuple_SetItem(python_function_arguments, 0, py_array);
    //PyTuple_SetItem(python_function_arguments, 1, PyFloat_FromDouble(0.5));




    //PyObject* return_value_from_python_function = PyObject_CallObject(py_handler->nuclei_detection_function, python_function_arguments);
    
    PyObject* return_value_from_python_function = PyObject_CallFunctionObjArgs(py_handler->nuclei_detection_function, py_array,python_function_argument_nms_thresh,NULL);


    //return_value_from_python_function = PyObject_CallObject(py_handler->nuclei_detection_function, python_function_arguments);
    //Py_DECREF(python_add_function);

    //

    if (!return_value_from_python_function) {
        std::cerr << "call to function failed" << std::endl;
        check_and_print_python_error();
        return;
    }

    /*
    t_end = std::chrono::high_resolution_clock::now();

    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout << "total elapsed time for function call: " << elapsed_time_ms << std::endl;
    */
    PyArrayObject* python_return_label_np_array = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(return_value_from_python_function, 0));
    PyArrayObject* python_return_center_points_np_array = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(return_value_from_python_function, 1));

    int num_points = static_cast<int>(PyArray_SHAPE(python_return_center_points_np_array)[0]);

    int elem_per_point = static_cast<int>(PyArray_SHAPE(python_return_center_points_np_array)[1]);
    
    //std::cout << "in C++: " << num_points << " coords per point : " << elem_per_point << std::endl;

    //long double* c_output = reinterpret_cast<long double*>(PyArray_DATA(python_return_np_array));

    uint64_t* c_center_point_output = reinterpret_cast<uint64_t*>(PyArray_DATA(python_return_center_points_np_array));

    int32_t* c_label_output = reinterpret_cast<int32_t*>(PyArray_DATA(python_return_label_np_array));


    for (int i = 0; i < num_points ; i++) {

        Feature_Point_Data new_feature_point_data;//{channel, c_center_point_output[i * elem_per_point],c_center_point_output[i * elem_per_point + 1],0.0,0.0,0.0,0.0,0,0,0.0f,0.0f,0.0f };

            new_feature_point_data.channel = channel;
            new_feature_point_data.row = (uint16_t)c_center_point_output[i * elem_per_point];
            new_feature_point_data.col = (uint16_t)c_center_point_output[i * elem_per_point + 1];;
            new_feature_point_data.peak_value = 0;
            new_feature_point_data.normalize_peak_value = 0.0;
            new_feature_point_data.local_search_dim = 0;

            new_feature_point_data.local_mean = 0;
            new_feature_point_data.local_std_dev = 0;

            new_feature_point_data.local_mean_forground_only = 0;
            new_feature_point_data.local_std_dev_forground_only = 0; 

            new_feature_point_data.angle = 0.0f;
            new_feature_point_data.relative_distance_center_boundary = 0.0f;
            new_feature_point_data.relative_distance_center_max_distance = 0.0f;


        feature_points.push_back(new_feature_point_data);
    }

    //std::cout << "num_feature_points extracted by StarDist" << num_points << std::endl;


    for (int output_rows = 0; output_rows < organoid_image->rows; output_rows++) {
        for (int output_cols = 0; output_cols < organoid_image->cols; output_cols++) {

            //std::cout << output_cols << " " << output_rows << std::endl;



            int32_t current_output_label = c_label_output[output_rows * organoid_image->cols + output_cols];

            //std::cout << current_output_label << std::endl;
            //cv::Vec3s output_color = colors[current_output_label];
            //std::cout << output_cols << " " << output_rows << " label: " << current_output_label << std::endl;

            if (current_output_label > 0) {
                Feature_Point_Data current_feature_point = feature_points.at(current_output_label - 1);
                feature_points.at(current_output_label - 1).local_mean += (double)organoid_image->at<uint16_t>(output_rows, output_cols) / (double)256;
                feature_points.at(current_output_label - 1).peak_value++;


                // we only use the local_mean_forground_only and local_std_dev_forground_only variables to temprarily store the accumulated rows and cols
                feature_points.at(current_output_label - 1).local_mean_forground_only += output_rows;
                feature_points.at(current_output_label - 1).local_std_dev_forground_only += output_cols;
                //int diff_rows = abs(output_rows - current_feature_point.row);
                //int diff_cols = abs(output_cols - current_feature_point.col);

                //output_label_img.at<uint8_t>(output_rows, output_cols) = (double)organoid_image->at<uint16_t>(output_rows, output_cols) / (double)256;//feature_points.at(current_output_label - 1).local_mean;//std::max<int>(diff_rows, diff_cols) * 10;
            }/*
            else {
                output_label_img.at<uint8_t>(output_rows, output_cols) = 0;
            }*/

        }

    }

    double total_peak_values = 0;

    for (int i = 0; i < feature_points.size(); i++) {
        double num_pixels_of_organoid = (double)feature_points.at(i).peak_value;

        feature_points.at(i).peak_value = feature_points.at(i).local_mean / (double)feature_points.at(i).peak_value;

        feature_points.at(i).normalize_peak_value = (double)feature_points.at(i).peak_value / 256.0;

        total_peak_values += feature_points.at(i).peak_value;
        feature_points.at(i).local_mean = 0.0;

        feature_points.at(i).row = feature_points.at(i).local_mean_forground_only / num_pixels_of_organoid;
        feature_points.at(i).col = feature_points.at(i).local_std_dev_forground_only / num_pixels_of_organoid;

        feature_points.at(i).local_mean_forground_only = 0.0;
        feature_points.at(i).local_std_dev_forground_only = 0.0;
        //std::cout << feature_points.at(i).peak_value << " " << feature_points.at(i).normalize_peak_value << std::endl;

    }

    double avg_peak_value = total_peak_values / feature_points.size();


    /*
    for (int i = 0; i < num_points; i++) {
        std::cout << feature_points.at(i).peak_value << " " << avg_peak_value << " " << (double)feature_points.at(i).peak_value / avg_peak_value << std::endl;
    }*/

    //std::cout << organoid_image->rows << " " << organoid_image->cols << std::endl;

    for (int output_rows = 0; output_rows < organoid_image->rows; output_rows++) {
        for (int output_cols = 0; output_cols < organoid_image->cols; output_cols++) {

            int32_t current_output_label = c_label_output[output_rows * organoid_image->cols + output_cols];

            if (current_output_label > 0) {
                Feature_Point_Data current_feature_point = feature_points.at(current_output_label - 1);

 

                output_label_img.at<uint8_t>(output_rows, output_cols) = 127 * std::min<double>(2.0, (double)feature_points.at(current_output_label - 1).peak_value / avg_peak_value);
            } else {
                 output_label_img.at<uint8_t>(output_rows, output_cols) = 0;
            }

        }

    }

    static bool colors_initialized = false;

    static std::vector<cv::Vec3b> colors(256);

    if(!colors_initialized){
        colors[0] = cv::Vec3b(0, 0, 0);//background
        for (int label = 1; label < 256; ++label) {
            colors[label] = cv::Vec3b(255 * sample_uniform_0_1(), 255 * sample_uniform_0_1(), 255 * sample_uniform_0_1());
        }
        colors_initialized = true;
    }
    
    CPP_PYTHON_DEBUG(

        static int img_num = 0;

        cv::namedWindow("Nuclei Labels", cv::WINDOW_KEEPRATIO);
        cv::imshow("Nuclei Labels", output_label_img);

        int key_code = (cv::waitKey(0) & 0xEFFFFF);

        if (key_code == 27) {
            

            cv::Mat coloured_label_output(output_label_img.size(), CV_8UC3);

            for (int r = 0; r < coloured_label_output.rows; ++r) {
                for (int c = 0; c < coloured_label_output.cols; ++c) {
                    int label = output_label_img.at<uint8_t>(r, c);
                    cv::Vec3b& pixel = coloured_label_output.at<cv::Vec3b>(r, c);

                    float ratio = (float)label / 255.0f;

                    if (ratio < 0.4f) {
                        ratio = 0.4f;
                    }
                     
                    pixel = ratio * colors[label];
                }
                //std::cout << std::endl;
            }

            std::string org_img_string = "nuclei_original_image_" + std::to_string(img_num) + ".png"; 
            std::string feature_img_string = std::to_string(nms_thresh) + "_nuclei_feature_" + std::to_string(img_num) + ".png";

            for (int i = 0; i < num_points ; i++) {

                Feature_Point_Data current_feature_point_data = feature_points[i];
                
                cv::circle(coloured_label_output, cv::Point(current_feature_point_data.col, current_feature_point_data.row), 3, cv::Scalar(0,0,255), -1);

            }

            cv::imwrite(org_img_string, *organoid_image);
            cv::imwrite(feature_img_string, coloured_label_output);

            img_num++;
        }
    )

    //std::cout << "C++: " << PyLong_AsLong(return_value_from_python_function) << std::endl;

    for (int i = 0; i < feature_points.size(); i++) {
        if (feature_points.at(i).normalize_peak_value < MIN_RELATIVE_PEAK_VALUE) {
            feature_points.erase(feature_points.begin() + i);
            i--;
        }
    }


    Py_DECREF(return_value_from_python_function);
    Py_DECREF(py_array);
    Py_DECREF(python_function_argument_nms_thresh);

    /*
    t_end = std::chrono::high_resolution_clock::now();

    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout << "total elapsed time to call python nuclei segmentation function was: " << elapsed_time_ms << std::endl << std::endl;
    */
}


int python_extract_organoid_labels(Python_Handler* py_handler, cv::Mat* organoid_image, cv::Mat* output_label_image, double nms_thresh) {

    if(py_handler == nullptr){
        std::cout << "Python Handler was NULL in python_extract_feature_points" << std::endl;
        return 0;
    }

    int elem_size = organoid_image->elemSize1();

    npy_intp dims[2] = { organoid_image->rows, organoid_image->cols };
    //npy_intp dims[2] = { 512, 512 };

    uint16_t* matrix_data_pointer = organoid_image->ptr<uint16_t>(0);


    PyObject* py_array;

    int array_elem_type = NPY_UINT8;

    if (elem_size == 1) {
        array_elem_type = NPY_UINT8;
    }
    else if (elem_size == 2) {
        array_elem_type = NPY_UINT16;
    }
    else {
        std::cerr << "unsupported element size while constructing function arguments in C++ to pass to Python function. Element size was: " << elem_size << std::endl;
        return 0;
    }

    py_array = PyArray_SimpleNewFromData(2, dims, array_elem_type, matrix_data_pointer);

    if (!py_array) {
        std::cerr << "Could not create python array from matrix data" << std::endl;
        return 0;
    }

    PyObject* python_function_argument_nms_thresh = PyFloat_FromDouble(nms_thresh);

    PyObject* return_value_from_python_function = PyObject_CallFunctionObjArgs(py_handler->organoid_detection_function, py_array, python_function_argument_nms_thresh, NULL);


    if (!return_value_from_python_function) {
        std::cerr << "call to function failed" << std::endl;
        check_and_print_python_error();
        return 0;
    }

    //PyTupleObject* python_return_tuple = reinterpret_cast<PyTupleObject*>(return_value_from_python_function);

    PyArrayObject* python_return_np_array = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(return_value_from_python_function,0));

    uint64_t num_differnet_labels = PyLong_AsLong(PyTuple_GetItem(return_value_from_python_function, 1));

    int num_points = static_cast<int>(PyArray_SHAPE(python_return_np_array)[0]);

    int elem_per_point = static_cast<int>(PyArray_SHAPE(python_return_np_array)[1]);

    int32_t* c_output = reinterpret_cast<int32_t*>(PyArray_DATA(python_return_np_array));


    for (int output_rows = 0; output_rows < organoid_image->rows; output_rows++) {
        for (int output_cols = 0; output_cols < organoid_image->cols; output_cols++) {

            int32_t current_output_label = c_output[output_rows * organoid_image->cols + output_cols];

            if (current_output_label > 0) {
                output_label_image->at<uint16_t>(output_rows, output_cols) = current_output_label;
            }
            else {
                output_label_image->at<uint16_t>(output_rows, output_cols) = 0;
            }
            
        }

    }

    Py_DECREF(return_value_from_python_function);
    Py_DECREF(py_array);
    Py_DECREF(python_function_argument_nms_thresh);

    return num_differnet_labels;

}

void python_solve_qap(Python_Handler* py_handler,std::vector<int>* solution_instance, int num_keypoints, int num_candiates, int num_unary_expr, int num_quadr_expr, double* unary_costs, double* quadr_costs, int* edge_indices){

    if(py_handler->qap_solver_function == nullptr){
        CPP_PYTHON_DEBUG(std::cout << "qap_solver_function of py_handler was NULL in python_solve_qap" << std::endl;)
        return;
    }

    PyObject* py_header_array;

    npy_intp dims[1] = {4};

    int array_elem_type = NPY_UINT64;

    uint64_t header_data[4] = {(uint64)num_keypoints,(uint64)num_candiates,(uint64)num_unary_expr,(uint64)num_quadr_expr};

    py_header_array = PyArray_SimpleNewFromData(1, dims, array_elem_type, header_data);

    if (!py_header_array) {
        std::cerr << "Could not create python array from matrix data" << std::endl;
        return;
    }

    PyObject* py_unary_costs_array;
    dims[0] = num_unary_expr;
    py_unary_costs_array = PyArray_SimpleNewFromData(1,dims,NPY_DOUBLE,unary_costs);
    
    std::cout << "finished creating unary cost array" << std::endl;

    PyObject* py_quadr_costs_array;
    dims[0] = num_quadr_expr;
    py_quadr_costs_array = PyArray_SimpleNewFromData(1,dims,NPY_DOUBLE,quadr_costs);

    std::cout << "finished creating quadr cost array" << std::endl;

    PyObject* py_quadr_edge_indices_array;
    dims[0] = num_quadr_expr * 2;
    py_quadr_edge_indices_array = PyArray_SimpleNewFromData(1,dims,NPY_INT32,edge_indices);

    std::cout << "finished creating edge indices array" << std::endl;

    PyObject* return_value_from_python_function = PyObject_CallFunctionObjArgs(py_handler->qap_solver_function, py_header_array, py_unary_costs_array,py_quadr_costs_array,py_quadr_edge_indices_array,NULL);

    if(!return_value_from_python_function){
        std::cout << "Did not receive return_value_from_python_function in python_solve_qap:" << std::endl;
        check_and_print_python_error();
        return;
    }

    Py_DecRef(py_unary_costs_array);
    Py_DecRef(py_header_array);
    Py_DecRef(py_quadr_edge_indices_array);


    Py_ssize_t list_size = PyList_Size(return_value_from_python_function);


    for(int i = 0; i < list_size;i++){
        PyObject* py_list_item = PyList_GetItem(return_value_from_python_function,i);

        if(!py_list_item){
            std::cout << "could not retrieve Item from PyList" << std::endl;
            check_and_print_python_error();
        }
        
        PyTypeObject* py_list_item_type = py_list_item->ob_type;

        if(strcmp(py_list_item->ob_type->tp_name,"NoneType") == 0){
            solution_instance->push_back(-1);
        }else{

            long item_as_long = PyLong_AsLong(PyList_GetItem(return_value_from_python_function,i));

            check_and_print_python_error();

            solution_instance->push_back(item_as_long);
            //std::cout << PyLong_AsLong(PyList_GetItem(return_value_from_python_function,i)) << std::endl;
        }

    }


    Py_DecRef(return_value_from_python_function);

}

void destroy_python_handler(Python_Handler** py_handler) {

    if(*py_handler != nullptr){
        Py_DECREF((*py_handler)->nuclei_detection_function);
        Py_DECREF((*py_handler)->organoid_detection_function);
        if((*py_handler)->qap_solver_function != nullptr){
            Py_DECREF((*py_handler)->qap_solver_function);
        }
    }


    if (Py_FinalizeEx() == -1) {
        std::cerr << "error in Py_FinalizeEx" << std::endl;
    }

	free((*py_handler));

	*py_handler = nullptr;
}