#pragma once

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "image_processing.h"

typedef struct Python_Handler {
	PyObject* nuclei_detection_function;
	PyObject* organoid_detection_function;
	PyObject* qap_solver_function;
}Python_Handler;

extern Python_Handler* global_python_handler;

Python_Handler* init_python_handler(std::filesystem::path python_scripts_folder_path);

void python_extract_feature_points(Python_Handler* py_handler, cv::Mat* organoid_image, std::vector<Feature_Point_Data>& feature_points, double nms_thresh, Channel_Type channel);

int python_extract_organoid_labels(Python_Handler* py_handler, cv::Mat* organoid_image, cv::Mat* output_label_image, double nms_thresh);

void python_solve_qap(Python_Handler* py_handler,std::vector<int>* solution_instance, int num_keypoints, int num_candiates, int num_unary_expr, int num_quadr_expr, double* unary_costs, double* quadr_costs, int* edge_indices);

void destroy_python_handler(Python_Handler** py_handler);