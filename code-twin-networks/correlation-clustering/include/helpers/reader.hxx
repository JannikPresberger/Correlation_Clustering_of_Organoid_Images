#pragma once
#ifndef HELPERS_READER_HXX
#define HELPERS_READER_HXX

#include "andres/graph/hdf5/hdf5.hxx"
#include "H5Cpp.h"

using namespace H5;

namespace helpers {
    template<class T>
    inline
    void readHDF5CorrelationClustering(
            const std::string & fileName,
            size_t &numberOfElements,
            std::vector<T> & data,
            std::vector<std::string> & labels,
            std::vector<std::string> & subsets,
            std::vector<std::string> & fileIds
    ) {
        std::vector<char*> labelCharData;
        std::vector<char*> subsetCharData;
        std::vector<char*> fileIdsCharData;

        H5File file(fileName, H5F_ACC_RDONLY);
        hid_t fileId = file.getId();

        std::vector<size_t> affinityShape;

        andres::graph::hdf5::load(fileId, "affinities", affinityShape, data);

        std::vector<size_t> labelShape;

        andres::graph::hdf5::load(fileId, "labels", labelShape, labelCharData);

        std::vector<size_t> subsetShape;
        andres::graph::hdf5::load(fileId, "subsets", subsetShape, subsetCharData);

        std::vector<size_t > fileIdsShape;
        andres::graph::hdf5::load(fileId, "file_ids", fileIdsShape, fileIdsCharData);


        for (char* cLabel : labelCharData){
            labels.emplace_back(cLabel);
        }
        for (char* cSubset: subsetCharData){
            subsets.emplace_back(cSubset);
        }
        for (char* cFileId: fileIdsCharData){
            fileIds.emplace_back(cFileId);
        }

        // free memory associated with string pointers
        for (char* cLabel : labelCharData){
            delete[] cLabel;
        }
        for (char* cSubset: subsetCharData){
            delete[] cSubset;
        }
        for (char* cFileId: fileIdsCharData){
            delete[] cFileId;
        }

        file.close();

        numberOfElements = labels.size();
    }

    inline
    void readHDF5ClassifierProtocol(
            const std::string & fileName,
            size_t &numberOfElements,
            std::vector<long> & truthData,
            std::vector<long> & predictedData,
            std::vector<std::string> & labels,
            std::vector<std::string> & fileIds
    ) {
        std::vector<char*> labelCharData;
        std::vector<char*> fileIdsCharData;

        H5File file(fileName, H5F_ACC_RDONLY);
        hid_t fileId = file.getId();

        std::vector<size_t> trueIdxsShape;
        andres::graph::hdf5::load(fileId, "trueIdxs", trueIdxsShape, truthData);

        std::vector<size_t > predictedIdxsShape;
        andres::graph::hdf5::load(fileId, "predictionIdxs", predictedIdxsShape, predictedData);

        std::vector<size_t> labelShape;
        andres::graph::hdf5::load(fileId, "labels", labelShape, labelCharData);

        std::vector<size_t > fileIdsShape;
        andres::graph::hdf5::load(fileId, "file_ids", fileIdsShape, fileIdsCharData);


        for (char* cLabel : labelCharData){
            labels.emplace_back(cLabel);
        }
        for (char* cFileId: fileIdsCharData){
            fileIds.emplace_back(cFileId);
        }

        // free memory associated with string pointers
        for (char* cLabel : labelCharData){
            delete[] cLabel;
        }
        for (char* cFileId: fileIdsCharData){
            delete[] cFileId;
        }

        file.close();

        numberOfElements = labels.size();
    }


    inline
    void readHDF5ClassifierEvaluation(
            const std::string & fileName,
            size_t &numberOfElements,
            std::vector<long> & truthData,
            std::vector<long> & predictedData
    ) {
        std::vector<char*> labelCharData;
        std::vector<char*> fileIdsCharData;

        H5File file(fileName, H5F_ACC_RDONLY);
        hid_t fileId = file.getId();

        std::vector<size_t> trueIdxsShape;
        andres::graph::hdf5::load(fileId, "labels", trueIdxsShape, truthData);

        std::vector<size_t > predictedIdxsShape;
        andres::graph::hdf5::load(fileId, "predictions", predictedIdxsShape, predictedData);


        file.close();

        numberOfElements = predictedData.size();
    }

}

#endif //HELPERS_READER_HXX
