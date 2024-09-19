//
// Created by RobertFlesch on 16.09.2024.
//

#ifndef SMARTIINT_TENSORFLOWDLLHANDLERLINUX_H
#define SMARTIINT_TENSORFLOWDLLHANDLERLINUX_H

#include "TensorflowDllHandler.h"
#include <dlfcn.h>

class TensorflowDllHandlerLinux : public TensorflowDllHandler {
public:
    explicit TensorflowDllHandlerLinux(const char* filename);
    ~TensorflowDllHandlerLinux() override;

    TfLiteStatus interpreterAllocateTensors(TfLiteInterpreter *interpreter) override {
        return reinterpret_cast<TfLiteStatus>((*_f_interpreterAllocateTensors)(interpreter));
    }

    TfLiteModel *createModelFromFile(const char *model_path) override {
        return reinterpret_cast<TfLiteModel *>((*_f_createModelFromFile)(model_path));
    }

    void interpreterDelete(TfLiteInterpreter *interpreter) override {
        (*_f_interpreterDelete)(interpreter);
    }

    void interpreterOptionsDelete(TfLiteInterpreterOptions* options) override {
        (*_f_interpreterOptionsDelete)(options);
    }

    void modelDelete(TfLiteModel *model) override {
        (*_f_modelDelete)(model);
    }

    TfLiteInterpreter *interpreterCreate(const TfLiteModel *model, const TfLiteInterpreterOptions *optional_options) override {
        return reinterpret_cast<TfLiteInterpreter *>((*_f_interpreterCreate)(model, optional_options));
    }

    int32_t interpreterGetInputTensorCount(const TfLiteInterpreter *interpreter) override {
        return (*_f_interpreterGetInputTensorCount)(interpreter);
    }

    TfLiteTensor *interpreterGetInputTensor(const TfLiteInterpreter *interpreter, int32_t input_index) override {
        return reinterpret_cast<TfLiteTensor *>((*_f_interpreterGetInputTensor)(interpreter, input_index));
    }

    const TfLiteTensor *interpreterGetOutputTensor(const TfLiteInterpreter *interpreter, int32_t output_index) override {
        return reinterpret_cast<const TfLiteTensor *>((*_f_interpreterGetOutputTensor)(interpreter, output_index));
    }

    int32_t interpreterGetOutputTensorCount(const TfLiteInterpreter *interpreter) override {
        return (*_f_interpreterGetOutputTensorCount)(interpreter);
    }

    TfLiteStatus interpreterInvoke(TfLiteInterpreter *interpreter) override {
        return (*_f_interpreterInvoke)(interpreter);
    }

    TfLiteStatus interpreterResizeInputTensor(TfLiteInterpreter *interpreter, int32_t input_index, const int *dims, int32_t dims_size) override {
        return (*_f_interpreterResizeInputTensor)(interpreter, input_index, dims, dims_size);
    }

    TfLiteStatus interpreterModifyGraphWithDelegate(TfLiteInterpreter *interpreter, TfLiteDelegate *delegate) override {
        return (*_f_interpreterModifyGraphWithDelegate)(interpreter, delegate);
    }

    TfLiteInterpreterOptions *interpreterOptionsCreate() override {
        return reinterpret_cast<TfLiteInterpreterOptions *>((*_f_interpreterOptionsCreate)());
    }

    void interpreterOptionsSetNumThreads(TfLiteInterpreterOptions *options, int32_t num_threads) override {
        (*_f_interpreterOptionsSetNumThreads)(options, num_threads);
    }

    void interpreterOptionsAddDelegate(TfLiteInterpreterOptions *options, TfLiteDelegate *delegate) override {
        (*_f_interpreterOptionsAddDelegate)(options, delegate);
    }

    int32_t tensorDim(const TfLiteTensor* tensor, int32_t dim_index) override {
        return (*_f_tensorDim)(tensor, dim_index);
    }

    int32_t tensorNumDims(const TfLiteTensor* tensor) override {
        return (*_f_tensorNumDims)(tensor);
    }

    TfLiteType tensorType(const TfLiteTensor* tensor) override {
        return (*_f_tensorType)(tensor);
    }

    void* tensorData(const TfLiteTensor* tensor) override {
        return (*_f_tensorData)(tensor);
    }

    size_t tensorByteSize(const TfLiteTensor* tensor) override {
        return (*_f_tensorByteSize)(tensor);
    }

private:
    void* _module;

    PFN_CREATEMODELFROMFILE _f_createModelFromFile;
    PFN_INTERPRETERDELETE _f_interpreterDelete;
    PFN_INTERPRETEROPTIONSDELETE _f_interpreterOptionsDelete;
    PFN_MODELDELETE _f_modelDelete;
    PFN_INTERPRETERCREATE _f_interpreterCreate;
    PFN_INTERPRETERALLOCATETENSORS _f_interpreterAllocateTensors;
    PFN_INTERPRETERGETINPUTTENSORCOUNT _f_interpreterGetInputTensorCount;
    PFN_INTERPRETERGETINPUTTENSOR _f_interpreterGetInputTensor;
    PFN_INTERPRETERGETOUTPUTTENSOR _f_interpreterGetOutputTensor;
    PFN_INTERPRETERGETOUTPUTTENSORCOUNT _f_interpreterGetOutputTensorCount;
    PFN_INTERPRETERINVOKE _f_interpreterInvoke;
    PFN_INTERPRETERRESIZEINPUTTENSOR _f_interpreterResizeInputTensor;
    PFN_INTERPRETERMODIFYGRAPHWITHDELEGATE _f_interpreterModifyGraphWithDelegate;
    PFN_INTERPRETEROPTIONSCREATE _f_interpreterOptionsCreate;
    PFN_INTERPRETEROPTIONSSETNUMTHREADS _f_interpreterOptionsSetNumThreads;
    PFN_INTERPRETEROPTIONSADDDELEGATE _f_interpreterOptionsAddDelegate;
    PFN_TENSORDIM _f_tensorDim;
    PFN_TENSORNUMDIMS _f_tensorNumDims;
    PFN_TENSORTYPE _f_tensorType;
    PFN_TENSORDATA _f_tensorData;
    PFN_TENSORBYTESIZE _f_tensorByteSize;

    static void handleError();
};

#endif //SMARTIINT_TENSORFLOWDLLHANDLERLINUX_H