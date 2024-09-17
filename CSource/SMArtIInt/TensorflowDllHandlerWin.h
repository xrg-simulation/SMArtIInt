//
// Created by RobertFlesch on 16.09.2024.
//

#ifndef SMARTIINT_TENSORFLOWDLLHANDLERWIN_H
#define SMARTIINT_TENSORFLOWDLLHANDLERWIN_H


#include <minwindef.h>
#include <libloaderapi.h>
#include "TensorflowDllHandler.h"

class TensorflowDllHandlerWin : public TensorflowDllHandler {
public:
    explicit TensorflowDllHandlerWin(LPCTSTR filename);
    ~TensorflowDllHandlerWin() override { FreeLibrary(_module); }

    TfLiteStatus interpreterAllocateTensors(TfLiteInterpreter *interpreter) override {
        return reinterpret_cast<TfLiteStatus>((*_f_interpreterAllocateTensors)(interpreter));
    }

    TfLiteModel *createModelFromFile(const char *model_path) override {
        return reinterpret_cast<TfLiteModel *>((*_f_createModelFromFile)(model_path));
    }

    void interpreterDelete(TfLiteInterpreter *interpreter) override {
        (*_f_interpreterDelete)(interpreter);
    }

    void interpreterOptionsDelete(TfLiteInterpreter *interpreter) override {
        (*_f_interpreterOptionsDelete)(interpreter);
    }

    void modelDelete(TfLiteModel *model) override {
        (*_f_modelDelete)(model);
    }

    TfLiteInterpreter *
    interpreterCreate(const TfLiteModel *model, const TfLiteInterpreterOptions *optional_options) override {
        return reinterpret_cast<TfLiteInterpreter *>((*_f_interpreterCreate)(model, optional_options));
    }

    int32_t interpreterGetInputTensorCount(const TfLiteInterpreter *interpreter) override {
        return (*_f_interpreterGetInputTensorCount)(interpreter);
    }

    const TfLiteTensor *interpreterGetInputTensor(const TfLiteInterpreter *interpreter, int32_t input_index) override {
        return reinterpret_cast<const TfLiteTensor *>((*_f_interpreterGetInputTensor)(interpreter, input_index));
    }

    const TfLiteTensor *
    interpreterGetOutputTensor(const TfLiteInterpreter *interpreter, int32_t output_index) override {
        return reinterpret_cast<const TfLiteTensor *>((*_f_interpreterGetOutputTensor)(interpreter, output_index));
    }

    int32_t interpreterGetOutputTensorCount(const TfLiteInterpreter *interpreter) override {
        return (*_f_interpreterGetOutputTensorCount)(interpreter);
    }

    TfLiteStatus interpreterInvoke(TfLiteInterpreter *interpreter) override {
        return (*_f_interpreterInvoke)(interpreter);
    }

    TfLiteStatus interpreterResizeInputTensor(TfLiteInterpreter *interpreter, int32_t input_index, const int *dims,
                                              int32_t dims_size) override {
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


private:
    HMODULE _module;

    ProcPtr _f_createModelFromFile;
    ProcPtr _f_interpreterDelete;
    ProcPtr _f_interpreterOptionsDelete;
    ProcPtr _f_modelDelete;
    ProcPtr _f_interpreterCreate;
    ProcPtr _f_interpreterAllocateTensors;
    ProcPtr _f_interpreterGetInputTensorCount;
    ProcPtr _f_interpreterGetInputTensor;
    ProcPtr _f_interpreterGetOutputTensor;
    ProcPtr _f_interpreterGetOutputTensorCount;
    ProcPtr _f_interpreterInvoke;
    ProcPtr _f_interpreterResizeInputTensor;
    ProcPtr _f_interpreterModifyGraphWithDelegate;
    ProcPtr _f_interpreterOptionsCreate;
    ProcPtr _f_interpreterOptionsSetNumThreads;
    ProcPtr _f_interpreterOptionsAddDelegate;


    
};


#endif //SMARTIINT_TENSORFLOWDLLHANDLERWIN_H
