//
// Created by RobertFlesch on 16.09.2024.
//

#include "tensorflow/lite/c/c_api.h"

#ifndef SMARTIINT_TENSORFLOWDLLHANDLER_H
#define SMARTIINT_TENSORFLOWDLLHANDLER_H


class TensorflowDllHandler {

protected:
    virtual ~TensorflowDllHandler() = default;

public:
    virtual TfLiteModel* createModelFromFile(const char* model_path) = 0;
    virtual void interpreterDelete(TfLiteInterpreter* interpreter) = 0;
    virtual void interpreterOptionsDelete(TfLiteInterpreterOptions* options) = 0;
    virtual void modelDelete(TfLiteModel* model) = 0;
    virtual TfLiteInterpreter* interpreterCreate(const TfLiteModel* model, const TfLiteInterpreterOptions* optional_options) = 0;
    virtual TfLiteStatus interpreterAllocateTensors(TfLiteInterpreter* interpreter) = 0;
    virtual int32_t interpreterGetInputTensorCount(const TfLiteInterpreter* interpreter) = 0;
    virtual TfLiteTensor *
    interpreterGetInputTensor(const TfLiteInterpreter *interpreter, int32_t input_index) = 0;

    virtual const TfLiteTensor *
    interpreterGetOutputTensor(const TfLiteInterpreter *interpreter, int32_t output_index) = 0;

    virtual int32_t interpreterGetOutputTensorCount(const TfLiteInterpreter *interpreter) = 0;

    virtual TfLiteStatus interpreterInvoke(TfLiteInterpreter *interpreter) = 0;

    virtual TfLiteStatus
    interpreterResizeInputTensor(TfLiteInterpreter *interpreter, int32_t input_index, const int *dims,
                                 int32_t dims_size) = 0;

    virtual TfLiteStatus
    interpreterModifyGraphWithDelegate(TfLiteInterpreter *interpreter, TfLiteDelegate *delegate) = 0;

    virtual TfLiteInterpreterOptions *interpreterOptionsCreate() = 0;

    virtual void interpreterOptionsSetNumThreads(TfLiteInterpreterOptions *options, int32_t num_threads) = 0;

    virtual void interpreterOptionsAddDelegate(TfLiteInterpreterOptions *options, TfLiteDelegate *delegate) = 0;
};


#endif //SMARTIINT_TENSORFLOWDLLHANDLER_H
