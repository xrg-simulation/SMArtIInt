//
// Created by RobertFlesch on 16.09.2024.
//

#include "tensorflow/lite/c/c_api.h"

#ifndef SMARTIINT_TENSORFLOWDLLHANDLER_H
#define SMARTIINT_TENSORFLOWDLLHANDLER_H


typedef TfLiteModel* (__cdecl *PFN_CREATEMODELFROMFILE) (const char*);

typedef void (__cdecl *PFN_INTERPRETERDELETE)(TfLiteInterpreter *);

typedef void (__cdecl *PFN_INTERPRETEROPTIONSDELETE)(TfLiteInterpreterOptions *);

typedef void (__cdecl *PFN_MODELDELETE)(TfLiteModel *);

typedef TfLiteInterpreter *(__cdecl *PFN_INTERPRETERCREATE)(const TfLiteModel *, const TfLiteInterpreterOptions *);

typedef TfLiteStatus (__cdecl *PFN_INTERPRETERALLOCATETENSORS)(TfLiteInterpreter *);

typedef int32_t (__cdecl *PFN_INTERPRETERGETINPUTTENSORCOUNT)(const TfLiteInterpreter *);

typedef TfLiteTensor *(__cdecl *PFN_INTERPRETERGETINPUTTENSOR)(const TfLiteInterpreter *, int32_t input_index);

typedef const TfLiteTensor *(__cdecl *PFN_INTERPRETERGETOUTPUTTENSOR)(const TfLiteInterpreter *, int32_t output_index);

typedef int32_t (__cdecl *PFN_INTERPRETERGETOUTPUTTENSORCOUNT)(const TfLiteInterpreter *);

typedef TfLiteStatus (__cdecl *PFN_INTERPRETERINVOKE)(TfLiteInterpreter *);

typedef TfLiteStatus (__cdecl *PFN_INTERPRETERRESIZEINPUTTENSOR)(TfLiteInterpreter *, int32_t input_index, const int *,
                                                             int32_t dims_size);

typedef TfLiteStatus (__cdecl *PFN_INTERPRETERMODIFYGRAPHWITHDELEGATE)(TfLiteInterpreter *, TfLiteDelegate *);

typedef TfLiteInterpreterOptions *(__cdecl *PFN_INTERPRETEROPTIONSCREATE)();

typedef void (__cdecl *PFN_INTERPRETEROPTIONSSETNUMTHREADS)(TfLiteInterpreterOptions *, int32_t num_threads);

typedef void (__cdecl *PFN_INTERPRETEROPTIONSADDDELEGATE)(TfLiteInterpreterOptions *, TfLiteDelegate *);

typedef int32_t (__cdecl *PFN_TENSORDIM)(const TfLiteTensor*, int32_t);

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

    virtual int32_t tensorDim(const TfLiteTensor* tensor, int32_t dim_index) = 0;

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
