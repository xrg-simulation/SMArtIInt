//
// Created by RobertFlesch on 16.09.2024.
//

#include "tensorflow/lite/c/c_api.h"

#ifndef SMARTIINT_TENSORFLOWDLLHANDLER_H
#define SMARTIINT_TENSORFLOWDLLHANDLER_H

#if defined(_MSC_VER)
#define TF_CDECL __cdecl
#else
#define TF_CDECL
#endif

typedef TfLiteModel* (TF_CDECL *PFN_CREATEMODELFROMFILE) (const char*);

typedef void (TF_CDECL *PFN_INTERPRETERDELETE)(TfLiteInterpreter *);

typedef void (TF_CDECL *PFN_INTERPRETEROPTIONSDELETE)(TfLiteInterpreterOptions *);

typedef void (TF_CDECL *PFN_MODELDELETE)(TfLiteModel *);

typedef TfLiteInterpreter *(TF_CDECL *PFN_INTERPRETERCREATE)(const TfLiteModel *, const TfLiteInterpreterOptions *);

typedef TfLiteStatus (TF_CDECL *PFN_INTERPRETERALLOCATETENSORS)(TfLiteInterpreter *);

typedef int32_t (TF_CDECL *PFN_INTERPRETERGETINPUTTENSORCOUNT)(const TfLiteInterpreter *);

typedef TfLiteTensor *(TF_CDECL *PFN_INTERPRETERGETINPUTTENSOR)(const TfLiteInterpreter *, int32_t input_index);

typedef const TfLiteTensor *(TF_CDECL *PFN_INTERPRETERGETOUTPUTTENSOR)(const TfLiteInterpreter *, int32_t output_index);

typedef int32_t (TF_CDECL *PFN_INTERPRETERGETOUTPUTTENSORCOUNT)(const TfLiteInterpreter *);

typedef TfLiteStatus (TF_CDECL *PFN_INTERPRETERINVOKE)(TfLiteInterpreter *);

typedef TfLiteStatus (TF_CDECL *PFN_INTERPRETERRESIZEINPUTTENSOR)(TfLiteInterpreter *, int32_t input_index, const int *,
                                                             int32_t dims_size);

typedef TfLiteStatus (TF_CDECL *PFN_INTERPRETERMODIFYGRAPHWITHDELEGATE)(TfLiteInterpreter *, TfLiteDelegate *);

typedef TfLiteInterpreterOptions *(TF_CDECL *PFN_INTERPRETEROPTIONSCREATE)();

typedef void (TF_CDECL *PFN_INTERPRETEROPTIONSSETNUMTHREADS)(TfLiteInterpreterOptions *, int32_t num_threads);

typedef void (TF_CDECL *PFN_INTERPRETEROPTIONSADDDELEGATE)(TfLiteInterpreterOptions *, TfLiteDelegate *);

typedef int32_t (TF_CDECL *PFN_TENSORDIM)(const TfLiteTensor*, int32_t);

typedef int32_t (TF_CDECL *PFN_TENSORNUMDIMS)(const TfLiteTensor* tensor);

typedef TfLiteType (TF_CDECL *PFN_TENSORTYPE)(const TfLiteTensor* tensor);

typedef void* (TF_CDECL *PFN_TENSORDATA)(const TfLiteTensor*);

typedef size_t (TF_CDECL *PFN_TENSORBYTESIZE)(const TfLiteTensor* tensor);

class TensorflowDllHandler {

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

    virtual int32_t tensorNumDims(const TfLiteTensor* tensor) = 0;

    virtual TfLiteType tensorType(const TfLiteTensor* tensor) = 0;

    virtual void* tensorData(const TfLiteTensor* tensor) = 0;

    virtual size_t tensorByteSize(const TfLiteTensor* tensor) = 0;

    virtual ~TensorflowDllHandler() = default;
};


#endif //SMARTIINT_TENSORFLOWDLLHANDLER_H
