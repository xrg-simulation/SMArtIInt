//
// Created by RobertFlesch on 16.09.2024.
//

#include "TensorflowDllHandlerLinux.h"
#include <dlfcn.h>
#include <stdexcept>

TensorflowDllHandlerLinux::TensorflowDllHandlerLinux(const char* filename) {
    _module = dlopen(filename, RTLD_LAZY);
    if (!_module) {
        throw std::runtime_error(dlerror());
    }

    _f_createModelFromFile = (PFN_CREATEMODELFROMFILE) dlsym(_module, "TfLiteModelCreateFromFile");
    if (!_f_createModelFromFile) handleError();

    _f_interpreterDelete = (PFN_INTERPRETERDELETE) dlsym(_module, "TfLiteInterpreterDelete");
    if (!_f_interpreterDelete) handleError();

    _f_interpreterOptionsDelete = (PFN_INTERPRETEROPTIONSDELETE) dlsym(_module, "TfLiteInterpreterOptionsDelete");
    if (!_f_interpreterOptionsDelete) handleError();

    _f_modelDelete = (PFN_MODELDELETE) dlsym(_module, "TfLiteModelDelete");
    if (!_f_modelDelete) handleError();

    _f_interpreterCreate = (PFN_INTERPRETERCREATE) dlsym(_module, "TfLiteInterpreterCreate");
    if (!_f_interpreterCreate) handleError();

    _f_interpreterAllocateTensors = (PFN_INTERPRETERALLOCATETENSORS) dlsym(_module, "TfLiteInterpreterAllocateTensors");
    if (!_f_interpreterAllocateTensors) handleError();

    _f_interpreterGetInputTensorCount = (PFN_INTERPRETERGETINPUTTENSORCOUNT) dlsym(_module, "TfLiteInterpreterGetInputTensorCount");
    if (!_f_interpreterGetInputTensorCount) handleError();

    _f_interpreterGetInputTensor = (PFN_INTERPRETERGETINPUTTENSOR) dlsym(_module, "TfLiteInterpreterGetInputTensor");
    if (!_f_interpreterGetInputTensor) handleError();

    _f_interpreterGetOutputTensor = (PFN_INTERPRETERGETOUTPUTTENSOR) dlsym(_module, "TfLiteInterpreterGetOutputTensor");
    if (!_f_interpreterGetOutputTensor) handleError();

    _f_interpreterGetOutputTensorCount = (PFN_INTERPRETERGETOUTPUTTENSORCOUNT) dlsym(_module, "TfLiteInterpreterGetOutputTensorCount");
    if (!_f_interpreterGetOutputTensorCount) handleError();

    _f_interpreterInvoke = (PFN_INTERPRETERINVOKE) dlsym(_module, "TfLiteInterpreterInvoke");
    if (!_f_interpreterInvoke) handleError();

    _f_interpreterResizeInputTensor = (PFN_INTERPRETERRESIZEINPUTTENSOR) dlsym(_module, "TfLiteInterpreterResizeInputTensor");
    if (!_f_interpreterResizeInputTensor) handleError();

    _f_interpreterModifyGraphWithDelegate = (PFN_INTERPRETERMODIFYGRAPHWITHDELEGATE) dlsym(_module,"TfLiteInterpreterModifyGraphWithDelegate");
    if (!_f_interpreterModifyGraphWithDelegate) handleError();

    _f_interpreterOptionsCreate = (PFN_INTERPRETEROPTIONSCREATE)dlsym(_module, "TfLiteInterpreterOptionsCreate");
    if (!_f_interpreterOptionsCreate) handleError();

    _f_interpreterOptionsSetNumThreads = (PFN_INTERPRETEROPTIONSSETNUMTHREADS) dlsym(_module, "TfLiteInterpreterOptionsSetNumThreads");
    if (!_f_interpreterOptionsSetNumThreads) handleError();

    _f_interpreterOptionsAddDelegate = (PFN_INTERPRETEROPTIONSADDDELEGATE) dlsym(_module, "TfLiteInterpreterOptionsAddDelegate");
    if (!_f_interpreterOptionsAddDelegate) handleError();

    _f_tensorDim = (PFN_TENSORDIM) dlsym(_module, "TfLiteTensorDim");
    if (!_f_tensorDim) handleError();

    _f_tensorNumDims = (PFN_TENSORNUMDIMS) dlsym(_module, "TfLiteTensorNumDims");
    if (!_f_tensorNumDims) handleError();

    _f_tensorType = (PFN_TENSORTYPE) dlsym(_module, "TfLiteTensorType");
    if (!_f_tensorType) handleError();

    _f_tensorData = (PFN_TENSORDATA) dlsym(_module, "TfLiteTensorData");
    if (!_f_tensorData) handleError();

    _f_tensorByteSize = (PFN_TENSORBYTESIZE) dlsym(_module, "TfLiteTensorByteSize");
    if (!_f_tensorByteSize) handleError();
}

TensorflowDllHandlerLinux::~TensorflowDllHandlerLinux() {
    if (_module) {
        dlclose(_module);
    }
}

void TensorflowDllHandlerLinux::handleError() {
    throw std::runtime_error(dlerror());
}