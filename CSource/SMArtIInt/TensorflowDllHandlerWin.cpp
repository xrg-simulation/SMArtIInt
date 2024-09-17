//
// Created by RobertFlesch on 16.09.2024.
//

#include "TensorflowDllHandlerWin.h"

TensorflowDllHandlerWin::TensorflowDllHandlerWin(LPCTSTR filename) {
    _module = LoadLibrary(filename);
    _f_createModelFromFile = (ProcPtr) GetProcAddress(_module, "createModelFromFile");
    _f_createModelFromFile = (ProcPtr) GetProcAddress(_module, "TfLiteModelCreate");
    _f_interpreterDelete = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterDelete");
    _f_interpreterOptionsDelete = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterOptionsDelete");
    _f_modelDelete = (ProcPtr) GetProcAddress(_module, "TfLiteModelDelete");
    _f_interpreterCreate = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterCreate");
    _f_interpreterAllocateTensors = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterAllocateTensors");
    _f_interpreterGetInputTensorCount = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterGetInputTensorCount");
    _f_interpreterGetInputTensor = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterGetInputTensor");
    _f_interpreterGetOutputTensor = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterGetOutputTensor");
    _f_interpreterGetOutputTensorCount = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterGetOutputTensorCount");
    _f_interpreterInvoke = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterInvoke");
    _f_interpreterResizeInputTensor = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterResizeInputTensor");
    _f_interpreterModifyGraphWithDelegate = (ProcPtr) GetProcAddress(_module,
                                                                     "TfLiteInterpreterModifyGraphWithDelegate");
    _f_interpreterOptionsCreate = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterOptionsCreate");
    _f_interpreterOptionsSetNumThreads = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterOptionsSetNumThreads");
    _f_interpreterOptionsAddDelegate = (ProcPtr) GetProcAddress(_module, "TfLiteInterpreterOptionsAddDelegate");

}
