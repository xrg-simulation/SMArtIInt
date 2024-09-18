//
// Created by RobertFlesch on 16.09.2024.
//

#include "TensorflowDllHandlerWin.h"


TensorflowDllHandlerWin::TensorflowDllHandlerWin(LPCTSTR filename) {
    _module = LoadLibrary(filename);
    _f_createModelFromFile = (PFN_CREATEMODELFROMFILE) GetProcAddress(_module,
                                                                      "TfLiteModelCreateFromFile");

    _f_interpreterDelete = (PFN_INTERPRETERDELETE) GetProcAddress(_module,
                                                                  "TfLiteInterpreterDelete");

    _f_interpreterOptionsDelete = (PFN_INTERPRETEROPTIONSDELETE)
            GetProcAddress(_module,"TfLiteInterpreterOptionsDelete");

    _f_modelDelete = (PFN_MODELDELETE) GetProcAddress(_module, "TfLiteModelDelete");

    _f_interpreterCreate = (PFN_INTERPRETERCREATE) GetProcAddress(_module,
                                                                  "TfLiteInterpreterCreate");

    _f_interpreterAllocateTensors = (PFN_INTERPRETERALLOCATETENSORS)
            GetProcAddress(_module,"TfLiteInterpreterAllocateTensors");

    _f_interpreterGetInputTensorCount = (PFN_INTERPRETERGETINPUTTENSORCOUNT)
            GetProcAddress(_module, "TfLiteInterpreterGetInputTensorCount");

    _f_interpreterGetInputTensor = (PFN_INTERPRETERGETINPUTTENSOR)
            GetProcAddress(_module, "TfLiteInterpreterGetInputTensor");

    _f_interpreterGetOutputTensor = (PFN_INTERPRETERGETOUTPUTTENSOR)
            GetProcAddress(_module, "TfLiteInterpreterGetOutputTensor");

    _f_interpreterGetOutputTensorCount = (PFN_INTERPRETERGETOUTPUTTENSORCOUNT )
            GetProcAddress(_module, "TfLiteInterpreterGetOutputTensorCount");

    _f_interpreterInvoke = (PFN_INTERPRETERINVOKE)
            GetProcAddress(_module, "TfLiteInterpreterInvoke");

    _f_interpreterResizeInputTensor = (PFN_INTERPRETERRESIZEINPUTTENSOR )
            GetProcAddress(_module, "TfLiteInterpreterResizeInputTensor");

    _f_interpreterModifyGraphWithDelegate = (PFN_INTERPRETERMODIFYGRAPHWITHDELEGATE)
            GetProcAddress(_module,"TfLiteInterpreterModifyGraphWithDelegate");

    _f_interpreterOptionsCreate = (PFN_INTERPRETEROPTIONSCREATE)
            GetProcAddress(_module, "TfLiteInterpreterOptionsCreate");

    _f_interpreterOptionsSetNumThreads = (PFN_INTERPRETEROPTIONSSETNUMTHREADS)
            GetProcAddress(_module, "TfLiteInterpreterOptionsSetNumThreads");

    _f_interpreterOptionsAddDelegate = (PFN_INTERPRETEROPTIONSADDDELEGATE)
            GetProcAddress(_module, "TfLiteInterpreterOptionsAddDelegate");

    _f_tensorDim = (PFN_TENSORDIM) GetProcAddress(_module, "TfLiteTensorDim");

    _f_tensorNumDims = (PFN_TENSORNUMDIMS) GetProcAddress(_module, "TfLiteTensorNumDims");

    _f_tensorType = (PFN_TENSORTYPE ) GetProcAddress(_module, "TfLiteTensorType");

    _f_tensorData = (PFN_TENSORDATA ) GetProcAddress(_module, "TfLiteTensorData");

    _f_tensorByteSize = (PFN_TENSORBYTESIZE) GetProcAddress(_module, "TfLiteTensorByteSize");
}
