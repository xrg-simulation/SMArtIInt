#pragma once
#include "../../SMArtIInt/Resources/Include/ModelicaUtilityHelper.h"


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

	__declspec(dllexport) void* NeuralNet_createObject(void* modelicaUtilityHelper, const char* tfLiteModelPath, unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes, bool stateful, double fixStep);

	__declspec(dllexport) void NeuralNet_destroyObject(void* externalObject);

	__declspec(dllexport) void NeuralNet_runInferenceFlatTensor(void* externalObject, double time, double* input, unsigned int inputLength,
		double* output, unsigned int outputLength);

	__declspec(dllexport) void NeuralNet_initializeStates(void* externalObject, double* states, unsigned int nStateValues);

#ifdef __cplusplus
}
#endif  // __cplusplus


