#pragma once
#include "../../SMArtIInt/Resources/Include/ModelicaUtilityHelper.h"

#if defined(_MSC_VER)
//  Microsoft 
#define EXPORT __declspec(dllexport)
#define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
//  GCC
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#else
//  do nothing and hope for the best?
#define EXPORT
#define IMPORT
#pragma warning Unknown dynamic link import/export semantics.
#endif


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

	EXPORT void* NeuralNet_createObject(void* modelicaUtilityHelper, const char* tfLiteModelPath, unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes, bool stateful, double fixStep);

	EXPORT void NeuralNet_destroyObject(void* externalObject);

	EXPORT void NeuralNet_runInferenceFlatTensor(void* externalObject, double time, double* input, unsigned int inputLength,
		double* output, unsigned int outputLength);

	EXPORT void NeuralNet_initializeStates(void* externalObject, double* states, unsigned int nStateValues);

#ifdef __cplusplus
}
#endif  // __cplusplus


