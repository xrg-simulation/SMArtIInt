#include "InterfaceFunctions.h"
#include "NeuralNet.h"

void* NeuralNet_createObject(void* modelicaUtilityHelper, const char* tfLiteModelPath, unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes, bool stateful, double fixStep)
{
	auto* p_modelicaUtilityHelper = (ModelicaUtilityHelper*)modelicaUtilityHelper;
	#ifndef NDEBUG
	p_modelicaUtilityHelper->ModelicaMessage("!! SMArtIInt running in Debug Mode !!\n\n");
    #endif

	auto*  p_neuralNet = new NeuralNet (p_modelicaUtilityHelper, tfLiteModelPath, dymInputDim, p_dymInputSizes, dymOutputDim, p_dymOutputSizes, stateful, fixStep);
	
	return  (void*)p_neuralNet;
}

void NeuralNet_destroyObject(void* externalObject)
{
	// cast to actual pointer
	auto* p_neuralNet = (NeuralNet*)externalObject;
    delete p_neuralNet;

}

void NeuralNet_runInferenceFlatTensor(void* externalObject, double time, double* input, unsigned int inputLength,
	double* output, unsigned int outputLength)
{
	// cast to actual pointer
	auto* p_neuralNet = (NeuralNet*)externalObject;
	p_neuralNet->runInferenceFlatTensor(time, input, inputLength, output, outputLength);
}

void NeuralNet_initializeStates(void* externalObject, double time,  double* states, unsigned int nStateValues)
{
	// cast to actual pointer
	auto* p_neuralNet = (NeuralNet*)externalObject;
	p_neuralNet->initializeStates(time, states, nStateValues);
}