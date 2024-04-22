#include "InterfaceFunctions.h"
#include "NeuralNet.h"
#include "../../SMArtIInt/Resources/Include/ModelicaUtilityHelper.h"

void* NeuralNet_createObject(void* modelicaUtilityHelper, const char* tfLiteModelPath, unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes, bool stateful, double fixStep)
{
	ModelicaUtilityHelper* p_modelicaUtilityHelper = (ModelicaUtilityHelper*)modelicaUtilityHelper;
	#ifndef NDEBUG
	p_modelicaUtilityHelper->ModelicaMessage("!! SMArtIInt running in Debug Mode !!\n\n");
    #endif
    //ToDo: check fixStep (must it always be >0???)
	
	NeuralNet* p_neuralNet = new NeuralNet (p_modelicaUtilityHelper, tfLiteModelPath, dymInputDim, p_dymInputSizes, dymOutputDim, p_dymOutputSizes, stateful, fixStep);
	
	return (void*)p_neuralNet;
}

void NeuralNet_destroyObject(void* externalObject)
{
	// cast to actual pointer
	NeuralNet* p_neuralNet = (NeuralNet*)externalObject;

	if(p_neuralNet)
	{
		delete p_neuralNet;
	}
}

void NeuralNet_runInferenceFlatTensor(void* externalObject, double time, double* input, unsigned int inputLength,
	double* output, unsigned int outputLength)
{
	// cast to actual pointer
	NeuralNet* p_neuralNet = (NeuralNet*)externalObject;
	p_neuralNet->runInferenceFlatTensor(time, input, inputLength, output, outputLength);
	return;
}

void NeuralNet_initializeStates(void* externalObject, double* states, unsigned int nStateValues)
{
	// cast to actual pointer
	NeuralNet* p_neuralNet = (NeuralNet*)externalObject;
	p_neuralNet->initializeStates(states, nStateValues);
	return;
}