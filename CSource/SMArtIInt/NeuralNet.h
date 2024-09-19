#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include "tensorflow/lite/c/common.h"
#include "../../SMArtIInt/Resources/Include/ModelicaUtilityHelper.h"
#include "tensorflow/lite/c/c_api.h"
#include "InputManagement.h"
#include <cstring>
#include "TensorflowDllHandler.h"

class NeuralNet
{
public:
	NeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* tfLiteModelPath,
		int32_t dymInputDim, const unsigned int* p_dymInputSizes, unsigned int dymOutputDim, const unsigned int* p_dymOutputSizes,
		bool stateful, double fixInterval);

	~NeuralNet();

	void runInferenceFlatTensor(double time, double* input, unsigned int inputLength, double* output, unsigned int outputLength); // invoke the model

	void initializeStates(double* p_stateValues, const unsigned int& nStateValues); // function to intialize states with given values

private:
	
	void loadAndInit(const char* tfliteModelPath); // internal function to prepare model - called by constructor

	void setInputCastFunction(TfLiteTensor* mp_flatInputTensor); // function to set the casting function for inputs
	void setOutputCastFunction(const TfLiteTensor* tensor); // function to set the casting function for outputs

	// util function to check sizes
	void checkInputTensorSize(); // check if the tensor sizes defined in modelica are equal to those in the model
	void checkOutputTensorSize(const TfLiteTensor* p_flatOutputTensor); // check if the tensor sizes defined in modelica are equal to those in the model

private:

    TensorflowDllHandler* mp_tfdll;

	ModelicaUtilityHelper* mp_modelicaUtilityHelper; // attribute to access dymola utility functions

	InputManagement* mp_timeStepMngmt; // time step manager used for stateful RNNs

	const char* m_tfliteModelPath = ""; // path of the model

	TfLiteModel* mp_model = nullptr; // pointer to model
	TfLiteInterpreterOptions* mp_options = nullptr; // pointer to model options
	TfLiteInterpreter* mp_interpreter = nullptr; // pointer to interpreter

	void (*mfp_castInput)(const double&, void*, unsigned int); // pointer to input casting function
	void (*mfp_castOutput)(double&, void*, unsigned int); // pointer to output casting function

	// in and output
	int32_t m_inputDim = 0; // dimension of input as specified in modelica
	unsigned int* mp_inputSizes = nullptr; // sizes of input as specified in modelica
	unsigned int m_nInputEntries; // total number of input entries
	TfLiteTensor* mp_flatInputTensor = nullptr; // pointer to flat input tensors from nn

	unsigned int m_outputDim = 0; // dimensions of output
	unsigned int* mp_outputSizes = nullptr; // sizes of output as specified in modelica
	unsigned int m_nOutputEntries; // total number of output elements

	bool m_firstInvoke = true; // flag if outputs needs to be allocated etc
};


/*
static std::wstring charToWString(const char* text)
{
	const size_t size = std::strlen(text);
	std::wstring wstr;
	if (size > 0) {
		wstr.resize(size+1);
		//std::mbstowcs(&wstr[0], text, size);
		size_t outSize;
		mbstowcs_s(&outSize, &wstr[0], size+1, text, size);
	}
	return wstr.substr(0, wstr.size() - 1);;
}
*/