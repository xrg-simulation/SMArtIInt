
#include "NeuralNet.h"
#include "../../SMArtIInt/Resources/Include/ModelicaUtilityHelper.h"
#include <iostream>
#include "tensorflow/lite/c/c_api.h"
#include "Utils.h"
#include <stdexcept>

NeuralNet::NeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* tfLiteModelPath, unsigned int dymInputDim,
	unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes,
	bool stateful, double fixInterval)
{

	// set member to access dymola functions
	mp_modelicaUtilityHelper = p_modelicaUtilityHelper;

	// handling of the input
	m_inputDim = dymInputDim;
	// clear the array if available
	if (mp_inputSizes) delete mp_inputSizes;
	// allocate array for input sizes
	mp_inputSizes = new unsigned int[m_inputDim];
	if (m_inputDim >= 1) {
		m_nInputEntries = 1;
	}
	else {
		mp_modelicaUtilityHelper->ModelicaError("Expecting at least one input dimension");
	}
	// store the values from function call
	for (unsigned int i = 0; i < m_inputDim; i++) {
		m_nInputEntries *= p_dymInputSizes[i];
		mp_inputSizes[i] = p_dymInputSizes[i];
	}

	// handling of the output
	m_outputDim = dymOutputDim;
	// clear the array if available
	if (mp_outputSizes) delete mp_outputSizes;
	// allocate array for output dimensions
	mp_outputSizes = new unsigned int[m_outputDim];
	// check input
	if (m_outputDim >= 1) {
		m_nOutputEntries = 1;
	}
	else {
		mp_modelicaUtilityHelper->ModelicaError("Expecting at least one output dimension");
	}
	// copy values from function call
	for (unsigned int i = 0; i < m_outputDim; i++) {
		m_nOutputEntries *= p_dymOutputSizes[i];
		mp_outputSizes[i] = p_dymOutputSizes[i];
	}

	// create the time step management object
	mp_timeStepMngmt = new InputManagement(stateful, fixInterval, m_nInputEntries);

	// perform steps to create model
	loadAndInit(tfLiteModelPath);
}

NeuralNet::~NeuralNet()
{
	// clean up own arrays
	if (mp_inputSizes) delete mp_inputSizes;
	mp_inputSizes = nullptr;
	m_inputDim = 0;
	m_nInputEntries = 0;

	if (mp_outputSizes) delete mp_outputSizes;
	mp_outputSizes = nullptr;
	m_outputDim = 0;
	m_nOutputEntries = 0;

	// clean up allocated tflite stuff
	if (mp_interpreter) TfLiteInterpreterDelete(mp_interpreter);
	if (mp_options) TfLiteInterpreterOptionsDelete(mp_options);
	if (mp_model) TfLiteModelDelete(mp_model);

	// clean up time step manager
	if (mp_timeStepMngmt) delete mp_timeStepMngmt;
}

void NeuralNet::loadAndInit(const char* tfliteModelPath)
{
	m_tfliteModelPath = tfliteModelPath;

	//mp_model = TfLiteModelCreateFromFile(m_tfliteModelPath);
	mp_model = TfLiteModelCreateFromFile(m_tfliteModelPath);
	// zero pointer is returned if not found
	if (!mp_model) {
		std::string message = Utils::string_format("SMArtInt: Model not found - check path: %s", m_tfliteModelPath);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	};
	mp_options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(mp_options, 1);

	// Create the interpreter.
	mp_interpreter = TfLiteInterpreterCreate(mp_model, mp_options);
	if (!mp_interpreter) {
		mp_modelicaUtilityHelper->ModelicaError("Failed to create interpreter");
	}

	// Allocate tensor buffers.
	if (TfLiteInterpreterAllocateTensors(mp_interpreter) != kTfLiteOk)
		mp_modelicaUtilityHelper->ModelicaError("Failed to allocate tensors!");
	// Find input tensors.
	if (TfLiteInterpreterGetInputTensorCount(mp_interpreter) != 1 && !mp_timeStepMngmt->isActive())
		mp_modelicaUtilityHelper->ModelicaError("SMArtInt can only handle models with single input");

	mp_flatInputTensor = TfLiteInterpreterGetInputTensor(mp_interpreter, 0);

	// set the casting function to correct types
	setInputCastFunction(mp_flatInputTensor);

	if (!mp_flatInputTensor) mp_modelicaUtilityHelper->ModelicaError("Failed to create m_input_tensor");

	// check dimensions - function throws error if not matching
	checkInputTensorSize();

	// adjust first dimension which is batch size
	if (TfLiteTensorDim(mp_flatInputTensor, 0) != int(mp_inputSizes[0])) {

		std::string message = "SMArtInt: Adjust first dimension from " + Utils::string_format("%i", TfLiteTensorDim(mp_flatInputTensor, 0)) + " to batch size " + Utils::string_format("%i\n", mp_inputSizes[0]);
		mp_modelicaUtilityHelper->ModelicaMessage(message.c_str());

		int* p_dymInputSizes;
		p_dymInputSizes = new int[m_inputDim];
		for (unsigned int i = 0; i < m_inputDim; ++i) {
			p_dymInputSizes[i] = int(mp_inputSizes[i]);
		}
		TfLiteInterpreterResizeInputTensor(mp_interpreter, 0, p_dymInputSizes, m_inputDim);

		// Reallocate tensor buffers for updated sizes
		if (TfLiteInterpreterAllocateTensors(mp_interpreter) != kTfLiteOk) {
			mp_modelicaUtilityHelper->ModelicaError("Failed to allocate tensors!");
		}
	}

	// check the number of outputs
	if (TfLiteInterpreterGetOutputTensorCount(mp_interpreter) != 1) {
		if (mp_timeStepMngmt->isActive()) {
			if (TfLiteInterpreterGetOutputTensorCount(mp_interpreter) != TfLiteInterpreterGetInputTensorCount(mp_interpreter)) {
				mp_modelicaUtilityHelper->ModelicaError("SMArtInt: Stateful handling can only be done if model has the same number of inputs (=) and outputs");
			}
		}
		else {
			mp_modelicaUtilityHelper->ModelicaError("SMArtInt can only handle models with single output!");
		}
	}

	// Handle states as additional inputs
	if (mp_timeStepMngmt->isActive()) {
		mp_modelicaUtilityHelper->ModelicaMessage("Handling additional inputs as states");
		//mp_timeStepMngmt->setNumberOfStates(TfLiteInterpreterGetInputTensorCount(mp_interpreter) - 1);
		for (int i = 1; i < TfLiteInterpreterGetInputTensorCount(mp_interpreter); ++i) {
			try {
				mp_timeStepMngmt->addStateInp(TfLiteInterpreterGetInputTensor(mp_interpreter, i));
			}
			catch (const std::invalid_argument& e) {
				mp_modelicaUtilityHelper->ModelicaError(e.what());
			}
		}
		// Initialize states if available
		mp_timeStepMngmt->initialize();
	}

	// dimensions etc of output tensor is only available after calling invoke so we check the infos after
	// the call - that it is only done once use the following variable
	m_firstInvoke = true;

	return;
}

void NeuralNet::runInferenceFlatTensor(double time, double* input, unsigned int inputLength, double* output, unsigned int outputLength)
{
	// check the sizes
	if (m_nInputEntries != inputLength) {
		std::string message = Utils::string_format("SMArtInt: Wrong input length: in the interface were %i entries defined, whereas in current function call %i is specified!", m_nInputEntries, inputLength);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	};
	// check output size
	if (m_nOutputEntries != outputLength) {
		std::string message = Utils::string_format("SMArtInt: Wrong output length: in the interface were %i entries defined, whereas in current function call %i is specified!", m_nOutputEntries, outputLength);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	};

    unsigned int nSteps = 0;
    try {
        nSteps = mp_timeStepMngmt->manageNewStep(time, m_firstInvoke, input);
    } catch (std::exception& e) {
        mp_modelicaUtilityHelper->ModelicaError(e.what());
    }

	void* p_data = TfLiteTensorData(mp_flatInputTensor);

	for (unsigned int i = 0; i < nSteps; ++i)
	{
		double* inpInput = mp_timeStepMngmt->handleInpts(time, i, input, m_firstInvoke);

		// we write the data directly into the data array of the tensor - the casting function is set to the correct
		// type
		for (unsigned int i = 0; i < m_nInputEntries; ++i) {
			mfp_castInput(inpInput[i], p_data, i);
		}

		// Run inference
		if (TfLiteInterpreterInvoke(mp_interpreter) != kTfLiteOk) {
			mp_modelicaUtilityHelper->ModelicaError("Inference failed");
		};
	}

	const TfLiteTensor* p_flatOutputTensor = TfLiteInterpreterGetOutputTensor(mp_interpreter, 0);

	if (m_firstInvoke) {
		// check sizes
		checkOutputTensorSize(p_flatOutputTensor);

		// set the casting function to correct types
		setOutputCastFunction(p_flatOutputTensor);

		// handle additional outputs for states
		for (int i = 1; i < TfLiteInterpreterGetOutputTensorCount(mp_interpreter); ++i) {
			try {
				mp_timeStepMngmt->addStateOut(TfLiteInterpreterGetOutputTensor(mp_interpreter, i));
			}
			catch (const std::invalid_argument& e) {
				mp_modelicaUtilityHelper->ModelicaError(e.what());
			}
		}
		// everything is fine - so it should not be done again
		m_firstInvoke = false;
	}

	// directly access the tensor data
	p_data = TfLiteTensorData(p_flatOutputTensor);
	for (unsigned int i = 0; i < m_nOutputEntries; ++i) {
		mfp_castOutput(output[i], p_data, i);
	}

	mp_timeStepMngmt->updateFinishedStep(time, nSteps);

	return;
}

void NeuralNet::initializeStates(double* p_stateValues, const unsigned int& nStateValues)
{
	try {
		mp_timeStepMngmt->initialize(p_stateValues, nStateValues);
	}
	catch (const std::invalid_argument& e) {
		mp_modelicaUtilityHelper->ModelicaError(e.what());
	}
}

void NeuralNet::checkInputTensorSize()
{
	if (TfLiteTensorNumDims(mp_flatInputTensor) != m_inputDim)
	{
		std::string message = Utils::string_format("SMArtInt: Wrong input dimensions : the loaded model has %i dimensions whereas in the interface %i is specified!", TfLiteTensorNumDims(mp_flatInputTensor), m_inputDim);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	}
	// check the sizes in each dimension except for the first which is the batch size
	for (unsigned int i = 1; i < m_inputDim; ++i) {
		if (TfLiteTensorDim(mp_flatInputTensor, i) != int(mp_inputSizes[i]))
		{
			std::string message = "SMArtInt: Wrong input sizes. The loaded model has the sizes {";
			for (unsigned int j = 0; j < m_inputDim; ++j) {
				message += Utils::string_format("%i", TfLiteTensorDim(mp_flatInputTensor, j));
				if (j < (m_inputDim - 1)) message += ", ";
			}
			message += "}, whereas in the interface the sizes {";
			for (unsigned int j = 0; j < m_inputDim; ++j) {
				message += Utils::string_format("%i", mp_inputSizes[j]);
				if (j < m_inputDim - 1) message += ", ";
			}
			message += "} were specified!";
			mp_modelicaUtilityHelper->ModelicaError(message.c_str());
		}
	}
}

void NeuralNet::checkOutputTensorSize(const TfLiteTensor* p_flatOutputTensor)
{
	// check dimensions
	if (TfLiteTensorNumDims(p_flatOutputTensor) != m_outputDim)
	{
		std::string message = Utils::string_format("SMArtInt: Wrong output dimensions : the loaded model has %i dimensions whereas in the interface %i is specified!", TfLiteTensorNumDims(p_flatOutputTensor), m_outputDim);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	}
	for (unsigned int i = 0; i < m_outputDim; ++i) {
		if (TfLiteTensorDim(p_flatOutputTensor, i) != int(mp_outputSizes[i]))
		{
			std::string message = "SMArtInt: Wrong output sizes. The loaded model has the sizes {";
			for (unsigned int j = 0; j < m_outputDim; ++j) {
				message += Utils::string_format("%i", TfLiteTensorDim(p_flatOutputTensor, j));
				if (j < (m_outputDim - 1)) message += ", ";
			}
			message += "}, whereas in the interface the sizes {";
			for (unsigned int j = 0; j < m_outputDim; ++j) {
				message += Utils::string_format("%i", mp_outputSizes[j]);
				if (j < m_outputDim - 1) message += ", ";
			}
			message += "} were specified!";
			mp_modelicaUtilityHelper->ModelicaError(message.c_str());
		}
	}
}

void NeuralNet::setInputCastFunction(TfLiteTensor* tensor)
{
	switch (TfLiteTensorType(tensor)) {
	case kTfLiteFloat32:
		mfp_castInput = &Utils::castToFloat;
		break;
	default:
		mp_modelicaUtilityHelper->ModelicaError("Could not convert input data - SMArtIInt currently only supports TFLite models using floats");
		break;
	}
}

void NeuralNet::setOutputCastFunction(const TfLiteTensor* tensor)
{
	switch (TfLiteTensorType(tensor))
	{

	case kTfLiteFloat32:
		mfp_castOutput = &Utils::castFromFloat;
		break;
	default:
		mp_modelicaUtilityHelper->ModelicaError("Could not convert output data - SMArtIInt currently only supports TFLite models using floats");
		break;
	}
}