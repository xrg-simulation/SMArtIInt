
#include "NeuralNet.h"
#include "Utils.h"
#include <stdexcept>
#ifdef _WIN32
#include "TensorflowDllHandlerWin.h"
#else
#include "TensorflowDllHandlerLinux.h"
#endif

#include <vector>

NeuralNet::NeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* tfLiteModelPath, int32_t dymInputDim,
	const unsigned int* p_dymInputSizes, unsigned int dymOutputDim, const unsigned int* p_dymOutputSizes,
	bool stateful, double fixInterval)
{

	// set member to access dymola functions
	mp_modelicaUtilityHelper = p_modelicaUtilityHelper;

	// handling of the input
	m_inputDim = dymInputDim;
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
    if (stateful && fixInterval <= 0) {
        mp_modelicaUtilityHelper->ModelicaError("A stateful RNN is used with a samplePeriod less or equal than 0. "
                                                "Please enter the grid interval used to train the model as samplePeriod!");
    }


#ifdef _WIN32
    std::string tensorflowDllPath = Utils::getTensorflowDllPathWin();
    mp_tfdll = new TensorflowDllHandlerWin(tensorflowDllPath.c_str());
#else
    std::string tensorflowDllPath = Utils::getTensorflowDllPathLinux();
        mp_tfdll = new TensorflowDllHandlerLinux(tensorflowDllPath.c_str());
#endif

    mp_timeStepMngmt = new InputManagement(stateful, fixInterval, m_nInputEntries, mp_tfdll);

	// perform steps to create model
	loadAndInit(tfLiteModelPath);

}

NeuralNet::~NeuralNet()
{
	// clean up own arrays
	delete mp_inputSizes;
	mp_inputSizes = nullptr;
	m_inputDim = 0;
	m_nInputEntries = 0;

	delete mp_outputSizes;
	mp_outputSizes = nullptr;
	m_outputDim = 0;
	m_nOutputEntries = 0;

	// clean up allocated tflite stuff
	if (mp_interpreter) mp_tfdll->interpreterDelete(mp_interpreter);
	if (mp_options) mp_tfdll->interpreterOptionsDelete(mp_options);
	if (mp_model)  mp_tfdll->modelDelete(mp_model);

	// clean up time step manager
	delete mp_timeStepMngmt;

    // clean up the dll handler
    delete mp_tfdll;
}

void NeuralNet::loadAndInit(const char* tfliteModelPath)
{

	m_tfliteModelPath = tfliteModelPath;

	//mp_model = TfLiteModelCreateFromFile(m_tfliteModelPath);
	mp_model = mp_tfdll->createModelFromFile(m_tfliteModelPath);
	// zero pointer is returned if not found
	if (!mp_model) {
		std::string message = Utils::string_format("SMArtInt: Model not found - check path: %s", m_tfliteModelPath);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	}
	mp_options = mp_tfdll->interpreterOptionsCreate();
    mp_tfdll->interpreterOptionsSetNumThreads(mp_options, 1);

	// Create the interpreter.
	mp_interpreter = mp_tfdll->interpreterCreate(mp_model, mp_options);
	if (!mp_interpreter) {
		mp_modelicaUtilityHelper->ModelicaError("Failed to create interpreter");
	}

	// Allocate tensor buffers.
	if (mp_tfdll->interpreterAllocateTensors(mp_interpreter) != kTfLiteOk)
		mp_modelicaUtilityHelper->ModelicaError("Failed to allocate tensors!");
	// Find input tensors.
	if (mp_tfdll->interpreterGetInputTensorCount(mp_interpreter) != 1 && !mp_timeStepMngmt->isActive())
		mp_modelicaUtilityHelper->ModelicaError("SMArtInt can only handle models with single input");

	mp_flatInputTensor = mp_tfdll->interpreterGetInputTensor(mp_interpreter, 0);

	// set the casting function to correct types
	setInputCastFunction(mp_flatInputTensor);

	if (!mp_flatInputTensor) mp_modelicaUtilityHelper->ModelicaError("Failed to create m_input_tensor");

	// check dimensions - function throws error if not matching
	checkInputTensorSize();

	// adjust first dimension which is batch size
	if (mp_tfdll->tensorDim(mp_flatInputTensor, 0) != int(mp_inputSizes[0])) {

		std::string message = "SMArtInt: Adjust first dimension from " + Utils::string_format("%i", mp_tfdll->tensorDim(mp_flatInputTensor, 0)) + " to batch size " + Utils::string_format("%i\n", mp_inputSizes[0]);
		mp_modelicaUtilityHelper->ModelicaMessage(message.c_str());

		int* p_dymInputSizes;
		p_dymInputSizes = new int[m_inputDim];
		for (unsigned int i = 0; i < m_inputDim; ++i) {
			p_dymInputSizes[i] = int(mp_inputSizes[i]);
		}
        mp_tfdll->interpreterResizeInputTensor(mp_interpreter, 0, p_dymInputSizes,
                                               m_inputDim);

		// Reallocate tensor buffers for updated sizes
		if (mp_tfdll->interpreterAllocateTensors(mp_interpreter) != kTfLiteOk) {
			mp_modelicaUtilityHelper->ModelicaError("Failed to allocate tensors!");
		}
	}

	// check the number of outputs
	if (mp_tfdll->interpreterGetOutputTensorCount(mp_interpreter) != 1) {
		if (mp_timeStepMngmt->isActive()) {
			if (mp_tfdll->interpreterGetOutputTensorCount(mp_interpreter) != mp_tfdll->interpreterGetInputTensorCount(mp_interpreter)) {
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
		for (int i = 1; i < mp_tfdll->interpreterGetInputTensorCount(mp_interpreter); ++i) {
			try {
				mp_timeStepMngmt->addStateInp(mp_tfdll->interpreterGetInputTensor(mp_interpreter, i));
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
}

void NeuralNet::runInferenceFlatTensor(double time, double* input, unsigned int inputLength, double* output, unsigned int outputLength)
{
	// check the sizes
	if (m_nInputEntries != inputLength) {
		std::string message = Utils::string_format("SMArtInt: Wrong input length: in the interface were %i entries defined, whereas in current function call %i is specified!", m_nInputEntries, inputLength);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	}
	// check output size
	if (m_nOutputEntries != outputLength) {
		std::string message = Utils::string_format("SMArtInt: Wrong output length: in the interface were %i entries defined, whereas in current function call %i is specified!", m_nOutputEntries, outputLength);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	}

    unsigned int nSteps = 0;
    try {
        nSteps = mp_timeStepMngmt->manageNewStep(time, m_firstInvoke, input);
    } catch (std::exception& e) {
        mp_modelicaUtilityHelper->ModelicaError(e.what());
    }

	void* p_data = mp_tfdll->tensorData(mp_flatInputTensor);

	for (unsigned int i = 0; i < nSteps; ++i)
	{
		double* inpInput = mp_timeStepMngmt->handleInpts(time, i, input, m_firstInvoke);

		// we write the data directly into the data array of the tensor - the casting function is set to the correct
		// type
		for (unsigned int j = 0; j < m_nInputEntries; ++j) {
			mfp_castInput(inpInput[j], p_data, j);
		}

		// Run inference
		if (mp_tfdll->interpreterInvoke(mp_interpreter) != kTfLiteOk) {
			mp_modelicaUtilityHelper->ModelicaError("Inference failed");
		}
	}

	const TfLiteTensor* p_flatOutputTensor = mp_tfdll->interpreterGetOutputTensor(mp_interpreter, 0);

	if (m_firstInvoke) {
		// check sizes
		checkOutputTensorSize(p_flatOutputTensor);

		// set the casting function to correct types
		setOutputCastFunction(p_flatOutputTensor);

		// handle additional outputs for states
		for (int i = 1; i < mp_tfdll->interpreterGetOutputTensorCount(mp_interpreter); ++i) {
			try {
				mp_timeStepMngmt->addStateOut(mp_tfdll->interpreterGetOutputTensor(mp_interpreter, i));
			}
			catch (const std::invalid_argument& e) {
				mp_modelicaUtilityHelper->ModelicaError(e.what());
			}
		}
		// everything is fine - so it should not be done again
		m_firstInvoke = false;
	}

	// directly access the tensor data
	p_data = mp_tfdll->tensorData(p_flatOutputTensor);
	for (unsigned int i = 0; i < m_nOutputEntries; ++i) {
		mfp_castOutput(output[i], p_data, i);
	}

	mp_timeStepMngmt->updateFinishedStep(time, nSteps);
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
	if (mp_tfdll->tensorNumDims(mp_flatInputTensor) != m_inputDim)
	{
		std::string message = Utils::string_format(
                "SMArtInt: Wrong input dimensions : the loaded model has %i dimensions whereas in the "
                "interface %i is specified!", mp_tfdll->tensorNumDims(mp_flatInputTensor), m_inputDim);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	}
	// check the sizes in each dimension except for the first which is the batch size
	for (int32_t i = 1; i < m_inputDim; ++i) {
		if (mp_tfdll->tensorDim(mp_flatInputTensor, i) != int(mp_inputSizes[i]))
		{
			std::string message = "SMArtInt: Wrong input sizes. The loaded model has the sizes {";
			for (int32_t j = 0; j < m_inputDim; ++j) {
				message += Utils::string_format("%i", mp_tfdll->tensorDim(mp_flatInputTensor, j));
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
	if (mp_tfdll->tensorNumDims(p_flatOutputTensor) != m_outputDim)
	{
		std::string message = Utils::string_format("SMArtInt: Wrong output dimensions : the loaded model has %i dimensions whereas in the interface %i is specified!", mp_tfdll->tensorNumDims(p_flatOutputTensor), m_outputDim);
		mp_modelicaUtilityHelper->ModelicaError(message.c_str());
	}
	for (int32_t i = 0; i < m_outputDim; ++i) {
		if (mp_tfdll->tensorDim(p_flatOutputTensor, i) != int(mp_outputSizes[i]))
		{
			std::string message = "SMArtInt: Wrong output sizes. The loaded model has the sizes {";
			for (int32_t j = 0; j < m_outputDim; ++j) {
				message += Utils::string_format("%i", mp_tfdll->tensorDim(p_flatOutputTensor, j));
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
	switch (mp_tfdll->tensorType(tensor)) {
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
	switch (mp_tfdll->tensorType(tensor))
	{

	case kTfLiteFloat32:
		mfp_castOutput = &Utils::castFromFloat;
		break;
	default:
		mp_modelicaUtilityHelper->ModelicaError("Could not convert output data - SMArtIInt currently only supports TFLite models using floats");
		break;
	}
}