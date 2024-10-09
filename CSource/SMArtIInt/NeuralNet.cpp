
#include "NeuralNet.h"
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <filesystem>

NeuralNet::NeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* ModelPath, unsigned int dymInputDim,
	const unsigned int* p_dymInputSizes, unsigned int dymOutputDim, const unsigned int* p_dymOutputSizes,
	bool stateful, double fixInterval)
{

	// set member to access dymola functions
	mp_modelicaUtilityHelper = p_modelicaUtilityHelper;

	// handling of the input
	m_inputDim = static_cast<int32_t>(dymInputDim);
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
    // check if model is stateful and batched
    namespace fs = std::filesystem;
    fs::path file_path(ModelPath);
    std::string extension = file_path.extension().string();
    if (stateful && mp_inputSizes[0] != 1 && extension == ".onnx"){
        mp_modelicaUtilityHelper->ModelicaError("Stateful RNNs with batched inputs are not supported for onnx at the moment, "
                                                "but will be available in the future.");
    }

	// create the time step management object
    if (stateful && fixInterval <= 0) {
        mp_modelicaUtilityHelper->ModelicaError("A stateful RNN is used with a samplePeriod less or equal than 0. "
                                                "Please enter the grid interval used to train the model as samplePeriod!");
    }
}

NeuralNet::~NeuralNet()
{
    mp_modelicaUtilityHelper->ModelicaMessage("SMArtIInt: Destructor Base Neural Network\n");
	// clean up own arrays
	delete mp_inputSizes;
	mp_inputSizes = nullptr;
	m_inputDim = 0;
	m_nInputEntries = 0;

	delete mp_outputSizes;
	mp_outputSizes = nullptr;
	m_outputDim = 0;
	m_nOutputEntries = 0;
}
