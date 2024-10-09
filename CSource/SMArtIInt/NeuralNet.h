#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include "tensorflow/lite/c/common.h"
#include "../../SMArtIInt/Resources/Include/ModelicaUtilityHelper.h"
#include "tensorflow/lite/c/c_api.h"
#include "InputManagementTF.h"
#include "InputManagementONNX.h"
#include <cstring>

class NeuralNet
{
public:
	NeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* tfLiteModelPath,
        unsigned int dymInputDim, const unsigned int* p_dymInputSizes, unsigned int dymOutputDim, const unsigned int* p_dymOutputSizes,
		bool stateful, double fixInterval);

	virtual ~NeuralNet();

    virtual void printType() {
        std::string message = Utils::string_format("\nSMArtIInt: Type is %s\n", "BaseClass");
        mp_modelicaUtilityHelper->ModelicaMessage(message.c_str());
        }

protected:

	ModelicaUtilityHelper* mp_modelicaUtilityHelper; // attribute to access dymola utility functions

	// in and output
    int32_t m_inputDim = 0; // dimension of input as specified in modelica
	unsigned int* mp_inputSizes = nullptr; // sizes of input as specified in modelica
	unsigned int m_nInputEntries; // total number of input entries

	unsigned int m_outputDim = 0; // dimensions of output
	unsigned int* mp_outputSizes = nullptr; // sizes of output as specified in modelica
	unsigned int m_nOutputEntries; // total number of output elements

	bool m_firstInvoke = true; // flag if outputs needs to be allocated etc
    bool m_statesInitialized = false;
};
