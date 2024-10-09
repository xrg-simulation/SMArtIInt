#include "InterfaceFunctions.h"
#include "NeuralNetTF.h"
#include "NeuralNetONNX.h"
#include <filesystem>
#include <variant>

void* NeuralNet_createObject(void* modelicaUtilityHelper, const char* ModelPath, unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes, bool stateful, double fixStep)
{
	auto* p_modelicaUtilityHelper = (ModelicaUtilityHelper*)modelicaUtilityHelper;
	#ifndef NDEBUG
	p_modelicaUtilityHelper->ModelicaMessage("!! SMArtIInt running in Debug Mode !!\n\n");
    #endif

    // check for model format
    namespace fs = std::filesystem;
    fs::path file_path(ModelPath);
    if (fs::exists(file_path)) {
        if (fs::is_regular_file(file_path)) {
            std::string extension = file_path.extension().string();
            if (extension == ".tflite") {
                std::string message = Utils::string_format("SMArtIInt: TF-Lite Model recognized  - at path: %s\n",
                                                           ModelPath);
                p_modelicaUtilityHelper->ModelicaMessage(message.c_str());
                // create TfLiteNeuralNet pointer
                auto *p_neuralNet = new TfLiteNeuralNet(p_modelicaUtilityHelper, ModelPath,
                                                                   dymInputDim, p_dymInputSizes, dymOutputDim,
                                                                   p_dymOutputSizes, stateful, fixStep);
                p_neuralNet->printType();
                return (void *) p_neuralNet;

            } else if (extension == ".onnx") {
                std::string message = Utils::string_format("SMArtIInt: ONNX Model recognized  - at path: %s\n",
                                                           ModelPath);
                p_modelicaUtilityHelper->ModelicaMessage(message.c_str());
                // to be added later!
                auto *p_neuralNet = new OnnxNeuralNet(p_modelicaUtilityHelper, ModelPath, dymInputDim,
                                                       p_dymInputSizes, dymOutputDim, p_dymOutputSizes, stateful,
                                                       fixStep);
                p_neuralNet->printType();
                return (void *) p_neuralNet;
            } else {
                std::string message = Utils::string_format("SMArtIInt: No known model type recognized  - at path: %s\n",
                                                           ModelPath);
                p_modelicaUtilityHelper->ModelicaError(message.c_str());
                return nullptr;

            }
        }
    }
    std::string message = Utils::string_format("SMArtIInt: Path to model is not correct: %s\n", ModelPath);
    p_modelicaUtilityHelper->ModelicaError(message.c_str());
    return nullptr;
}

void NeuralNet_destroyObject(void* externalObject)
{
    auto* neuralNetPtr = static_cast<NeuralNet*>(externalObject);
    delete neuralNetPtr;
}

void NeuralNet_runInferenceFlatTensor(void* externalObject, double time, double* input, unsigned int inputLength,
	double* output, unsigned int outputLength)
{
    auto* neuralNetPtr = static_cast<NeuralNet*>(externalObject);
    auto p_neuralNetTF = dynamic_cast<TfLiteNeuralNet*>(neuralNetPtr);
    if(p_neuralNetTF)
    {
        // run inference
        p_neuralNetTF->runInferenceFlatTensor(time, input, inputLength, output, outputLength);
    }
    else
    {
        auto p_neuralNetONNX = dynamic_cast<OnnxNeuralNet*>(neuralNetPtr);
        if(p_neuralNetONNX)
        {
            // run inference
            p_neuralNetONNX->runInferenceFlatTensor(time, input, inputLength, output, outputLength);
        }
    }
}

void NeuralNet_initializeStates(void* externalObject, double time,  double* states, unsigned int nStateValues)
{
    auto* neuralNetPtr = static_cast<NeuralNet*>(externalObject);
    auto p_neuralNetTF = dynamic_cast<TfLiteNeuralNet*>(neuralNetPtr);
    if(p_neuralNetTF)
    {
        // initialize states
        p_neuralNetTF->initializeStates(time, states, nStateValues);
    }
    else
    {
        auto p_neuralNetONNX = dynamic_cast<OnnxNeuralNet*>(neuralNetPtr);
        if(p_neuralNetONNX)
        {
            // initialize states
            p_neuralNetONNX->initializeStates(time, states, nStateValues);
        }
    }
}


