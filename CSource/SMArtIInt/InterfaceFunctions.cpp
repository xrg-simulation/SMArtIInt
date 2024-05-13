#include "InterfaceFunctions.h"
#include "NeuralNet.h"
#include "../../SMArtIInt/Resources/Include/ModelicaUtilityHelper.h"
#include <filesystem>
#include <variant>


void* NeuralNet_createObject(void* modelicaUtilityHelper, const char* ModelPath, unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes, bool stateful, double fixStep)
{
	ModelicaUtilityHelper* p_modelicaUtilityHelper = (ModelicaUtilityHelper*)modelicaUtilityHelper;
	#ifndef NDEBUG
	p_modelicaUtilityHelper->ModelicaMessage("!! SMArtIInt running in Debug Mode !!\n\n");
    #endif
    //ToDo: check fixStep (must it always be >0???)

    // check for model format
//    namespace fs = std::experimental::filesystem;
    namespace fs = std::filesystem;
    fs::path file_path(ModelPath);
    if (fs::exists(file_path)) {
        if (fs::is_regular_file(file_path)) {
            std::string extension = file_path.extension().string();
            if (extension == ".tflite") {
                std::string message = Utils::string_format("SMArtInt: TF-Lite Model recognized  - at path: %s\n",
                                                           ModelPath);
                p_modelicaUtilityHelper->ModelicaMessage(message.c_str());
                // create TfLiteNeuralNet pointer
                auto *p_neuralNet = new TfLiteNeuralNet(p_modelicaUtilityHelper, ModelPath,
                                                                   dymInputDim, p_dymInputSizes, dymOutputDim,
                                                                   p_dymOutputSizes, stateful, fixStep);
                p_neuralNet->printType();
                return (void *) p_neuralNet;

            } else if (extension == ".onnx") {
                std::string message = Utils::string_format("SMArtInt: ONNX Model recognized  - at path: %s\n",
                                                           ModelPath);
                p_modelicaUtilityHelper->ModelicaMessage(message.c_str());
                // to be added later!
                auto *p_neuralNet = new OnnxNeuralNet(p_modelicaUtilityHelper, ModelPath, dymInputDim,
                                                       p_dymInputSizes, dymOutputDim, p_dymOutputSizes, stateful,
                                                       fixStep);
                p_neuralNet->printType();
                return (void *) p_neuralNet;
            } else {
                std::string message = Utils::string_format("SMArtInt: No known model type recognized  - at path: %s\n",
                                                           ModelPath);
                p_modelicaUtilityHelper->ModelicaError(message.c_str());
                return nullptr;

            }
        }
    }
    std::string message = Utils::string_format("SMArtInt: Path to model is not correct: %s\n", ModelPath);
    p_modelicaUtilityHelper->ModelicaError(message.c_str());
    return nullptr;
}

void NeuralNet_destroyObject(void* externalObject)
{
    auto* neuralNetPtr = static_cast<NeuralNet*>(externalObject);
    auto p_neuralNet = dynamic_cast<TfLiteNeuralNet*>(neuralNetPtr);
    if(p_neuralNet)
    {
        p_neuralNet->printType();
        delete p_neuralNet;
    }
    else
    {
        auto p_neuralNet = dynamic_cast<OnnxNeuralNet*>(neuralNetPtr);
        if(p_neuralNet)
        {
            p_neuralNet->printType();
            //delete p_neuralNet;
        }
    }
}

void NeuralNet_runInferenceFlatTensor(void* externalObject, double time, double* input, unsigned int inputLength,
	double* output, unsigned int outputLength)
{
    auto* neuralNetPtr = static_cast<NeuralNet*>(externalObject);
    auto p_neuralNet = dynamic_cast<TfLiteNeuralNet*>(neuralNetPtr);
    if(p_neuralNet)
    {
        // run inference
        p_neuralNet->runInferenceFlatTensor(time, input, inputLength, output, outputLength);
    }
    else
    {
        auto p_neuralNet = dynamic_cast<OnnxNeuralNet*>(neuralNetPtr);
        if(p_neuralNet)
        {
            // run inference
            p_neuralNet->runInferenceFlatTensor(time, input, inputLength, output, outputLength);
        }
    }
}

void NeuralNet_initializeStates(void* externalObject, double* states, unsigned int nStateValues)
{
    auto* neuralNetPtr = static_cast<NeuralNet*>(externalObject);
    auto p_neuralNet = dynamic_cast<TfLiteNeuralNet*>(neuralNetPtr);
    if(p_neuralNet)
    {
        // initialize states
        p_neuralNet->initializeStates(states, nStateValues);
    }
    else
    {
        auto p_neuralNet = dynamic_cast<OnnxNeuralNet*>(neuralNetPtr);
        if(p_neuralNet)
        {
            // initialize states
            p_neuralNet->initializeStates(states, nStateValues);
        }
    }
}


