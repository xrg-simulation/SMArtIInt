#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include "tensorflow/lite/c/common.h"
#include <stdlib.h>
#include "../../SMArtIInt/Resources/Include/ModelicaUtilityHelper.h"
#include "tensorflow/lite/c/c_api.h"
#include "InputManagement.h"
#include <cstring>
#include "../External/onnx/onnxruntime/include/onnxruntime_cxx_api.h"

class NeuralNet
{
public:
	NeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* tfLiteModelPath,
		unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes,
		bool stateful, double fixInterval);

	~NeuralNet();

    virtual void printType() {
        std::string message = Utils::string_format("\nSMArtInt: Type is %s\n", "BaseClass");
        mp_modelicaUtilityHelper->ModelicaMessage(message.c_str());
        }

protected:

	ModelicaUtilityHelper* mp_modelicaUtilityHelper; // attribute to access dymola utility functions

	InputManagement* mp_timeStepMngmt; // time step manager used for stateful RNNs

	// in and output
	unsigned int m_inputDim = 0; // dimension of input as specified in modelica
	unsigned int* mp_inputSizes = nullptr; // sizes of input as specified in modelica
	unsigned int m_nInputEntries; // total number of input entries

	unsigned int m_outputDim = 0; // dimensions of output
	unsigned int* mp_outputSizes = nullptr; // sizes of output as specified in modelica
	unsigned int m_nOutputEntries; // total number of output elements

	bool m_firstInvoke = true; // flag if outputs needs to be allocated etc
};

class TfLiteNeuralNet :public NeuralNet
{
public:
    TfLiteNeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* tfLiteModelPath,
                    unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes,
                    bool stateful, double fixInterval);

    ~TfLiteNeuralNet();

    void runInferenceFlatTensor(double time, double* input, unsigned int inputLength, double* output, unsigned int outputLength); // invoke the model

    void initializeStates(double* p_stateValues, const unsigned int& nStateValues); // function to intialize states with given values

    void printType() override{
        std::string message = Utils::string_format("\nSMArtInt: Type is %s\n", m_modelType);
        mp_modelicaUtilityHelper->ModelicaMessage(message.c_str());
    };
    const char* m_modelType = "TfLite";

    void loadAndInit(const char* tfliteModelPath);

private:
    const char* m_tfliteModelPath = ""; // path of the model

    TfLiteModel* mp_model = nullptr; // pointer to model
    TfLiteInterpreterOptions* mp_options = nullptr; // pointer to model options
    TfLiteInterpreter* mp_interpreter = nullptr; // pointer to interpreter

    TfLiteTensor* mp_flatInputTensor = nullptr; // pointer to flat input tensors from nn

    void (*mfp_castInput)(const double&, void*, unsigned int); // pointer to input casting function
    void (*mfp_castOutput)(double&, void*, unsigned int); // pointer to output casting function

    // internal function to prepare model - called by constructor
    void setInputCastFunction(TfLiteTensor* mp_flatInputTensor); // function to set the casting function for inputs
    void setOutputCastFunction(const TfLiteTensor* tensor); // function to set the casting function for outputs
    // util function to check sizes
    void checkInputTensorSize(); // check if the tensor sizes defined in modelica are equal to those in the model
    void checkOutputTensorSize(const TfLiteTensor* p_flatOutputTensor); // check if the tensor sizes defined in modelica are equal to those in the model

};

class OnnxNeuralNet :public NeuralNet
{
public:
    OnnxNeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* onnxModelPath,
                    unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes,
                    bool stateful, double fixInterval);

    ~OnnxNeuralNet();

    void runInferenceFlatTensor(double time, double* input, unsigned int inputLength, double* output, unsigned int outputLength); // invoke the model

    void initializeStates(double* p_stateValues, const unsigned int& nStateValues); // function to intialize states with given values

    void printType() override{
        std::string message = Utils::string_format("\nSMArtInt: Type is %s\n", m_modelType);
        mp_modelicaUtilityHelper->ModelicaMessage(message.c_str());
    };
    const char* m_modelType = "ONNX";

private:
    const char* m_onnxModelPath = ""; // path of the model

    Ort::Env* mp_model; // pointer to model
    Ort::SessionOptions mp_options; // pointer to model options
    Ort::Session* mp_session; // pointer to interpreter
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu( OrtDeviceAllocator, OrtMemTypeDefault); // onnx memory info

    std::vector<std::string> m_input_names; // vector with input names
    std::vector<std::int64_t> m_input_shapes; // vector with input shapes
    std::vector<std::string> m_output_names; // vector with input names
    std::vector<std::int64_t> m_output_shapes; // vector with input shapes

    std::vector<float>* input_data; // data for feature input
    std::vector<std::vector<float>>* tensorData; // data for state inputs
    std::vector<Ort::Value> output_tensors; // tensors to store the results

    std::vector<const char*> input_names_char; // input names as char; needed for onnx inference
    std::vector<const char*> output_names_char; // output names as char; needed for onnx inference

    void loadAndInit(const char* onnxModelPath); // internal function to prepare model - called by constructor

    void checkInputTensorSize(); // check if the tensor sizes defined in modelica are equal to those in the model
    void checkOutputTensorSize(); // check if the tensor sizes defined in modelica are equal to those in the model

    static std::vector<float> values_to_float(const std::vector<Ort::Value>& values); // convert tensor data to float vector

    static void print_tensor_data(const Ort::Value& value); // print tensor data in the console (only for debugging)

    static std::string print_shape(const std::vector<std::int64_t>& v); // print in- & output shapes of tensors in dymola
};