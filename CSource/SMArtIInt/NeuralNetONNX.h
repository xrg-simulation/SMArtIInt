//
// Created by TimHanke on 08.10.2024.
//

#include "NeuralNet.h"

#ifndef SMARTIINT_NEURALNETONNX_H
#define SMARTIINT_NEURALNETONNX_H

class OnnxNeuralNet :public NeuralNet
{
public:
    OnnxNeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* onnxModelPath,
                  unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes,
                  bool stateful, double fixInterval);

    ~OnnxNeuralNet() override;

    void runInferenceFlatTensor(double time, double* input, unsigned int inputLength, double* output, unsigned int outputLength); // invoke the model

    void initializeStates(double time, double* p_stateValues, const unsigned int& nStateValues); // function to initialize states with given values

    void printType() override{
        std::string message = Utils::string_format("\nSMArtIInt: Type is %s\n", m_modelType);
        mp_modelicaUtilityHelper->ModelicaMessage(message.c_str());
    };
    const char* m_modelType = "ONNX";

private:
    const char* m_onnxModelPath = ""; // path of the model

    InputManagementONNX* mp_timeStepMngmt; // time step manager used for stateful RNNs

    Ort::Env* mp_model{}; // pointer to model
    Ort::SessionOptions mp_options; // pointer to model options
    Ort::Session* mp_session{}; // pointer to interpreter
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu( OrtDeviceAllocator, OrtMemTypeDefault); // onnx memory info

    std::vector<std::string> m_input_names; // vector with input names
    std::vector<std::int64_t> m_input_shapes; // vector with input shapes
    std::vector<std::string> m_output_names; // vector with input names
    std::vector<std::int64_t> m_output_shapes; // vector with input shapes

    std::vector<float>* input_data{}; // data for feature input
    std::vector<std::vector<float>>* tensorData{}; // data for state inputs
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

#endif //SMARTIINT_NEURALNETONNX_H
