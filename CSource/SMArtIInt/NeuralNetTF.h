//
// Created by TimHanke on 08.10.2024.
//

#include "NeuralNet.h"

#ifndef SMARTIINT_NEURALNETTF_H
#define SMARTIINT_NEURALNETTF_H

class TfLiteNeuralNet :public NeuralNet
{
public:
    TfLiteNeuralNet(ModelicaUtilityHelper* p_modelicaUtilityHelper, const char* tfLiteModelPath,
                    unsigned int dymInputDim, unsigned int* p_dymInputSizes, unsigned int dymOutputDim, unsigned int* p_dymOutputSizes,
                    bool stateful, double fixInterval);

    ~TfLiteNeuralNet() override;

    void runInferenceFlatTensor(double time, double* input, unsigned int inputLength, double* output, unsigned int outputLength); // invoke the model

    void initializeStates(double time, double* p_stateValues, const unsigned int& nStateValues); // function to initialize states with given values

    void printType() override{
        std::string message = Utils::string_format("\nSMArtIInt: Type is %s\n", m_modelType);
        mp_modelicaUtilityHelper->ModelicaMessage(message.c_str());
    };
    const char* m_modelType = "TfLite";

    void loadAndInit(const char* tfliteModelPath);

private:
    TensorflowDllHandler* mp_tfdll;

    InputManagementTF* mp_timeStepMngmt; // time step manager used for stateful RNNs

    const char* m_tfliteModelPath = ""; // path of the model

    TfLiteModel* mp_model = nullptr; // pointer to model
    TfLiteInterpreterOptions* mp_options = nullptr; // pointer to model options
    TfLiteInterpreter* mp_interpreter = nullptr; // pointer to interpreter

    TfLiteTensor* mp_flatInputTensor = nullptr; // pointer to flat input tensors from nn

    void (*mfp_castInput)(const double&, void*, unsigned int){}; // pointer to input casting function
    void (*mfp_castOutput)(double&, void*, unsigned int){}; // pointer to output casting function

    // internal function to prepare model - called by constructor
    void setInputCastFunction(TfLiteTensor* mp_flatInputTensor); // function to set the casting function for inputs
    void setOutputCastFunction(const TfLiteTensor* tensor); // function to set the casting function for outputs
    // util function to check sizes
    void checkInputTensorSize(); // check if the tensor sizes defined in modelica are equal to those in the model
    void checkOutputTensorSize(const TfLiteTensor* p_flatOutputTensor); // check if the tensor sizes defined in modelica are equal to those in the model
};

#endif //SMARTIINT_NEURALNETTF_H
