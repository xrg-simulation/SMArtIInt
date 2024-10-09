//
// Created by TimHanke on 08.10.2024.
//

#include "NeuralNetTF.h"
#ifdef _WIN32
#include "TensorflowDllHandlerWin.h"
#else
#include "TensorflowDllHandlerLinux.h"
#endif

TfLiteNeuralNet::TfLiteNeuralNet(ModelicaUtilityHelper *p_modelicaUtilityHelper, const char *tfLiteModelPath,
                                 unsigned int dymInputDim, unsigned int *p_dymInputSizes, unsigned int dymOutputDim,
                                 unsigned int *p_dymOutputSizes, bool stateful, double fixInterval) : NeuralNet(
        p_modelicaUtilityHelper, tfLiteModelPath,
        dymInputDim, p_dymInputSizes, dymOutputDim, p_dymOutputSizes,
        stateful, fixInterval) {

#ifdef _WIN32
    try {
        std::string tensorflowDllPath = Utils::getTensorflowDllPathWin();
        mp_tfdll = new TensorflowDllHandlerWin(tensorflowDllPath.c_str());
    } catch (std::runtime_error& e) {
        mp_modelicaUtilityHelper->ModelicaError("Unable to detect tensorflow path");
    }
#else
    try {
        std::string tensorflowDllPath = Utils::getTensorflowDllPathLinux();
        mp_tfdll = new TensorflowDllHandlerLinux(tensorflowDllPath.c_str());
    } catch (std::runtime_error& e) {
        mp_modelicaUtilityHelper->ModelicaError("Unable to detect tensorflow path");
    }
#endif

    mp_timeStepMngmt = new InputManagementTF(stateful, fixInterval, m_nInputEntries, mp_tfdll);

    // perform steps to create model
    TfLiteNeuralNet::loadAndInit(tfLiteModelPath);

}

TfLiteNeuralNet::~TfLiteNeuralNet() {
    mp_modelicaUtilityHelper->ModelicaMessage("SMArtIInt: Destructor TFLite Neural Network\n");
    // clean up allocated tflite stuff
    if (mp_interpreter) mp_tfdll->interpreterDelete(mp_interpreter);
    if (mp_options) mp_tfdll->interpreterOptionsDelete(mp_options);
    if (mp_model) mp_tfdll->modelDelete(mp_model);
    // clean up time step manager
    delete mp_timeStepMngmt;
}

void TfLiteNeuralNet::loadAndInit(const char* tfliteModelPath)
{

    m_tfliteModelPath = tfliteModelPath;
    mp_model = mp_tfdll->createModelFromFile(m_tfliteModelPath);
    // zero pointer is returned if not found
    if (!mp_model) {
        std::string message = Utils::string_format("SMArtIInt: Model not found - check path: %s",
                                                   m_tfliteModelPath);
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
        mp_modelicaUtilityHelper->ModelicaError("SMArtIInt can only handle models with single input");

    mp_flatInputTensor = mp_tfdll->interpreterGetInputTensor(mp_interpreter, 0);

    // set the casting function to correct types
    setInputCastFunction(mp_flatInputTensor);

    if (!mp_flatInputTensor) mp_modelicaUtilityHelper->ModelicaError("Failed to create m_input_tensor");

    // check dimensions - function throws error if not matching
    checkInputTensorSize();

    // adjust first dimension which is batch size
    if (mp_tfdll->tensorDim(mp_flatInputTensor, 0) != int(mp_inputSizes[0])) {

        std::string message = "SMArtIInt: Adjust first dimension from " +
                              Utils::string_format("%i", mp_tfdll->tensorDim(mp_flatInputTensor, 0)) +
                              " to batch size " + Utils::string_format("%i\n", mp_inputSizes[0]);
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
            if (mp_tfdll->interpreterGetOutputTensorCount(mp_interpreter) !=
                mp_tfdll->interpreterGetInputTensorCount(mp_interpreter)) {
                mp_modelicaUtilityHelper->ModelicaError("SMArtIInt: Stateful handling can only be done if model has "
                                                        "the same number of inputs (=) and outputs");
            }
        }
        else {
            mp_modelicaUtilityHelper->ModelicaError("SMArtIInt can only handle models with single output!");
        }
    }

    // Handle states as additional inputs
    if (mp_timeStepMngmt->isActive()) {
        mp_modelicaUtilityHelper->ModelicaMessage("SMArtIInt: Handling additional inputs as states");
        //mp_timeStepMngmt->setNumberOfStates(TfLiteInterpreterGetInputTensorCount(mp_interpreter) - 1);
        for (int i = 1; i < mp_tfdll->interpreterGetInputTensorCount(mp_interpreter); ++i) {
            try {
                mp_timeStepMngmt->addStateInp(
                        mp_tfdll->interpreterGetInputTensor(mp_interpreter, i));
            }
            catch (const std::invalid_argument& e) {
                mp_modelicaUtilityHelper->ModelicaError(e.what());
            }
        }
    }

    // dimensions etc of output tensor is only available after calling invoke so we check the infos after
    // the call - that it is only done once use the following variable
    m_firstInvoke = true;
}

void TfLiteNeuralNet::runInferenceFlatTensor(double time, double* input, unsigned int inputLength, double* output,
                                             unsigned int outputLength)
{
    // check the sizes
    if (m_nInputEntries != inputLength) {
        std::string message = Utils::string_format("SMArtIInt: Wrong input length: in the interface were %i "
                                                   "entries defined, whereas in current function call %i is specified!",
                                                   m_nInputEntries, inputLength);
        mp_modelicaUtilityHelper->ModelicaError(message.c_str());
    }
    // check output size
    if (m_nOutputEntries != outputLength) {
        std::string message = Utils::string_format("SMArtIInt: Wrong output length: in the interface were %i "
                                                   "entries defined, whereas in current function call %i is specified!",
                                                   m_nOutputEntries, outputLength);
        mp_modelicaUtilityHelper->ModelicaError(message.c_str());
    }
    if (m_firstInvoke & !m_statesInitialized) {
        // Initialize states if available
        mp_timeStepMngmt->InputManagement::initialize(time);
        m_statesInitialized = true;
    }

    unsigned int nSteps = 0;
    try {
        mp_timeStepMngmt->storeInputs(time, input);
        nSteps = mp_timeStepMngmt->calculateNumberOfSteps(time, m_firstInvoke);
    } catch (std::exception& e) {
        mp_modelicaUtilityHelper->ModelicaError(e.what());
    }

    void* p_data = mp_tfdll->tensorData(mp_flatInputTensor);

    for (unsigned int i = 0; i < nSteps; ++i)
    {
        // for each intermediate step, call the methods handling the inputs: it will interpolate the inputs to the
        // required grid time, and it will fill the tensors of the state inputs
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
                mp_timeStepMngmt->addStateOut(
                        mp_tfdll->interpreterGetOutputTensor(mp_interpreter, i)
                );
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

    mp_timeStepMngmt->updateFinishedStep(nSteps);
}

void TfLiteNeuralNet::initializeStates(double time, double* p_stateValues, const unsigned int& nStateValues)
{
    m_statesInitialized = true;
    try {
        mp_timeStepMngmt->initialize(time, p_stateValues, nStateValues);
    }
    catch (const std::invalid_argument& e) {
        mp_modelicaUtilityHelper->ModelicaError(e.what());
    }
}

void TfLiteNeuralNet::setInputCastFunction(TfLiteTensor* tensor)
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

void TfLiteNeuralNet::setOutputCastFunction(const TfLiteTensor* tensor)
{
    switch (mp_tfdll->tensorType(tensor)) {
        case kTfLiteFloat32:
            mfp_castOutput = &Utils::castFromFloat;
            break;
        default:
            mp_modelicaUtilityHelper->ModelicaError("Could not convert output data - SMArtIInt currently only supports TFLite models using floats");
            break;
    }
}

void TfLiteNeuralNet::checkInputTensorSize()
{
    if (mp_tfdll->tensorNumDims(mp_flatInputTensor) != m_inputDim)
    {
        std::string message = Utils::string_format(
                "SMArtIInt: Wrong input dimensions : the loaded model has %i dimensions whereas in the "
                "interface %i is specified!", mp_tfdll->tensorNumDims(mp_flatInputTensor), m_inputDim);
        mp_modelicaUtilityHelper->ModelicaError(message.c_str());
    }
    // check the sizes in each dimension except for the first which is the batch size
    for (int32_t i = 1; i < m_inputDim; ++i) {
        if (mp_tfdll->tensorDim(mp_flatInputTensor, i) != int(mp_inputSizes[i]))
        {
            std::string message = "SMArtIInt: Wrong input sizes. The loaded model has the sizes {";
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

void TfLiteNeuralNet::checkOutputTensorSize(const TfLiteTensor* p_flatOutputTensor)
{
    // check dimensions
    if (mp_tfdll->tensorNumDims(p_flatOutputTensor) != m_outputDim)
    {
        std::string message = Utils::string_format("SMArtIInt: Wrong output dimensions : the loaded model has %i dimensions whereas in the interface %i is specified!", mp_tfdll->tensorNumDims(p_flatOutputTensor), m_outputDim);
        mp_modelicaUtilityHelper->ModelicaError(message.c_str());
    }
    for (int32_t i = 0; i < m_outputDim; ++i) {
        if (mp_tfdll->tensorDim(p_flatOutputTensor, i) != int(mp_outputSizes[i]))
        {
            std::string message = "SMArtIInt: Wrong output sizes. The loaded model has the sizes {";
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