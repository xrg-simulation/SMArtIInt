//
// Created by TimHanke on 08.10.2024.
//

#include <algorithm>
#include "NeuralNetONNX.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "Utils.h"

OnnxNeuralNet::OnnxNeuralNet(ModelicaUtilityHelper *p_modelicaUtilityHelper, const char *onnxModelPath,
                             unsigned int dymInputDim, unsigned int *p_dymInputSizes, unsigned int dymOutputDim,
                             unsigned int *p_dymOutputSizes, bool stateful, double fixInterval) : NeuralNet(
        p_modelicaUtilityHelper, onnxModelPath,
        dymInputDim, p_dymInputSizes, dymOutputDim, p_dymOutputSizes,
        stateful, fixInterval) {

    mp_timeStepMngmt = new InputManagementONNX(stateful, fixInterval, m_nInputEntries);

    // perform steps to create model
    OnnxNeuralNet::loadAndInit(onnxModelPath);
}

OnnxNeuralNet::~OnnxNeuralNet() {
    mp_modelicaUtilityHelper->ModelicaMessage("SMArtIInt: Destructor ONNX Neural Network\n");
    // clean up allocated onnx stuff - Correct way?
    delete(mp_session);
    delete(mp_model);
    delete(input_data);
    delete(tensorData);
    // clean up time step manager
    delete mp_timeStepMngmt;
}

void OnnxNeuralNet::loadAndInit(const char* onnxModelPath)
{
    m_onnxModelPath = onnxModelPath;

    mp_model = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test_onnx");

#ifdef _MSC_VER
    // convert const char* in wchar_t*
    size_t length = 0;
    mbstowcs_s(&length, nullptr, 0, onnxModelPath, _TRUNCATE);
    auto* model_path_wchar = new wchar_t[length + 1];
    // Create the interpreter.
    mbstowcs_s(nullptr, model_path_wchar, length + 1, onnxModelPath, length);
    mp_session = new Ort::Session(*mp_model,  model_path_wchar, mp_options);
    std::cout << "MSC" << std::endl;
#else
    mp_session = new Ort::Session(*mp_model,  onnxModelPath, mp_options);
    std::cout << "WSL" << std::endl;
#endif

    // Allocate tensor buffers.
    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t i = 0; i < mp_session->GetInputCount(); i++) {
        m_input_names.emplace_back(std::move(mp_session->GetInputNameAllocated(i, allocator)).get());
        m_input_shapes = mp_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }
    m_input_shapes = mp_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    for (std::size_t i = 0; i < mp_session->GetOutputCount(); i++) {
        m_output_names.emplace_back(std::move(mp_session->GetOutputNameAllocated(i, allocator)).get());
        m_output_shapes = mp_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }
    m_output_shapes = mp_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    input_names_char = std::vector<const char *>(m_input_names.size(), nullptr);
    std::transform(std::begin(m_input_names), std::end(m_input_names), std::begin(input_names_char),
                   [&](const std::string &str) { return str.c_str(); });

    output_names_char = std::vector<const char *>(m_output_names.size(), nullptr);
    std::transform(std::begin(m_output_names), std::end(m_output_names), std::begin(output_names_char),
                   [&](const std::string &str) { return str.c_str(); });

    if (mp_session->GetInputCount() != 1 && !mp_timeStepMngmt->isActive()) {
        mp_modelicaUtilityHelper->ModelicaError("SMArtIInt can only handle models with single input");
    }

    // adjust first dimension which is batch size if batch size is dynamic
    m_input_shapes[0] = (m_input_shapes[0] == -1 && mp_inputSizes[0] == 1) ? 1 : m_input_shapes[0];
    m_output_shapes[0] = (m_output_shapes[0] == -1 && mp_outputSizes[0] == 1) ? 1 : m_output_shapes[0];

    // ToDo Adjusting Batchsize for stateful models: every input and output (state in- and outputs) needed to be adjusted with an batchsize
    if (m_input_shapes[0] != mp_inputSizes[0]){
        std::string message = "SMArtIInt: Adjust first dimension from " + Utils::string_format("%i", m_input_shapes[0]) + " to batch size " + Utils::string_format("%i\n", mp_inputSizes[0]);
        mp_modelicaUtilityHelper->ModelicaMessage(message.c_str());

        m_input_shapes[0] = mp_inputSizes[0]; //1
        m_output_shapes[0] = mp_outputSizes[0];
    }

    // check input and output size
    checkInputTensorSize();
    checkOutputTensorSize();

    // check the number of outputs
    if (mp_session->GetOutputCount() != 1) {
        if (mp_timeStepMngmt->isActive()) {
            if (mp_session->GetOutputCount() != mp_session->GetInputCount()) {
                mp_modelicaUtilityHelper->ModelicaError("SMArtIInt: Stateful handling can only be done if model has the same number of inputs (=) and outputs");
            }
        }
        else {
            mp_modelicaUtilityHelper->ModelicaError("SMArtIInt can only handle models with single output!");
        }
    }

    // Handle states as additional inputs
    if (mp_timeStepMngmt->isActive()) {
        mp_modelicaUtilityHelper->ModelicaMessage("SMArtIInt: Handling additional inputs as states");
        tensorData = new std::vector<std::vector<float>>(static_cast<int>(mp_session->GetInputCount()) -1);
        for (int i = 1; i < mp_session->GetInputCount(); ++i) {
            try {
                std::vector<int64_t> input_shape;
                input_shape = mp_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                input_shape[0] = (input_shape[0] == -1) ? 1 : input_shape[0];

                size_t totalSize = 1;
                for (int64_t dim : input_shape) {
                    totalSize *= dim;
                }
                tensorData->at(i-1) = std::vector<float>(totalSize, 0.0f);
                auto* tensor = new Ort::Value(Ort::Value::CreateTensor<float>(memInfo, (*tensorData)[i-1].data(), (*tensorData)[i-1].size(), input_shape.data(), input_shape.size()));
                mp_timeStepMngmt->addStateInp(tensor);

            }
            catch (const std::invalid_argument& e) {
                mp_modelicaUtilityHelper->ModelicaError(e.what());
            }
        }
    }
    input_data = new std::vector<float>(m_nInputEntries);
}

std::string OnnxNeuralNet::print_shape(const std::vector<std::int64_t>& v) {
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

void OnnxNeuralNet::runInferenceFlatTensor(double time, double* input, unsigned int inputLength, double* output, unsigned int outputLength)
{
    // check the sizes
    if (m_nInputEntries != inputLength) {
        std::string message = Utils::string_format("SMArtIInt: Wrong input length: in the interface were %i entries defined, whereas in current function call %i is specified!", m_nInputEntries, inputLength);
        mp_modelicaUtilityHelper->ModelicaError(message.c_str());
    }
    // check output size
    if (m_nOutputEntries != outputLength) {
        std::string message = Utils::string_format("SMArtIInt: Wrong output length: in the interface were %i entries defined, whereas in current function call %i is specified!", m_nOutputEntries, outputLength);
        mp_modelicaUtilityHelper->ModelicaError(message.c_str());
    }

    if (m_firstInvoke & !m_statesInitialized) {
        // Initialize states if available
        mp_timeStepMngmt->InputManagement::initialize(time);
        m_statesInitialized = true;
    }

    static std::vector<float> result;
    unsigned int nSteps = 0;
    try {
        mp_timeStepMngmt->storeInputs(time, input);
        nSteps = mp_timeStepMngmt->calculateNumberOfSteps(time, m_firstInvoke);
    } catch (std::exception& e) {
        mp_modelicaUtilityHelper->ModelicaError(e.what());
    }

    for (unsigned int i = 0; i < nSteps; ++i)
    {
        double* inpInput = mp_timeStepMngmt->handleInpts(time, i, input, m_firstInvoke);

        // we write the data directly into the data array of the tensor
        for (unsigned int j = 0; j < m_nInputEntries; ++j) {
            (*input_data)[j] = static_cast<float>(inpInput[j]);
        }

        // Feature input
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memInfo, input_data->data(), input_data->size(), m_input_shapes.data(), m_input_shapes.size()));

        // Additional state inputs
        auto* stateInputs1 = new std::vector<std::vector<float>>();
        int count = 0;
        for (auto element: mp_timeStepMngmt->mp_OnnxStateInpTensors) {
            stateInputs1->emplace_back(element->GetTensorTypeAndShapeInfo().GetElementCount(), 1.0f);
            auto stateShape = element->GetTensorTypeAndShapeInfo().GetShape();
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memInfo, (*stateInputs1)[count].data(), (*stateInputs1)[count].size(), stateShape.data(), stateShape.size()));
            auto size = input_tensors[count + 1].GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(input_tensors[count + 1].GetTensorTypeAndShapeInfo().GetElementType());
            std::memcpy(input_tensors[count + 1].GetTensorMutableRawData(),mp_timeStepMngmt->mp_OnnxStateInpTensors[count]->GetTensorMutableRawData(), size);
            count = count +1;
        }
//        auto input_check = values_to_float(input_tensors);

        // Run inference
        try {
            output_tensors = mp_session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                             input_names_char.size(), output_names_char.data(), output_names_char.size());

            result = values_to_float(output_tensors);

            if (m_firstInvoke) {
                for (int j = 1; j < mp_session->GetOutputCount(); ++j) {
                    try {
                        mp_timeStepMngmt->addStateOut(&output_tensors[j]);
                    }
                    catch (const std::invalid_argument &e) {
                        mp_modelicaUtilityHelper->ModelicaError(e.what());
                    }
                }
                m_firstInvoke = false;
            }
            else{
                mp_timeStepMngmt->mp_OnnxStateOutTensors.clear();
                for (int j = 1; j < mp_session->GetOutputCount(); ++j) {
                    mp_timeStepMngmt->updateStateOut(&output_tensors[j]);
                }
            }

        } catch (const Ort::Exception& exception) {
            std::string message = "ERROR running model inference: " + std::string(exception.what()) + "\n";
            mp_modelicaUtilityHelper->ModelicaError(message.c_str());
            exit(-1);
        }
        delete (stateInputs1);
    }

    for (int j = 0; j<m_nOutputEntries; j++){
        output[j] = static_cast<double>(result[j]);
    }

    mp_timeStepMngmt->updateFinishedStep(nSteps);

}

void OnnxNeuralNet::initializeStates(double time, double* p_stateValues, const unsigned int& nStateValues)
{
    try {
        mp_timeStepMngmt->initialize(time, p_stateValues, nStateValues);
    }
    catch (const std::invalid_argument& e) {
        mp_modelicaUtilityHelper->ModelicaError(e.what());
    }
}

void OnnxNeuralNet::checkInputTensorSize()
{
    if (mp_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetDimensionsCount() != m_inputDim)
    {
        std::string message = Utils::string_format("SMArtIInt: Wrong input dimensions : the loaded model has %i dimensions whereas in the interface %i is specified!", mp_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetDimensionsCount(), m_inputDim);
        mp_modelicaUtilityHelper->ModelicaError(message.c_str());
    }
    // check the sizes in each dimension except for the first which is the batch size
    for (unsigned int i = 1; i < m_inputDim; ++i) {
        if (mp_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[i] != int(mp_inputSizes[i]))
        {
            std::string message = "SMArtIInt: Wrong input sizes. The loaded model has the sizes {";
            for (unsigned int j = 0; j < m_inputDim; ++j) {
                message += Utils::string_format("%i", abs(mp_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[j]));
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

void OnnxNeuralNet::checkOutputTensorSize()
{
    // check dimensions
    if (mp_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetDimensionsCount() != m_outputDim)
    {
        std::string message = Utils::string_format("SMArtIInt: Wrong output dimensions : the loaded model has %i dimensions whereas in the interface %i is specified!", mp_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetDimensionsCount(), m_outputDim);
        mp_modelicaUtilityHelper->ModelicaError(message.c_str());
    }
    for (unsigned int i = 0; i < m_outputDim; ++i) {
        if (i == 0 && mp_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[i] == -1){

        }
        else if (abs(mp_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[i]) != int(mp_outputSizes[i]))
        {
            std::string message = "SMArtIInt: Wrong output sizes. The loaded model has the sizes {";
            for (unsigned int j = 0; j < m_outputDim; ++j) {
                message += Utils::string_format("%i", abs(mp_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[j])); // abs for not defined batch size (-1)
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


std::vector<float> OnnxNeuralNet::values_to_float(const std::vector<Ort::Value>& values) {
    std::vector<float> result;
    for (const auto& value : values) {
        if (value.IsTensor()) {
            auto tensor_info = value.GetTensorTypeAndShapeInfo();
            if (tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                // get values as floats
                auto* tensor_data = value.GetTensorData<float>();
                result.insert(result.end(), tensor_data, tensor_data + tensor_info.GetElementCount());
            } else {
                throw std::runtime_error("Tensor Data Type not supported!");
            }
        } else {
            throw std::runtime_error("Value is not a Tensor!");
        }
    }
    return result;
}

void OnnxNeuralNet::print_tensor_data(const Ort::Value& value) {
    // check if value is tensor
    if (value.IsTensor()) {
        // access the tensor shape
        auto tensor_info = value.GetTensorTypeAndShapeInfo();
        auto tensor_shape = tensor_info.GetShape();

        // check if type of tensor is a float
        if (tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            // access data
            const auto* tensor_data = value.GetTensorData<float>();

            // print data
            std::cout << "Tensor Data: [";
            for (size_t i = 0; i < tensor_info.GetElementCount(); ++i) {
                std::cout << tensor_data[i];
                if (i < tensor_info.GetElementCount() - 1) std::cout << ", ";
            }
            std::cout << "]\n" << std::endl;
        } else {
            std::cout << "Tensor Data Type not supported!" << std::endl;
        }
    } else {
        std::cout << "Value is not a Tensor!" << std::endl;
    }
}