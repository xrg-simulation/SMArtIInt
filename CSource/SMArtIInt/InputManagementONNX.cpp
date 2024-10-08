//
// Created by RobertFlesch on 07.10.2024.
//

#include "InputManagementONNX.h"

InputManagementONNX::InputManagementONNX(bool stateful, double fixInterval, unsigned int nInputEntries) :
InputManagement(stateful, fixInterval, nInputEntries){
}

void InputManagementONNX::addStateOut(Ort::Value* stateOutTensor)
{
    size_t i = mp_OnnxStateOutTensors.size();
    if (i < m_nStateArr) {
        mp_OnnxStateOutTensors.push_back(stateOutTensor);
        unsigned int unmatchedVals[2];
        int ret = Utils::compareTensorSizes(mp_OnnxStateInpTensors[i], mp_OnnxStateOutTensors[i], unmatchedVals);
        if (ret < 0) {
            throw std::invalid_argument(Utils::string_format("Unmatched number of dimension for state "
                                                             "input and output # %i"
                                                             " (Input has %i dimensions whereas output has %i "
                                                             "dimensions)!", i, unmatchedVals[0], unmatchedVals[1]));
        }
        else if (ret > 0) {
            throw std::invalid_argument(Utils::string_format("Unmatched number of sizes for state "
                                                             "input and output # %i in dimension %i "
                                                             "(Input has %i entries whereas output has %i entries)!"
                    , i, ret, unmatchedVals[0], unmatchedVals[1]));
        }
    }
    else {
        // Error
        throw std::invalid_argument(Utils::string_format("SMArtIInt can only handle states in "
                                                         "stateful=True if state inputs and state outputs are "
                                                         "matching!"));
    }
    //ToDo check type (and sizes??)
}

bool InputManagementONNX::updateStateOut(Ort::Value* stateOutTensor)
{
    mp_OnnxStateOutTensors.push_back(stateOutTensor);
    return true;
}

double* InputManagementONNX::handleInpts(double time, unsigned int iStep, double* flatInp, bool firstInvoke)
{
    // stateful NN need to be evaluated a grid times. This method interpolates the inputs @time to the previous grid
    // interval specified with iStep; additionally the states itself are either taken from the buffer for an initial
    // step (iStep=0) or they are copied from the outputs which contain the values from the previous invoke

    double* input_pointer;

    if (m_active && m_fixTimeIntv > 0) {
        // Interpolation of the regular input onto grid
        if (mp_inputBuffer.size() > 1) {
            std::vector<double>* currentInput = mp_inputBuffer.getCurrentValue();
            std::vector<double>* prevInput = mp_inputBuffer.getPrevValue();
            // calculate the grid time at which the NNs has to be evaluated
            double gridTime = m_startTime + (int((mp_inputBuffer.getPrevIdx() - m_startTime) / m_fixTimeIntv)
                                             + (iStep + 1.0)) * m_fixTimeIntv;
            for (std::size_t i = 0; i < currentInput->size(); ++i) {
                mp_flatInterpolatedInp[i] = prevInput->at(i) +
                                            (flatInp[i] - prevInput->at(i)) / (time - mp_inputBuffer.getPrevIdx())
                                            * (gridTime - mp_inputBuffer.getPrevIdx());
            }
        }
        else {
            for (unsigned int i = 0; i < m_nInputEntries; ++i) {
                mp_flatInterpolatedInp[i] = flatInp[i];
            }
        }
        // Handling of the state inputs
        if (iStep == 0) {
            // initialize states with results from previously accepted step (take it from buffer)
            // previously an empty entry is created in the state buffer - this point here will be called multiple times
            // when iterating the current step: in order to use the value of the previous accepted step we will create
            // the empty entry first and use the previous value
            Utils::StateInputsContainer* stateInputs = m_stateBuffer.getPrevValue();
            for (unsigned int i = 0; i < m_nStateArr; ++i) {
                std::memcpy(mp_OnnxStateInpTensors[i]->GetTensorMutableRawData(), stateInputs->at(i),
                            stateInputs->byteSizeAt(i));
            }
        }
        else {
            // copy state output to input
            for (unsigned int i = 0; i < m_nStateArr; ++i) {
                std::memcpy(mp_OnnxStateInpTensors[i]->GetTensorMutableRawData(),
                            mp_OnnxStateOutTensors[i]->GetTensorMutableRawData(),
                            sizeof(mp_OnnxStateOutTensors[i]->GetTensorTypeAndShapeInfo().GetElementType()) * \
                                        mp_OnnxStateOutTensors[i]->GetTensorTypeAndShapeInfo().GetElementCount()
                            );
            }
        }
        input_pointer = mp_flatInterpolatedInp;
    }
    else {
        input_pointer = flatInp;
    }

    return input_pointer;
}

bool InputManagementONNX::addStateInp(Ort::Value* stateInpTensor)
{
    m_nStateArr += 1;
    mp_OnnxStateInpTensors.push_back(stateInpTensor);
    m_nStateValues += stateInpTensor->GetTensorTypeAndShapeInfo().GetElementCount();
    return true;
}

bool InputManagementONNX::updateFinishedStep(unsigned int nSteps)
{
    if (nSteps > 0) {
        const auto test = new Utils::StateInputsContainer();
        for (unsigned int i = 0; i < m_nStateArr; ++i) {
                test->addStateInput(mp_OnnxStateInpTensors[i]);
                std::memcpy(test->at(i), mp_OnnxStateOutTensors[i]->GetTensorMutableRawData(),
                            test->byteSizeAt(i));
        }
        m_stateBuffer.store(m_currentGridTime, test);
    }
    return true;
}

void InputManagementONNX::initialize(double time, double* p_stateValues, const unsigned int &nStateValues)
{
    unsigned int counter = 0;
    const auto test =  new Utils::StateInputsContainer();
    for (unsigned int iInput = 0; iInput < m_nStateArr; ++iInput) {
        // the initialization will be done with m_currIdx = 0 and m_prvIdx = m_nStoredSteps - 1
        // therefore we store the data in the last available index

        if (nStateValues != m_nStateValues) {
            throw std::invalid_argument(Utils::string_format(
                    "SMArtIInt needs to initialize %i but %i are given", m_nStateValues, nStateValues));
        }

        void (*castFunc)(const double &, void *, unsigned int);

        switch (mp_OnnxStateInpTensors[iInput]->GetTensorTypeAndShapeInfo().GetElementType()) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                castFunc = &Utils::castToFloat;
                break;
            default:
                throw std::invalid_argument(
                        "Could not convert state data - SMArtIInt currently only supports ONNX models using floats)!");
        }

        test->addStateInput(mp_OnnxStateInpTensors[iInput]);

        void *p_data = test->at(iInput);

        unsigned int n = mp_OnnxStateInpTensors[iInput]->GetTensorTypeAndShapeInfo().GetElementCount();

        for (unsigned int i = 0; i < n; ++i) {
            castFunc(0.0, p_data, i);
        }
        counter += 1;
    }
//m_stateBuffer.store(time, test);
    m_stateBuffer.initialize(test);
}
