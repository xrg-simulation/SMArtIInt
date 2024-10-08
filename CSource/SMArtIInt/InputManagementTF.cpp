//
// Created by RobertFlesch on 07.10.2024.
//

#include "InputManagementTF.h"

InputManagementTF::InputManagementTF(bool stateful, double fixInterval, unsigned int nInputEntries, TensorflowDllHandler* p_tfDll) :
        InputManagement(stateful, fixInterval, nInputEntries){
    mp_tfDll = p_tfDll;
}

bool InputManagementTF::addStateOut(const TfLiteTensor* stateOutTensor)
{
    size_t i = mp_stateOutTensors.size();
    if (i < m_nStateArr) {
        mp_stateOutTensors.push_back(stateOutTensor);
        unsigned int unmatchedVals[2];
        int ret = Utils::compareTensorSizes(mp_stateInpTensors[i], mp_stateOutTensors[i],
                                            unmatchedVals, mp_tfDll);
        if (ret < 0) {
            throw std::invalid_argument(Utils::string_format("Unmatched number of dimension for state "
                                                             "input and output # %i (Input has %i dimensions whereas "
                                                             "output has %i dimensions)!",
                                                             i, unmatchedVals[0], unmatchedVals[1]));
        }
        else if (ret > 0) {
            throw std::invalid_argument(Utils::string_format("Unmatched number of sizes for state input "
                                                             "and output # %i in dimension %i (Input has %i entries "
                                                             "whereas output has %i entries)!",
                                                             i, ret, unmatchedVals[0], unmatchedVals[1]));
        }
    }
    else {
        // Error
        throw std::invalid_argument("SMArtIInt can only handle states in "
                                    "stateful=True if state inputs and state outputs are "
                                    "matching!");
    }
    //ToDo check type (and sizes??)
    return true;
}

double* InputManagementTF::handleInpts(double time, unsigned int iStep, double* flatInp, bool firstInvoke)
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
            // the state buffer is filled after a successful step, so we have to take the current value
            Utils::StateInputsContainer* stateInputs = m_stateBuffer.getPrevValue();
            for (unsigned int i = 0; i < m_nStateArr; ++i) {
                std::memcpy(mp_tfDll->tensorData(mp_stateInpTensors[i]), stateInputs->at(i),
                            stateInputs->byteSizeAt(i));
            }
        }
        else {
            // copy state output to input
            for (unsigned int i = 0; i < m_nStateArr; ++i) {
                std::memcpy(mp_tfDll->tensorData(mp_stateInpTensors[i]),
                            mp_tfDll->tensorData(mp_stateOutTensors[i]),
                            mp_tfDll->tensorByteSize(mp_stateOutTensors[i]));
            }
        }
        input_pointer = mp_flatInterpolatedInp;
    }
    else {
        input_pointer = flatInp;
    }

    return input_pointer;
}

bool InputManagementTF::addStateInp(TfLiteTensor* stateInpTensor)
{
    m_nStateArr += 1;
    mp_stateInpTensors.push_back(stateInpTensor);
    m_nStateValues += Utils::getNumElementsTensor(stateInpTensor, mp_tfDll);
    return true;
}

bool InputManagementTF::updateFinishedStep(unsigned int nSteps)
{
    if (nSteps > 0) {
        const auto test = new Utils::StateInputsContainer();
        for (unsigned int i = 0; i < m_nStateArr; ++i) {
            test->addStateInput(mp_stateInpTensors[i], mp_tfDll);
            // handle the states
            std::memcpy(test->at(i), mp_tfDll->tensorData(mp_stateOutTensors[i]),
                        test->byteSizeAt(i));
        }
        m_stateBuffer.store(m_currentGridTime, test);
    }
    return true;
}

void InputManagementTF::initialize(double time, double* p_stateValues, const unsigned int &nStateValues)
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

        switch (mp_tfDll->tensorType(mp_stateInpTensors[iInput])) {
            case kTfLiteFloat32:
                castFunc = &Utils::castToFloat;
                break;
            default:
                throw std::invalid_argument(
                        "Could not convert state data - SMArtIInt currently only supports TFLite models using floats)!");
        }

        test->addStateInput(mp_stateInpTensors[iInput], mp_tfDll);

        void *p_data = test->at(iInput);

        unsigned int n = Utils::getNumElementsTensor(mp_stateInpTensors[iInput], mp_tfDll);

        for (unsigned int i = 0; i < n; ++i) {
            castFunc(p_stateValues[counter], p_data, i);
        }
        counter += 1;
    }
    m_stateBuffer.initialize(test);
    //m_stateBuffer.store(time, test);
}

