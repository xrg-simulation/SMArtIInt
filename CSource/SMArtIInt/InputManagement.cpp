#include "InputManagement.h"
#include <iostream>
#include "tensorflow/lite/c/c_api.h"
#include "Utils.h"
#include <vector>
#include <stdexcept>
#include <cstring>

InputManagement::InputManagement(bool stateful, double fixInterval, unsigned int nInputEntries)
{
	m_active = stateful;
	m_fixTimeIntv = fixInterval;
	m_nInputEntries = nInputEntries;

	if (m_active && m_fixTimeIntv > 0) {
		for (unsigned int i = 0; i < m_nStoredSteps; ++i) {
			mp_inputBuffer.getElement(i)->resize(nInputEntries);
		}
		mp_flatInterpolatedInp = new double[nInputEntries];
	}
	else {
		mp_flatInterpolatedInp = nullptr;
	}
	
	m_nStateArr = 0;
	m_nStateValues = 0;

	return;

}

InputManagement::~InputManagement()
{
	if (mp_flatInterpolatedInp) delete mp_flatInterpolatedInp;
}

bool InputManagement::isActive()
{
	return m_active;
}

bool InputManagement::addStateInp(TfLiteTensor* stateInpTensor)
{
	m_nStateArr += 1;
	for (unsigned int i = 0; i < m_nStoredSteps; ++i) {
		m_stateBuffer.getElement(i)->addStateInput(stateInpTensor);
	}
	mp_stateInpTensors.push_back(stateInpTensor);
	m_nStateValues += Utils::getNumElementsTensor(stateInpTensor);
	return true;
}

bool InputManagement::addStateOut(const TfLiteTensor* stateOutTensor)
{
	size_t i = mp_stateOutTensors.size();
	if (i < m_nStateArr) {
		mp_stateOutTensors.push_back(stateOutTensor);
		unsigned int unmatchedVals[2];
		int ret = Utils::compareTensorSizes(mp_stateInpTensors[i], mp_stateOutTensors[i], unmatchedVals);
		if (ret < 0) {
			throw std::invalid_argument(Utils::string_format("Unmatched number of dimension for state input and output # %i"
				" (Input has %i dimensions whereas output has %i dimensions)!", i, unmatchedVals[0], unmatchedVals[1]));
			return false;
		}
		else if (ret > 0) {
			throw std::invalid_argument(Utils::string_format("Unmatched number of sizes for state input and output # %i in dimension %i "
				"(Input has %i entries whereas output has %i entries)!"
				, i, ret, unmatchedVals[0], unmatchedVals[1]));
			return false;
		}
	}
	else {
		// Error
		throw std::invalid_argument(Utils::string_format("SMArtInt can only handle states in stateful=True if state inputs and state outputs are matching!"));
		return false;
	}
	//ToDo check type (and sizes??)
	return true;
}

double* InputManagement::handleInpts(double time, unsigned int iStep, double* flatInp, bool firstInvoke)
{
	// calculate the grid time at which the NNs has to be evaluated
	double gridTime = m_startTime + (int((mp_inputBuffer.getPrevIdx() - m_startTime) / m_fixTimeIntv) + (iStep + 1.0)) * m_fixTimeIntv;

	double* input_pointer;

	if (m_active && m_fixTimeIntv > 0) {
		// Interpolation of the regular input onto grid
		if (!firstInvoke) {
			std::vector<double>* currentInput = mp_inputBuffer.getCurrentValue();
			std::vector<double>* prevInput = mp_inputBuffer.getPrevValue();
			for (std::size_t i = 0; i < currentInput->size(); ++i) {
				mp_flatInterpolatedInp[i] = prevInput->at(i) + (flatInp[i] - prevInput->at(i)) / (time - mp_inputBuffer.getPrevIdx()) * (gridTime - mp_inputBuffer.getPrevIdx());
			}
		}
		else {
			for (unsigned int i = 0; i < m_nInputEntries; ++i) {
				mp_flatInterpolatedInp[i] = flatInp[i];
			}
		}
		// Handling of the state inputs
		if (iStep == 0) {
			// initialize states with results from previously accepted step
			Utils::stateInputsContainer* stateInputs = m_stateBuffer.getPrevValue();
			for (unsigned int i = 0; i < m_nStateArr; ++i) {
				std::memcpy(TfLiteTensorData(mp_stateInpTensors[i]), stateInputs->at(i), stateInputs->byteSizeAt(i));
			}
		}
		else {
			// copy state output to input
			for (unsigned int i = 0; i < m_nStateArr; ++i) {
				std::memcpy(TfLiteTensorData(mp_stateInpTensors[i]), TfLiteTensorData(mp_stateOutTensors[i]), TfLiteTensorByteSize(mp_stateOutTensors[i]));
			}
		}
		input_pointer = mp_flatInterpolatedInp;
	}
	else {
		input_pointer = flatInp;
	}

	return input_pointer;
}

unsigned int InputManagement::manageNewStep(double time, bool firstInvoke, double* input)
{
	unsigned int nSteps;
	if (m_active && m_fixTimeIntv > 0) {
		unsigned int iStep;
		if (firstInvoke) {
			m_startTime = time;
			nSteps = 1;
			mp_inputBuffer.initializeIdx(time, m_fixTimeIntv);
			std::vector<double>* value = mp_inputBuffer.getCurrentValue();
			for (std::size_t i = 0; i < value->size(); ++i) {
				value->at(i) = input[i];
			}
		}
		else
		{
			int test;
			if (!mp_inputBuffer.update(time, 1, test))
			{
				throw std::out_of_range(Utils::string_format("Index not found in buffer - need to go back more than %i steps after rejection. Contact support!", m_nStoredSteps));
			}

			std::vector<double>* value = mp_inputBuffer.getCurrentValue();
			for (std::size_t i = 0; i < value->size(); ++i) {
				value->at(i) = input[i];
			}

			iStep = int(time / m_fixTimeIntv);
			nSteps = iStep - int((mp_inputBuffer.getPrevIdx() - m_startTime) / m_fixTimeIntv);
			if (nSteps <= 0) nSteps = 0;
		}
		if (firstInvoke) {
			m_stateBuffer.initializeIdx(0, 2);
		}
		else {
			int test;
			if (!m_stateBuffer.update(iStep, 1, test))
			{
                throw std::out_of_range(Utils::string_format("Index not found in buffer - need to go back more than %i steps after rejection. Contact support!", m_nStoredSteps));
			}
		}
	}
	else {
		nSteps = 1;
	}
	return nSteps;
}

bool InputManagement::updateFinishedStep(double time, unsigned int nSteps)
{
	if (nSteps > 0) {
		for (unsigned int i = 0; i < m_nStateArr; ++i) {
			// handle the states
			std::memcpy(m_stateBuffer.getCurrentValue()->at(i), TfLiteTensorData(mp_stateOutTensors[i]), m_stateBuffer.getCurrentValue()->byteSizeAt(i));
		}
	}
	return true;
}

void InputManagement::initialize()
{
	for (unsigned int iInput = 0; iInput < m_nStateArr; ++iInput) {
		// the initialization will be done with m_currIdx = 0 and m_prvIdx = m_nStoredSteps - 1
		// therefore we store the the data in the last available index

		void (*castFunc)(const double&, void*, unsigned int);

		switch (TfLiteTensorType(mp_stateInpTensors[iInput])) {
			case kTfLiteFloat32:
				castFunc = &Utils::castToFloat;
				break;
			default:
				throw std::invalid_argument("Could not convert state data - SMArtIInt currently only supports TFLite models using floats)!");
				break;
		}

		void* p_data = m_stateBuffer.getPrevValue()->at(iInput);

		unsigned int n = Utils::getNumElementsTensor(mp_stateInpTensors[iInput]);

		for (unsigned int i = 0; i < n; ++i) {
			castFunc(0.0, p_data, i);
		}

	}
}

void InputManagement::initialize(double* p_stateValues, const unsigned int &nStateValues)
{
	unsigned int counter = 0;
	for (unsigned int iInput = 0; iInput < m_nStateArr; ++iInput) {
		// the initialization will be done with m_currIdx = 0 and m_prvIdx = m_nStoredSteps - 1
		// therefore we store the the data in the last available index

		if (nStateValues != m_nStateValues) {
			throw std::invalid_argument(Utils::string_format("SMArtIInt needs to initialize %i but %i are given", m_nStateValues, nStateValues));
		}

		void (*castFunc)(const double&, void*, unsigned int);

		switch (TfLiteTensorType(mp_stateInpTensors[iInput])) {
		case kTfLiteFloat32:
			castFunc = &Utils::castToFloat;
			break;
		default:
			throw std::invalid_argument("Could not convert state data - SMArtIInt currently only supports TFLite models using floats)!");
			break;
		}

		void* p_data = m_stateBuffer.getPrevValue()->at(iInput);

		unsigned int n = Utils::getNumElementsTensor(mp_stateInpTensors[iInput]);


		for (unsigned int i = 0; i < n; ++i) {
			castFunc(p_stateValues[counter], p_data, i);
		}
		counter += 1;
	}
}



