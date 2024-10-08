#include "InputManagement.h"
#include <vector>

InputManagement::InputManagement(bool stateful, double fixInterval,
                                 unsigned int nInputEntries) {
	m_active = stateful;
	m_fixTimeIntv = fixInterval;
	m_nInputEntries = nInputEntries;

	if (m_active && m_fixTimeIntv > 0) {
		mp_flatInterpolatedInp = new double[nInputEntries];
	}
	else {
		mp_flatInterpolatedInp = nullptr;
	}
	
	m_nStateArr = 0;
	m_nStateValues = 0;
}

InputManagement::~InputManagement()
{
	delete mp_flatInterpolatedInp;
}

bool InputManagement::isActive() const
{
	return m_active;
}

void InputManagement::storeInputs(double time, const double* input){
    if (m_active && m_fixTimeIntv > 0) {

        // store the values required for the inputs of the NN in the buffer
        auto *p_store = new std::vector<double>(input, input + m_nInputEntries);
        mp_inputBuffer.store(time, p_store);

    }
}

void InputManagement::createEmptyStateStorage(){
    if (m_active && m_fixTimeIntv > 0) {
        m_stateBuffer.createEmptyEntry(m_currentGridTime);
    }
}

unsigned int InputManagement::calculateNumberOfSteps(const double time, bool firstInvoke)
{
	unsigned int nSteps;
	if (m_active && m_fixTimeIntv > 0) {
		unsigned int iStep;

		if (firstInvoke || mp_inputBuffer.size() <= 1) {
			m_startTime = time;
            m_currentGridTime = time;
			nSteps = 1;
		}
		else
		{
			iStep = int(time / m_fixTimeIntv);
            m_currentGridTime = iStep*m_fixTimeIntv;
			nSteps = iStep - int((mp_inputBuffer.getPrevIdx() - m_startTime) / m_fixTimeIntv);
			if (nSteps <= 0) nSteps = 0;
		}
        createEmptyStateStorage();
	}
	else {
		nSteps = 1;
	}
	return nSteps;
}

void InputManagement::initialize(double time)
{
    auto* test = new double[m_nStateValues];
    for (unsigned int i=0;i<m_nStateValues;++i){
        test[i] = 0;
    }

    this->initialize(time, test, m_nStateValues);
    delete[] test;
}


