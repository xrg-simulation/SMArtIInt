#pragma once
#include "tensorflow/lite/c/c_api.h"
#include <string>
#include "NNBuffer.h"
#include <vector>
#include "Utils.h"

class InputManagement {

protected:
	bool m_active = false; // true if inputs are managed - for stateful NNs
	double m_fixTimeIntv = 0; // sample interval

	double m_startTime = 0; // time of the first call
    double m_currentGridTime{}; // time of the current grid

	NNBuffer<std::vector<double>, double> mp_inputBuffer = NNBuffer<std::vector<double>, double>(); // buffer for inputs
	NNBuffer<Utils::StateInputsContainer, double> m_stateBuffer = NNBuffer<Utils::StateInputsContainer, double>(); // buffer for states

	// arrays handling the normal function input
	unsigned int m_nInputEntries = 0; // total number of input elements
	double* mp_flatInterpolatedInp = nullptr; // storage for interpolated NN input

    // number of state arrays - is equal for in- and output
	unsigned int m_nStateValues = 0; // total number of state values

    // pointer to the inputs and outputs handling the states
    unsigned int m_nStateArr = 0;

    void createEmptyStateStorage();

public:
	InputManagement(bool stateful, double fixInterval, unsigned int nInputEntries);
	~InputManagement();

	[[nodiscard]] bool isActive() const; //check if is active

	// functions handling the stored state values
    void storeInputs(double time, const double* input);

	unsigned int calculateNumberOfSteps(double time, bool firstInvoke); // updates the buffer and returns the required number of calls of the neural net
	void initialize(double time); // initialize all states with zeros
    virtual void initialize(double time, double* p_stateValues, const unsigned int &nStateValues) = 0;
};

