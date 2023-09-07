#pragma once
#include "tensorflow/lite/c/c_api.h"
#include <string>
#include "RollingBuffer.h"
#include <vector>
#include "Utils.h"

class InputManagement
{

private:
	bool m_active = false; // true if inputs are managed - for stateful NNs
	double m_fixTimeIntv = 0; // sample interval

	double m_startTime = 0; // time of the first call

	static const int m_nStoredSteps = 1000; // number of store previous steps

	RollingBuffer<std::vector<double>, double> mp_inputBuffer = RollingBuffer<std::vector<double>, double>(m_nStoredSteps); // buffer for inputs
	RollingBuffer<Utils::stateInputsContainer, int> m_stateBuffer = RollingBuffer<Utils::stateInputsContainer, int>(m_nStoredSteps); // buffer for states

	// arrays handling the normal function input
	unsigned int m_nInputEntries = 0; // total number of input elements
	double* mp_flatInterpolatedInp = nullptr; // storage for interpolated NN input
	
	// pointer to the inputs and outputs handling the states
	unsigned int m_nStateArr = 0; // number of state arrays - is equal for in- and output
	unsigned int m_nStateValues = 0; // total number of state values
	std::vector<TfLiteTensor*> mp_stateInpTensors; // vector with pointers to state input tensors
	std::vector <const TfLiteTensor*> mp_stateOutTensors; // vector with pointers to state output tensors
	
public:
	InputManagement(bool stateful, double fixInterval, unsigned int nInputEntries);
	~InputManagement();

	bool isActive(); //check if is active

	// Allocation function to handle states
	bool addStateInp(TfLiteTensor* stateInpTensor); // add state input tensor 
	bool addStateOut(const TfLiteTensor* stateInpTensor); // add state output tensor

	// functions handling the stored state values
	unsigned int manageNewStep(double time, bool firstInvoke, double* input); // updates the buffer and returns the required number of calls of the neural net
	bool updateFinishedStep(double time, unsigned int nSteps); // updates the state buffer
	void initialize(); // initialize all states with zeros
	void initialize(double* p_stateValues, const unsigned int& nStateValues); // initialize all states with given values

	double* handleInpts(double time, unsigned int iStep, double* flatInp, bool firstInvoke); // function for interpolation of normal input data

};

