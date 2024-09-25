#pragma once
#include "tensorflow/lite/c/c_api.h"
#include <string>
#include "NNBuffer.h"
#include <vector>
#include "Utils.h"

class InputManagement
{

private:
	bool m_active = false; // true if inputs are managed - for stateful NNs
	double m_fixTimeIntv = 0; // sample interval

	double m_startTime = 0; // time of the first call

	NNBuffer<std::vector<double>, double> mp_inputBuffer = NNBuffer<std::vector<double>, double>(); // buffer for inputs
	NNBuffer<Utils::stateInputsContainer, double> m_stateBuffer = NNBuffer<Utils::stateInputsContainer, double>(); // buffer for states

	// arrays handling the normal function input
	unsigned int m_nInputEntries = 0; // total number of input elements
	double* mp_flatInterpolatedInp = nullptr; // storage for interpolated NN input
	
	// pointer to the inputs and outputs handling the states
	unsigned int m_nStateArr = 0; // number of state arrays - is equal for in- and output
	unsigned int m_nStateValues = 0; // total number of state values
	std::vector<TfLiteTensor*> mp_stateInpTensors; // vector with pointers to state input tensors
	std::vector <const TfLiteTensor*> mp_stateOutTensors; // vector with pointers to state output tensors
public:
    std::vector<Ort::Value*> mp_OnnxStateInpTensors; // vector with pointers to state input tensors
    std::vector <Ort::Value*> mp_OnnxStateOutTensors; // vector with pointers to state output tensors


public:
	InputManagement(bool stateful, double fixInterval, unsigned int nInputEntries, TensorflowDllHandler* p_tfDll);
	~InputManagement();

    TensorflowDllHandler* mp_tfDll;

	[[nodiscard]] bool isActive() const; //check if is active

	// Allocation function to handle states
	bool addStateInp(TfLiteTensor* stateInpTensor); // add state input tensor
    bool addStateInp(Ort::Value*);
	bool addStateOut(const TfLiteTensor* stateInpTensor); // add state output tensor
    bool addStateOut(Ort::Value* stateOutTensor);
    bool updateStateOut(Ort::Value* stateOutTensor);

	// functions handling the stored state values
    void storeInputs(double time, const double* input);
	unsigned int calculateNumberOfSteps(double time, bool firstInvoke); // updates the buffer and returns the required number of calls of the neural net
	bool updateFinishedStep(double time, unsigned int nSteps); // updates the state buffer
	void initialize(double time); // initialize all states with zeros
	void initialize(double time, double* p_stateValues, const unsigned int& nStateValues); // initialize all states with given values

	double* handleInpts(double time, unsigned int iStep, double* flatInp, bool firstInvoke); // function for interpolation of normal input data

};

