//
// Created by RobertFlesch on 07.10.2024.
//
#include "InputManagement.h"

#ifndef SMARTIINT_INPUTMANAGEMENTONNX_H
#define SMARTIINT_INPUTMANAGEMENTONNX_H


class InputManagementONNX : public  InputManagement {

public:
    InputManagementONNX(bool stateful, double fixInterval, unsigned int nInputEntries);
    void addStateOut(Ort::Value* stateOutTensor);
    bool updateStateOut(Ort::Value* stateOutTensor);
    double* handleInpts(double time, unsigned int iStep, double* flatInp, bool firstInvoke);
    bool addStateInp(Ort::Value* stateInpTensor);
    bool updateFinishedStep(unsigned int nSteps);
    void initialize(double time, double* p_stateValues, const unsigned int &nStateValues) override;
    std::vector<Ort::Value*> mp_OnnxStateInpTensors; // vector with pointers to state input tensors
    std::vector <Ort::Value*> mp_OnnxStateOutTensors; // vector with pointers to state output tensors
};


#endif //SMARTIINT_INPUTMANAGEMENTONNX_H
