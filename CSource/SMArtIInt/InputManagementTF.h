//
// Created by RobertFlesch on 07.10.2024.
//

#include "InputManagement.h"

#ifndef SMARTIINT_INPUTMANAGEMENTTF_H
#define SMARTIINT_INPUTMANAGEMENTTF_H


class InputManagementTF : public InputManagement {
public:
    InputManagementTF(bool stateful, double fixInterval, unsigned int nInputEntries, TensorflowDllHandler* p_tfDll);
    bool addStateOut(const TfLiteTensor* stateOutTensor);
    double* handleInpts(double time, unsigned int iStep, double* flatInp, bool firstInvoke);
    bool addStateInp(TfLiteTensor* stateInpTensor);
    bool updateFinishedStep(unsigned int nSteps);
    void initialize(double time, double* p_stateValues, const unsigned int &nStateValues) override;
private:
    TensorflowDllHandler* mp_tfDll;
    std::vector<TfLiteTensor*> mp_stateInpTensors; // vector with pointers to state input tensors
    std::vector <const TfLiteTensor*> mp_stateOutTensors; // vector with pointers to state output tensors
};


#endif //SMARTIINT_INPUTMANAGEMENTTF_H
