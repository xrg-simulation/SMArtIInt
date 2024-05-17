#include "InputManagement.h"
#include <iostream>
#include "tensorflow/lite/c/c_api.h"
#include "Utils.h"
#include <vector>
#include <stdexcept>
#include <cstring>

void print_tensor_data1(const Ort::Value& value) {
    // Überprüfe, ob der Wert ein Tensor ist
    if (value.IsTensor()) {
        // Zugriff auf die Form des Tensors
        auto tensor_info = value.GetTensorTypeAndShapeInfo();
        auto tensor_shape = tensor_info.GetShape();

        // Überprüfe, ob der Tensor vom Typ float ist
        if (tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            // Zugriff auf die Daten des Tensors als float
            const float* tensor_data = value.GetTensorData<float>();

            // Ausgabe der Daten des Tensors
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

bool InputManagement::addStateInp(Ort::Value* stateInpTensor)
{
    m_nStateArr += 1;
    std::cout << "State Test" <<std::endl;
    for (unsigned int i = 0; i < m_nStoredSteps; ++i) {
        m_stateBuffer.getElement(i)->addStateInput(stateInpTensor);
    }
    mp_OnnxStateInpTensors.push_back(stateInpTensor);
    m_nStateValues += stateInpTensor->GetTensorTypeAndShapeInfo().GetElementCount();
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

bool InputManagement::addStateOut(Ort::Value* stateOutTensor)
{
    size_t i = mp_stateOutTensors.size();
    if (i < m_nStateArr) {
        mp_OnnxStateOutTensors.push_back(stateOutTensor);
//        unsigned int unmatchedVals[2];
//        int ret = Utils::compareTensorSizes(mp_stateInpTensors[i], mp_stateOutTensors[i], unmatchedVals);
//        if (ret < 0) {
//            throw std::invalid_argument(Utils::string_format("Unmatched number of dimension for state input and output # %i"
//                                                             " (Input has %i dimensions whereas output has %i dimensions)!", i, unmatchedVals[0], unmatchedVals[1]));
//            return false;
//        }
//        else if (ret > 0) {
//            throw std::invalid_argument(Utils::string_format("Unmatched number of sizes for state input and output # %i in dimension %i "
//                                                             "(Input has %i entries whereas output has %i entries)!"
//                    , i, ret, unmatchedVals[0], unmatchedVals[1]));
//            return false;
//        }
    }
    else {
        // Error
        throw std::invalid_argument(Utils::string_format("SMArtInt can only handle states in stateful=True if state inputs and state outputs are matching!"));
        return false;
    }
    //ToDo check type (and sizes??)
    return true;
}

bool InputManagement::updateStateOut(Ort::Value* stateOutTensor)
{
    mp_OnnxStateOutTensors.push_back(stateOutTensor);
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
                if (size(mp_stateInpTensors) > 0) {
                    std::memcpy(TfLiteTensorData(mp_stateInpTensors[i]), stateInputs->at(i),
                                stateInputs->byteSizeAt(i));
                }
                else if (size(mp_OnnxStateInpTensors) > 0){
                    std::memcpy(mp_OnnxStateInpTensors[i]->GetTensorMutableRawData(), stateInputs->at(i), stateInputs->byteSizeAt(i));
//                    std::cout << "Test 4 how often " << i << std::endl;
//                    print_tensor_data1(*mp_OnnxStateInpTensors[i]);
//                    std::cout << "Test 5 how often " << i << std::endl;
                }
            }
		}
		else {
			// copy state output to input
			for (unsigned int i = 0; i < m_nStateArr; ++i) {
                if (size(mp_stateInpTensors) > 0) {
                    std::memcpy(TfLiteTensorData(mp_stateInpTensors[i]), TfLiteTensorData(mp_stateOutTensors[i]),
                                TfLiteTensorByteSize(mp_stateOutTensors[i]));
                }
                else if (size(mp_OnnxStateInpTensors) > 0) {
                    std::cout << "\nTest Input pre:" << std::endl;
                    print_tensor_data1(*mp_OnnxStateInpTensors[i]);
                    print_tensor_data1(*mp_OnnxStateOutTensors[i]);
                    std::memcpy(mp_OnnxStateInpTensors[i]->GetTensorMutableRawData(), mp_OnnxStateOutTensors[i]->GetTensorMutableRawData(),
                                sizeof(mp_OnnxStateOutTensors[i]->GetTensorTypeAndShapeInfo().GetElementType()) * mp_OnnxStateOutTensors[i]->GetTensorTypeAndShapeInfo().GetElementCount());
                    std::cout << "\nTest Input post:" << std::endl;
                    print_tensor_data1(*mp_OnnxStateInpTensors[i]);
                    print_tensor_data1(*mp_OnnxStateOutTensors[i]);
                    std::cout << "\nTest Finished\n" << std::endl;
                }
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
            if (mp_stateOutTensors.size() > 0) {
                std::memcpy(m_stateBuffer.getCurrentValue()->at(i), TfLiteTensorData(mp_stateOutTensors[i]),
                            m_stateBuffer.getCurrentValue()->byteSizeAt(i));
            }
            else if (mp_OnnxStateOutTensors.size() > 0){
                auto test = mp_OnnxStateOutTensors[i]->GetTensorTypeAndShapeInfo();
                std::memcpy(m_stateBuffer.getCurrentValue()->at(i), mp_OnnxStateOutTensors[i]->GetTensorMutableRawData(),
                            m_stateBuffer.getCurrentValue()->byteSizeAt(i));
//                std::cout << "Test 2 how often " << i << std::endl;
//
//                auto test1 = m_stateBuffer.getCurrentValue()->at(i);
//                int x = 1;
//                Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//                size_t numElements = 40;
//                std::vector<float> data(numElements); // Erstelle einen Vektor für float-Daten
//                // Fülle den Vektor mit den gewünschten Werten
//                for (size_t i = 0; i < numElements; ++i) {
//                    data[i] = i * 1.5f; // Beispielwerte, du kannst sie nach Bedarf ändern
//                }
//                auto type = mp_OnnxStateOutTensors[0]->GetTensorTypeAndShapeInfo().GetShape();
//                Ort::Value value = Ort::Value::CreateTensor<float>(memoryInfo, data.data(), data.size(), type.data(), type.size());
//                std::memcpy(value.GetTensorMutableRawData(), m_stateBuffer.getCurrentValue()->at(i),
//                            m_stateBuffer.getCurrentValue()->byteSizeAt(i));
//                print_tensor_data1(value);
//                std::cout << "Test 3 how often " << i << std::endl;


            }
		}
	}
	return true;
}

void InputManagement::initialize()
{
	for (unsigned int iInput = 0; iInput < m_nStateArr; ++iInput) {
		// the initialization will be done with m_currIdx = 0 and m_prvIdx = m_nStoredSteps - 1
		// therefore we store the data in the last available index


        if (size(mp_stateInpTensors) > 0) {

            void (*castFunc)(const double&, void*, unsigned int);
            switch (TfLiteTensorType(mp_stateInpTensors[iInput])) {
                case kTfLiteFloat32:
                    castFunc = &Utils::castToFloat;
                    break;
                default:
                    throw std::invalid_argument(
                            "Could not convert state data - SMArtIInt currently only supports TFLite models using floats)!");
                    break;
            }

            void* p_data = m_stateBuffer.getPrevValue()->at(iInput);

            unsigned int n = Utils::getNumElementsTensor(mp_stateInpTensors[iInput]);

            for (unsigned int i = 0; i < n; ++i) {
                castFunc(0.0, p_data, i);
            }
        }
        else if (size(mp_OnnxStateInpTensors) > 0) {
            void (*castFunc)(const double &, void *, unsigned int);
            switch (mp_OnnxStateInpTensors[iInput]->GetTensorTypeAndShapeInfo().GetElementType()) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    castFunc = &Utils::castToFloat;
                    break;
                default:
                    throw std::invalid_argument(
                            "Could not convert state data - SMArtIInt currently only supports ONNX models using floats)!");
                    break;
            }

            void *p_data = m_stateBuffer.getPrevValue()->at(iInput);

            unsigned int n = mp_OnnxStateInpTensors[iInput]->GetTensorTypeAndShapeInfo().GetElementCount();

            for (unsigned int i = 0; i < n; ++i) {
                castFunc(0.0, p_data, i);
            }
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

