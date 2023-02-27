#pragma once
#include <string>
//#include <tchar.h>
#include <iostream>
#include "tensorflow/lite/c/c_api.h"
#include <vector>
#include <memory>


namespace Utils
{
	// function to format messages to modelica
	template<typename ... Args> std::string string_format(const std::string& format, Args ... args)
	{
		int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
		if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
		auto size = static_cast<size_t>(size_s);
		auto buf = std::make_unique<char[]>(size);
		std::snprintf(buf.get(), size, format.c_str(), args ...);
		return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
	};

	int compareTensorSizes(const TfLiteTensor* A, const TfLiteTensor* B, unsigned int* unmatchedVals);

	int getNumElementsTensor(const TfLiteTensor* A);

	// individual casting functions
	void castToFloat(const double& value, void* store, unsigned int pos);

	void castFromFloat(double& value, void* p_store, unsigned int pos);

	class stateInputsContainer
	{
	public:
		~stateInputsContainer() {
			for (auto& stateStorage : m_stateStorage) {
				if (stateStorage) operator delete(stateStorage);
			};
		};

		void addStateInput(TfLiteTensor* stateInpTensor) {
			m_stateDataByteSizes.push_back(TfLiteTensorByteSize(stateInpTensor));
			m_stateStorage.push_back(operator new(TfLiteTensorByteSize(stateInpTensor)));
		}

		void* at(unsigned int i) {
			return m_stateStorage[i];
		}

		size_t byteSizeAt(unsigned int i) {
			return m_stateDataByteSizes[i];
		}

	private:
		std::vector<size_t> m_stateDataByteSizes;
		std::vector<void*> m_stateStorage;
	};
}
