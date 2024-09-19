
#include "Utils.h"
#include "InterfaceFunctions.h"
#ifdef _WIN32
#include "windows.h"
#else
#include <dlfcn.h>
#include <climits>
#include <unistd.h>
#endif

int Utils::compareTensorSizes(const TfLiteTensor* A, const TfLiteTensor* B, unsigned int* unmatchedVals,
                              TensorflowDllHandler* p_tfDll)
{
	// used to compare two tensors - return 0 if their sizes are equal - returns -1 if dimensions mismatchs - returns dimension
	// where size do not match
	if (p_tfDll->tensorNumDims(A) != p_tfDll->tensorNumDims(B))
	{
		unmatchedVals[0] = p_tfDll->tensorNumDims(A);
		unmatchedVals[1] = p_tfDll->tensorNumDims(B);
		return -1;
	}
	// check the sizes in each dimension except for the first which is the batch size
	for (int i = 1; i < p_tfDll->tensorNumDims(A); ++i) {
		if (p_tfDll->tensorDim(A, i) != p_tfDll->tensorDim(A, i)) {
			unmatchedVals[0] = p_tfDll->tensorDim(A, i);
			unmatchedVals[1] = p_tfDll->tensorDim(B, i);
			return i;
		}
	}
	return 0;
}

int Utils::getNumElementsTensor(const TfLiteTensor* A, TensorflowDllHandler* p_tfDll)
{
	int nElements = 1;
	int dim = p_tfDll->tensorNumDims(A);
	for (int iDim = 0; iDim < dim; ++iDim) {
		nElements *= p_tfDll->tensorDim(A, iDim);
	}
	return nElements;
}

void Utils::castToFloat(const double& value, void* p_store, unsigned int pos)
{
	// p_stores stores float values
	auto* p_float = (float*)p_store;
	p_float[pos] = (float)value;
}

void Utils::castFromFloat(double& value, void* p_store, unsigned int pos)
{
	// p_stores stores float values
	auto* p_float = (float*)p_store;
	value = p_float[pos];
}

#ifdef _WIN32
std::string Utils::getTensorflowDllPathWin() {
    char path[MAX_PATH];
    HMODULE hm = NULL;

    if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          (LPCSTR) &NeuralNet_createObject, &hm) == 0)
    {
        int ret = GetLastError();
        std::string message = Utils::string_format("SMArtInt: Unable to locate tensorflow dll");
        //mp_modelicaUtilityHelper->ModelicaError(message.c_str());
        throw std::runtime_error(message);
    }
    if (GetModuleFileName(hm, path, sizeof(path)) == 0)
    {
        int ret = GetLastError();
        std::string message = Utils::string_format("SMArtInt: Unable to locate tensorflow dll");
        throw std::runtime_error(message);
    }

    std::string folderPath(path);
    size_t lastSlash = folderPath.find_last_of("\\/");
    if (lastSlash != std::string::npos) {
        folderPath = folderPath.substr(0, lastSlash + 1);
    }
    // Build the new path for tensorflow_c.dll
    return folderPath + "tensorflowlite_c.dll";
}
#else
std::string Utils::getTensorflowDllPathLinux() {
    Dl_info dl_info;

    // Get the address of a symbol in the shared library
    if (dladdr((void*) &NeuralNet_createObject, &dl_info) == 0) {
        std::string message = "SMArtInt: Unable to locate tensorflow shared library";
        throw std::runtime_error(message);
    }

    char path[PATH_MAX];
    ssize_t count = readlink(dl_info.dli_fname, path, PATH_MAX);
    if (count == -1) {
        std::string message = "SMArtInt: Unable to locate tensorflow shared library path";
        throw std::runtime_error(message);
    }

    std::string folderPath(path, count);
    size_t lastSlash = folderPath.find_last_of("\\/");
    if (lastSlash != std::string::npos) {
        folderPath = folderPath.substr(0, lastSlash + 1);
    }
    // Build the new path for tensorflow_c.so
    return folderPath + "tensorflowlite_c.so";
}
#endif
