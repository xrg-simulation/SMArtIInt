
#include "Utils.h"
#include "InterfaceFunctions.h"
#ifdef _WIN32
#include "windows.h"
#else
#include <dlfcn.h>
#include <climits>
#include <unistd.h>
#include <cerrno>     // For errno
#include <cstring>    // For strerror
#include <sys/stat.h> // For lstat
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

int Utils::compareTensorSizes(Ort::Value* A, Ort::Value* B, unsigned int* unmatchedVals)
{
    // used to compare two tensors - return 0 if their sizes are equal - returns -1 if dimensions mismatchs - returns dimension
    // where size do not match
    if (A->GetTensorTypeAndShapeInfo().GetDimensionsCount() != B->GetTensorTypeAndShapeInfo().GetDimensionsCount())
    {
        unmatchedVals[0] = A->GetTensorTypeAndShapeInfo().GetDimensionsCount();
        unmatchedVals[1] = B->GetTensorTypeAndShapeInfo().GetDimensionsCount();
        return -1;
    }
    // check the sizes in each dimension except for the first which is the batch size
    for (int i = 1; i < A->GetTensorTypeAndShapeInfo().GetDimensionsCount(); ++i) {
        if (A->GetTensorTypeAndShapeInfo().GetShape()[i] != B->GetTensorTypeAndShapeInfo().GetShape()[i]) {
            unmatchedVals[0] = A->GetTensorTypeAndShapeInfo().GetShape()[i];
            unmatchedVals[1] = B->GetTensorTypeAndShapeInfo().GetShape()[i];
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
        std::string message = Utils::string_format("SMArtIInt: Unable to locate tensorflow dll");
        //mp_modelicaUtilityHelper->ModelicaError(message.c_str());
        throw std::runtime_error(message);
    }
    if (GetModuleFileName(hm, path, sizeof(path)) == 0)
    {
        int ret = GetLastError();
        std::string message = Utils::string_format("SMArtIInt: Unable to locate tensorflow dll");
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
        std::string message = "SMArtIInt: Unable to locate tensorflow shared library";
        throw std::runtime_error(message);
    }

    // Check if it's a symlink
    struct stat sb;
    if (lstat(dl_info.dli_fname, &sb) == -1) {
        std::string error_message = "lstat failed for: ";
        error_message += dl_info.dli_fname;
        error_message += " - Error: ";
        error_message += strerror(errno);
        throw std::runtime_error(error_message);
    }

    char path[PATH_MAX];
    ssize_t count;
    // If it's not a symbolic link, copy the path directly
    if (!S_ISLNK(sb.st_mode)) {
        std::cout << "Path is not a symbolic link, using the direct path: " << dl_info.dli_fname << std::endl;
        strncpy(path, dl_info.dli_fname, PATH_MAX - 1);
        path[PATH_MAX - 1] = '\0';  // Ensure null termination
        count = strlen(path);
    } else {
        // If it is a symbolic link, resolve it
        count = readlink(dl_info.dli_fname, path, PATH_MAX - 1);
        if (count == -1) {
            std::string error_message = "Failed to readlink for: ";
            error_message += dl_info.dli_fname;
            error_message += " - Error: ";
            error_message += strerror(errno);
            throw std::runtime_error(error_message); // Throw runtime_error with detailed error message
        }
        path[count] = '\0';  // Null-terminate the string
        std::cout << "Resolved symbolic link path: " << path << std::endl;
    }

    std::string folderPath(path, count);
    size_t lastSlash = folderPath.find_last_of("\\/");
    if (lastSlash != std::string::npos) {
        folderPath = folderPath.substr(0, lastSlash + 1);
    }
    // Build the new path for tensorflow_c.so
    return folderPath + "libtensorflowlite_c.so";
}

#ifdef _WIN32
#include <windows.h>
#include <stdio.h>
int Utils::is_debugger_present() {
    return IsDebuggerPresent();
}

void Utils::wait_for_debugger() {
    while (!is_debugger_present()) {
        printf("Waiting for debugger...\n");
        Sleep(1000); // Sleep for a second before checking again
    }
    printf("Debugger detected!\n");
}
#else
#include <cstdio>
#include <sys/ptrace.h>
#include <unistd.h>

int Utils::is_debugger_present() {

    // Attempt to trace the current process
    if (ptrace(PTRACE_TRACEME, 0, 1, 0) == -1) {
        return 1; // Debugger is present
    }
    // Detach if no debugger is detected
    ptrace(PTRACE_DETACH, 0, 1, 0);
    return 0;
}

void Utils::wait_for_debugger() {
    sleep(10);
    while (!is_debugger_present()) {
        printf("Waiting for debugger...\n");
        sleep(1); // Sleep for a second before checking again
    }
    printf("Debugger detected!\n");
}
#endif

#endif
