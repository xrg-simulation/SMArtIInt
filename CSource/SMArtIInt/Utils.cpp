#include "Utils.h"


int Utils::compareTensorSizes(const TfLiteTensor* A, const TfLiteTensor* B, unsigned int* unmatchedVals)
{
	// used to compare two tensors - return 0 if their sizes are equal - returns -1 if dimensions mismatchs - returns dimension
	// where size do not match
	if (TfLiteTensorNumDims(A) != TfLiteTensorNumDims(B))
	{
		unmatchedVals[0] = TfLiteTensorNumDims(A);
		unmatchedVals[1] = TfLiteTensorNumDims(B);
		return -1;
	}
	// check the sizes in each dimension except for the first which is the batch size
	for (int i = 1; i < TfLiteTensorNumDims(A); ++i) {
		if (TfLiteTensorDim(A, i) != TfLiteTensorDim(A, i)) {
			unmatchedVals[0] = TfLiteTensorDim(A, i);
			unmatchedVals[1] = TfLiteTensorDim(B, i);
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

int Utils::getNumElementsTensor(const TfLiteTensor* A)
{
	int nElements = 1;
	int dim = TfLiteTensorNumDims(A);
	for (int iDim = 0; iDim < dim; ++iDim) {
		nElements *= TfLiteTensorDim(A, iDim);
	}
	return nElements;
}

void Utils::castToFloat(const double& value, void* p_store, unsigned int pos)
{
	// p_stores stores float values
	float* p_float = (float*)p_store;
	p_float[pos] = (float)value;

	return;
}

void Utils::castFromFloat(double& value, void* p_store, unsigned int pos)
{
	// p_stores stores float values
	float* p_float = (float*)p_store;
	value = p_float[pos];
}

