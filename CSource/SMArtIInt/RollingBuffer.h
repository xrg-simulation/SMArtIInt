#pragma once
template<class T, class idT>
class RollingBuffer
{
public:
	explicit RollingBuffer(unsigned int bufferSize);
	~RollingBuffer();

	T* getCurrentValue(); // get the current value in buffer
	T* getPrevValue(); // get the previous value in buffer
	
	idT getCurrentIdx(); // get the current index
	idT getPrevIdx(); // get the previous index 

	void initializeIdx(const idT &bufferIdx, const idT &delta_idx); // initialize index

	T* getElement(unsigned int idx); // get pointer to element with given index

	bool update(const idT& bufferIdx, const int& jumpEstimate, int& updatedEstimate); // update the buffer with given index 

private:
	int m_bufferSize = 0; //Total number of previous elements

	idT* mp_bufferIdx = nullptr; // index of buffer

	T* mp_buffer = nullptr; // values of buffer

	int m_currentIdx = 0; // current buffer position
	int m_prevIdx = m_bufferSize - 1; // position of previous element in buffer

	void incrementCurrent(unsigned int n); // increment current position by n values
	void decrementCurrent(unsigned int n); // decrement current position by n values

	void incrementIndices(); // increment all indices

};
