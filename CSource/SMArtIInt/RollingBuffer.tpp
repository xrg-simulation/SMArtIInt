#include "RollingBuffer.h"
#include <vector>
#include "Utils.h"

template <class T, class idT>
RollingBuffer<T, idT>::RollingBuffer(unsigned int bufferSize)
{
	m_bufferSize = bufferSize;
	mp_bufferIdx = new idT[m_bufferSize];
	mp_buffer = new T[bufferSize];
	m_currentIdx = 0;
	m_prevIdx = m_bufferSize - 1;
}

template <class T, class idT>
RollingBuffer<T, idT>::~RollingBuffer()
{
	delete[] mp_buffer;
	delete[] mp_bufferIdx;
}

template <class T, class idT>
T* RollingBuffer<T, idT>::getElement(unsigned int idx)
{
	return &mp_buffer[idx];
}

template <class T, class idT>
T* RollingBuffer<T, idT>::getCurrentValue()
{
	return &mp_buffer[m_currentIdx];
}

template <class T, class idT>
T* RollingBuffer<T, idT>::getPrevValue()
{
	return &mp_buffer[m_prevIdx];
}

template <class T, class idT>
idT RollingBuffer<T, idT>::getCurrentIdx()
{
	return mp_bufferIdx[m_currentIdx];
}

template <class T, class idT>
idT RollingBuffer<T, idT>::getPrevIdx()
{
	return mp_bufferIdx[m_prevIdx];
}

template <class T, class idT>
void RollingBuffer<T, idT>::initializeIdx(const idT &time, const idT & delta_idx)
{
	for (int i=1; i<m_bufferSize; ++i){
		mp_bufferIdx[i] = time - delta_idx/2;
	}
	mp_bufferIdx[0] = time;
}

template <class T, class idT>
void RollingBuffer<T, idT>::incrementCurrent(unsigned int n)
{
	m_currentIdx = m_currentIdx + n;
	if (m_currentIdx >= m_bufferSize) {
		m_currentIdx = m_currentIdx - m_bufferSize;
	}
}

template <class T, class idT>
void RollingBuffer<T, idT>::decrementCurrent(unsigned int n)
{
	m_currentIdx = m_currentIdx - n;
	if (m_currentIdx < 0) {
		m_currentIdx = m_bufferSize + m_currentIdx;
	}
}

template <class T, class idT>
void RollingBuffer<T, idT>::incrementIndices()
{
	m_currentIdx += 1;
	m_prevIdx += 1;
	if (m_currentIdx >= m_bufferSize) {
		m_currentIdx = 0;
	}
	if (m_prevIdx >= m_bufferSize) {
		m_prevIdx = 0;
	}
}

template <class T, class idT>
bool RollingBuffer<T, idT>::update(const idT& bufferIdx, const int& jumpEstimate, int& updatedEstimate)
{
	bool found = false;
	if (bufferIdx > mp_bufferIdx[m_currentIdx]) {
        // the idx is larger than the current index
		incrementIndices();
		mp_bufferIdx[m_currentIdx] = bufferIdx;
		found = true;
	}
	else {
		// in many cases - the multi step solver will be used which means that the neighbouring
		// decremented element contains the previous accepted step. Therefore, we do not use
		// binary search.
		int sav_currentIdx = m_currentIdx;
		decrementCurrent(jumpEstimate);
		if (mp_bufferIdx[m_currentIdx] == bufferIdx) {
			found = true;
		} else if (bufferIdx > mp_bufferIdx[m_currentIdx]) {
			while (!found) {
				// at this point we cannot 
				incrementCurrent(1);
				if (mp_bufferIdx[m_currentIdx] >= bufferIdx) {
					found = true;
				}
			}
		}
		else {
			// m_timeStore[m_currIdx] > time -> we have to go further back
			int counter = jumpEstimate;
			while (!found) {
				decrementCurrent(1);
				counter += 1;
				if (mp_bufferIdx[m_currentIdx] == bufferIdx) {
					found = true;
				} else {
					if (counter >= m_bufferSize) {
						break;
					}
				}
			}
		}
		updatedEstimate = sav_currentIdx - m_currentIdx;
		if (updatedEstimate<0) updatedEstimate = m_bufferSize + updatedEstimate;
		// update the index to previous result
		m_prevIdx = m_currentIdx - 1;
		if (m_prevIdx < 0) {
			m_prevIdx = m_bufferSize + m_prevIdx;
		}
		mp_bufferIdx[m_currentIdx] = bufferIdx;
	}
	return found;
}

template class RollingBuffer<std::vector<double>, double>;
template class RollingBuffer<Utils::stateInputsContainer, int>;