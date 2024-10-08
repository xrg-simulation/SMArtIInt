#include <iostream>
#include <vector>

#pragma once

// Node definition
template<class T, class idT>
class Node {
public:
    idT index;
    T* data;
    Node* next;
    Node* prev;

    Node(idT value1, T* value2) : index(value1), data(value2), next(nullptr), prev(nullptr) {}

    ~Node() {
        delete data;
    }
};

template<class T, class idT>
class NNBuffer
{
public:
    explicit NNBuffer() : head(nullptr), tail(nullptr) {}

    ~NNBuffer() {
        Node<T, idT>* current = head;
        while (current != nullptr) {
            Node<T, idT>* next = current->next;
            delete current;
            current = next;
        }
    }

	T* getCurrentValue() {
        return (tail->data);
    }; // get the current value in buffer

	T* getPrevValue() {
        // generally the buffer cannot distinguish between calculation the calculation prior and after an event if
        // the event occurs at a grid interval, because the calculation looks the same of
        // an iteration call (as the step is not finished) or the subsequent call after a time event
        // as a result the state will hold the value of the calculation after the event and this value is used for
        // the next calculation
        // but a time event always occurs at the beginning of the simulation
        // e.g. if we start at 0 sec, the calculation will be called multiple times at 0 sec. For the first step with
        // t > 0 sec we want to have the state on its initial value - that's the reason we store the value as additional
        // value not in the buffer. If the buffer contains only 1 value (for 0 sec) and we call prev. value we return
        // the initial value
        // this has no impact on the input buffer as the buffer is not used at the initial time and afterward it is
        // called (current value and prev. value) if it already contains 2 values for 0 sec and the t>0 sec.

        if (nElements > 1) {
            return (tail->prev->data);
        } else {
            return &initialValue;
        }
    } // get the previous value in buffer

	idT getPrevIdx() {
        return (tail->prev->index);
    }

    int size() {
        return nElements;
    }

    void initialize(T *bufferValue) {
        initialValue = *bufferValue;
    }

    void createEmptyEntry(const idT& bufferIdx) {
        if (tail == nullptr || tail->index < bufferIdx) {
            append(bufferIdx);
        }
    }

    void store(const idT& bufferIdx, T *bufferValue) {
        if (tail != nullptr){
            auto current = tail;
            // here we check if the new Idx is larger or equal to the latest value. If it is equal or larger we
            // remove the element. In case of equal Idx the new element will replace the previous
            // clion complains that the check of current!=nullptr is not required, but it is! In case removeTail()
            // removes the only element current becomes nullptr and we have to stop removing elements
            while (current!= nullptr && (bufferIdx <= current->index)) {
                removeTail();
                current = tail;
            }
        }
//        // Write bufferIdx value to a text file named after objectName
//        std::ofstream outFile(objectName + ".txt", std::ios::app);
//        if (outFile.is_open()) {
//            if (!found) outFile << "x ";
//            outFile << std::setprecision(std::numeric_limits<double>::digits10 + 1);
//            outFile << bufferIdx << ", " << m_currentIdx << std::endl;
//            outFile.close();
//        }

        append(bufferIdx, bufferValue);
    }

private:
    Node<T, idT>* head;
    Node<T, idT>* tail;
    T initialValue;
    unsigned int nElements = 0;

    void append(idT value1) {
        auto* newNode = new Node<T, idT>(value1, nullptr);
        if (tail == nullptr) {
            head = tail = newNode;
        } else {
            tail->next = newNode;
            newNode->prev = tail;
            tail = newNode;
        }
        nElements ++;
    }

    void append(idT value1, T* value2) {
        auto* newNode = new Node<T, idT>(value1, value2);
        if (tail == nullptr) {
            head = tail = newNode;
        } else {
            tail->next = newNode;
            newNode->prev = tail;
            tail = newNode;
        }
        nElements ++;
    }

    // New method to remove the tail node
    void removeTail() {
        if (tail == nullptr) return; // List is empty
        nElements--;
        Node<T, idT>* oldTail = tail;
        if (tail->prev != nullptr) {
            tail->prev->next = nullptr;
        } else {
            head = nullptr; // The list will be empty after removing the tail
        }
        tail = tail->prev;
        delete oldTail;
    }
};

