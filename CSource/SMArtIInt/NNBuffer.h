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
    NNBuffer() : head(nullptr), tail(nullptr) {}

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
        return (tail->prev->data);
    } // get the previous value in buffer

	idT getPrevIdx() {
        return (tail->prev->index);
    }

    int size() {
        return nElements;
    }

    void store(const idT& bufferIdx, T *bufferValue) {
        if (tail!= nullptr){
            auto current = tail;
            // here we check if the new Idx is larger or equal to the the latest value. If it is equal or larger we
            // remove the element. In case of equal Idx the new element will replace the previous
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
    unsigned int nElements = 0;

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

