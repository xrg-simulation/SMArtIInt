cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0091 NEW)

################################################################################
# Source groups
################################################################################
set(Headerdateien
		${CMAKE_CURRENT_SOURCE_DIR}/../External/tensorflow/tensorflow/lite/c/c_api.h
		${CMAKE_CURRENT_SOURCE_DIR}/../External/onnx/onnxruntime/include/onnxruntime_cxx_api.h
		"InputManagement.h"
		"InputManagementONNX.h"
		"InputManagementTF.h"
		"InterfaceFunctions.h"
		"NeuralNet.h"
		"NeuralNetTF.h"
		"NeuralNetONNX.h"
		"Utils.h"
		"NNBuffer.h"
		"TensorflowDllHandler.h"
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	list(APPEND Headerdateien "TensorflowDllHandlerLinux.h")
else()
	list(APPEND Headerdateien "TensorflowDllHandlerWin.h")
endif()
source_group("Headerdateien" FILES ${Headerdateien})

set(Quelldateien
    "InputManagement.cpp"
	"InputManagementONNX.cpp"
	"InputManagementTF.cpp"
    "InterfaceFunctions.cpp"
    "NeuralNet.cpp"
	"NeuralNetTF.cpp"
	"NeuralNetONNX.cpp"
    "Utils.cpp"
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	list(APPEND Headerdateien "TensorflowDllHandlerLinux.cpp")
else()
	list(APPEND Headerdateien "TensorflowDllHandlerWin.cpp")
endif()
source_group("Quelldateien" FILES ${Quelldateien})

set(ALL_FILES
    ${Headerdateien}
    ${Quelldateien}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} SHARED ${ALL_FILES})


################################################################################
# Dependencies
################################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../External/tensorflow/)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../External/tensorflow/)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	target_link_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}//External")
	target_link_libraries(${PROJECT_NAME} "libonnxruntime_c.so")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=$ORIGIN")

else ()
	set(ADDITIONAL_LIBRARY_DEPENDENCIES "onnxruntime_c")
	target_link_libraries(${PROJECT_NAME} PUBLIC "${ADDITIONAL_LIBRARY_DEPENDENCIES}")
	target_link_directories(${PROJECT_NAME} PUBLIC
			"${CMAKE_CURRENT_SOURCE_DIR}//External"
	)
endif()
