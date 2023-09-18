cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0091 NEW)

################################################################################
# Source groups
################################################################################
set(Headerdateien
    ${CMAKE_CURRENT_SOURCE_DIR}/../External/tensorflow/tensorflow/lite/c/c_api.h
    "InputManagement.h"
    "InterfaceFunctions.h"
    "NeuralNet.h"
    "Utils.h"
    "RollingBuffer.h"
)
source_group("Headerdateien" FILES ${Headerdateien})

set(Quelldateien
    "InputManagement.cpp"
    "InterfaceFunctions.cpp"
    "NeuralNet.cpp"
    "Utils.cpp"
    "RollingBuffer.cpp"
)
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
	target_link_libraries(${PROJECT_NAME} "libtensorflowlite_c.so")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=$ORIGIN")
else ()
	set(ADDITIONAL_LIBRARY_DEPENDENCIES "tensorflowlite_c")
	target_link_libraries(${PROJECT_NAME} PUBLIC "${ADDITIONAL_LIBRARY_DEPENDENCIES}")
	target_link_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}//External"
    )
endif()
