cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

set(CMAKE_SYSTEM_VERSION 10.0 CACHE STRING "" FORCE)

project(SMArtIInt CXX)
message("Specified build type: ${CMAKE_BUILD_TYPE}")
if(NOT DEFINED CMAKE_BUILD_TYPE)
	message("Build type not specified - use release mode")
endif()

################################################################################
# Set target arch type if empty. Visual studio solution generator provides it.
################################################################################
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 64 bits
	if(NOT CMAKE_VS_PLATFORM_NAME)
		set(CMAKE_VS_PLATFORM_NAME "x64")
	endif()
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    # 32 bits
	if(NOT CMAKE_VS_PLATFORM_NAME)
		set(CMAKE_VS_PLATFORM_NAME "x86")
	endif()
endif()

message("${CMAKE_VS_PLATFORM_NAME} architecture in use")

if(NOT ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64"))
    message(WARNING "${CMAKE_VS_PLATFORM_NAME} arch is not supported for tensorflow - only ClaRaDelay will be created!")
endif()

################################################################################
# Global configuration types
################################################################################
set(CMAKE_CONFIGURATION_TYPES
    "Debug"
    "Release"
    CACHE STRING "" FORCE
)

# set the output directory
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	if (CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/linux64/")
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/linux64/")
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/linux64/")
	else ()
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/linux32/")
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/linux32/")
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/linux32/")
	endif()
else ()
	if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/win64/")
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/win64/")
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/win64/")
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/win64/")
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/win64/")
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/win64/")
	elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/win32/")
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/win32/")
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../SMArtIInt/Resources/Library/win32/")
	endif()
	
endif ()


################################################################################
# Sub-projects
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
	add_subdirectory(${PROJECT_SOURCE_DIR}/SMArtIInt)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/../External/ClaRaDelay/CSource "ClaRaDelay")



