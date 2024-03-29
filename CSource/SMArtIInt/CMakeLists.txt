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

# specific settings for MSVC
if(MSVC)
	use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
	set(ROOT_NAMESPACE SMArtIInt)

	set_target_properties(${PROJECT_NAME} PROPERTIES
		VS_GLOBAL_KEYWORD "Win32Proj"
	)

	if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
		set_target_properties(${PROJECT_NAME} PROPERTIES
			INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
		)
	elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
		set_target_properties(${PROJECT_NAME} PROPERTIES
			INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
		)
	endif()

	################################################################################
	# Compile definitions
	################################################################################
	if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
		target_compile_definitions(${PROJECT_NAME} PRIVATE
			"$<$<CONFIG:Debug>:"
				"_DEBUG"
			">"
			"$<$<CONFIG:Release>:"
				"NDEBUG"
			">"
			"_LIB;"
			"UNICODE;"
			"_UNICODE"
		)
	elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
		target_compile_definitions(${PROJECT_NAME} PRIVATE
			"$<$<CONFIG:Debug>:"
				"_DEBUG"
			">"
			"$<$<CONFIG:Release>:"
				"NDEBUG"
			">"
			"WIN32;"
			"_LIB;"
			"UNICODE;"
			"_UNICODE"
		)
	endif()

	################################################################################
	# Compile and link options
	################################################################################

	set_property(TARGET ${PROJECT_NAME} PROPERTY
				 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /Oi;
                /Gy
            >
            /permissive-;
            /sdl;
            /W3;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /Oi;
                /Gy
            >
            /permissive-;
            /sdl;
            /W3;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /INCREMENTAL
            >
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF;
                /INCREMENTAL:NO
            >
            /DEBUG
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
        )
    endif()
endif()

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
