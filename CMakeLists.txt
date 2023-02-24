cmake_minimum_required(VERSION 3.17.0)

set(PROJECT_NAME TritonCTF)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB SOURCE_FILES
	  src/*.cpp)
file(GLOB HEADER_FILES
	  includes/TritonCTF/*.hpp)

find_package(triton REQUIRED)

find_package(LIEF REQUIRED)

add_library(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(includes/)
target_include_directories(${PROJECT_NAME} PRIVATE ${TRITON_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC triton)
target_link_libraries(${PROJECT_NAME} PUBLIC LIEF::LIEF)
