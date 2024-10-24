cmake_minimum_required(VERSION 3.17)

set(CUDA_VERSION 12.6)
if (CMAKE_GENERATOR MATCHES "Visual Studio")
  set(CMAKE_GENERATOR_TOOLSET cuda=${CUDA_VERSION})
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS ON)

project(Test_Project LANGUAGES CXX CUDA)


add_executable(${CMAKE_PROJECT_NAME})

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    main.cu)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: >
)