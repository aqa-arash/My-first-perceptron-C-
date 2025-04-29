cmake_minimum_required(VERSION 3.22.1)
project(Perceptron LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_Fortran_COMPILER "")  # Disable Fortran checks (from Eigen)
# Set the optimization flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_BUILD_TYPE Release)

# FetchContent module to automatically download Eigen if not present
include(FetchContent)

FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)

# Make Eigen available
FetchContent_MakeAvailable(Eigen3)


# -----------------------------------IO------------------------------------------------
add_executable(read_dataset
        read_dataset.cpp
        DataHandling.h)

# Link Eigen to the project
target_link_libraries(read_dataset PRIVATE Eigen3::Eigen)

# Message to indicate completion
message(STATUS "CMake setup complete for read_dataset")


# -----------------------------------Model------------------------------------------------
add_executable(MnistModel
        MnistModel.cpp
        DataHandling.h
        Network/Components.h
        LoopLogger.cpp
        LoopLogger.h)

# Link Eigen to the project
target_link_libraries(MnistModel PRIVATE Eigen3::Eigen)

# Message to indicate completion
message(STATUS "CMake setup complete for MnistModel")