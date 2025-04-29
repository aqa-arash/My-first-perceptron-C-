// Types.h
#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>

// Define a type alias for precision. Change this to float for single precision.
using Precision = double;

// Define matrix and vector types based on Precision.
using Matrix = Eigen::Matrix<Precision, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Precision, Eigen::Dynamic, 1>;

#endif // TYPES_H