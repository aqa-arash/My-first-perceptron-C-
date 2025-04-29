//
// Created by Andi on 20/01/2025.
//

#ifndef PERCEPTRON_LOOPLOGGER_H
#define PERCEPTRON_LOOPLOGGER_H


#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <string>

class LoopLogger {
public:
    // Constructor that takes the maximum number of iterations
    LoopLogger(int maxIterations);

    // Destructor to clean up thread
    ~LoopLogger();

    // Method to update progress with the current iteration
    void updateProgress(int iteration, double error);

    void waitForCompletion();

private:
    // Logging method that runs in a separate thread
    void log();

    // Method to format the elapsed time
    std::string formatDuration(double seconds);

    // Member variables
    std::thread logThread;                        // Thread for logging
    std::atomic<bool> running;                    // Control flag for running status
    int lastIteration;                            // Last completed iteration
    std::atomic<int> currentIteration;            // Current iteration count
    std::atomic<double> currentError;             // Current error value>
    int maxIterations;                            // Total number of iterations
    std::chrono::time_point<std::chrono::steady_clock> startTime; // Start time of logging
};


#endif //PERCEPTRON_LOOPLOGGER_H
