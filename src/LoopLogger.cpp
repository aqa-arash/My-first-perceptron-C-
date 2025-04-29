#include "LoopLogger.h"

// Constructor
LoopLogger::LoopLogger(int maxIterations)
        : maxIterations(maxIterations), running(true), currentIteration(0), lastIteration(0) {
    startTime = std::chrono::steady_clock::now(); // Initialize start time
    logThread = std::thread(&LoopLogger::log, this);  // Start logging thread
}

// Destructor
LoopLogger::~LoopLogger() {
    running = false; // Stop the logging
    if (logThread.joinable()) {
        logThread.join(); // Wait for the logging thread to finish
    }
}

// Update function for progress
void LoopLogger::updateProgress(int iteration, double error) {
    currentIteration.store(iteration); // Update current iteration
    currentError.store(error); // Update current error
}

// Logging function that runs in a separate thread
void LoopLogger::log() {
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Sleep for a while
        if(lastIteration == currentIteration.load()) continue;
        auto now = std::chrono::steady_clock::now(); // Get current time
        std::chrono::duration<double> elapsedTime = now - startTime; // Calculate elapsed time

        // Calculate progress and estimated times
        double progress = static_cast<double>(currentIteration) / maxIterations;
        int remainingIterations = maxIterations - currentIteration;
        double estimatedTotalTime = elapsedTime.count() / progress;
        double remainingTime = estimatedTotalTime - elapsedTime.count();

        // Print formatted log output
        std::cout << "\rProgress: "
                  << currentIteration << "/" << maxIterations << " ("
                  << std::fixed << std::setprecision(2) << (progress * 100) << "%) "
                  << "Error: " << std::fixed << std::setprecision(6) << currentError.load() << " "
                  << "Time Elapsed: " << formatDuration(elapsedTime.count()) << ", "
                  << "Estimated Remaining Time: " << formatDuration(remainingTime) << std::flush;
        if (currentIteration == maxIterations) {
            running = false; // Stop logging if all iterations are completed
        }
        lastIteration = currentIteration.load();
    }
    std::cout << "\nLogging stopped.\n"; // Indicate logging stop
}

// Function to format duration into a readable string
std::string LoopLogger::formatDuration(double seconds) {
    int hours = static_cast<int>(seconds) / 3600;
    seconds -= hours * 3600;
    int minutes = static_cast<int>(seconds) / 60;
    seconds -= minutes * 60;
    return std::to_string(hours) + "h " + std::to_string(minutes) + "m " + std::to_string(static_cast<int>(seconds)) + "s";
}

void LoopLogger::waitForCompletion() {
    if (logThread.joinable()) {
        logThread.join(); // Wait for the logging thread to finish
    }
}