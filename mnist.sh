#!/bin/bash

echo "This script should trigger the training and testing of your neural network implementation..."

# write a bash script that will trigger the training and testing of your neural network implementation
# The script should take in the following arguments:
# 1. The path to the configuration file for the neural network

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <path_to_config_file>"
	exit 1
fi

CONFIG_FILE=$1

./build/MnistModel "$CONFIG_FILE"