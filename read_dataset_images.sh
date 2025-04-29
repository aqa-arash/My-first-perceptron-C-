echo "This script should read a dataset image into a tensor and pretty-print it into a text file..."
#!/bin/bash

# Check that the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <image_dataset_input> <image_tensor_output> <image_index>"
    exit 1
fi

# Read the arguments
image_dataset_input=$1
image_tensor_output=$2
image_index=$3

# Check if the image dataset input file exists
if [ ! -f "$image_dataset_input" ]; then
    echo "Error: File '$image_dataset_input' not found!"
    exit 1
fi

./build/read_dataset $image_dataset_input $image_tensor_output $image_index 1
