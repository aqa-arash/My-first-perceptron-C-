echo "This script should read a dataset label into a tensor and pretty-print it into a text file..."
#!/bin/bash

# Check that the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <label_dataset_input> <label_tensor_output> <label_index>"
    exit 1
fi

# Read the arguments
label_dataset_input=$1
label_tensor_output=$2
label_index=$3

# Check if the label dataset input file exists
if [ ! -f "$label_dataset_input" ]; then
    echo "Error: File '$label_dataset_input' not found!"
    exit 1
fi

./build/read_dataset $label_dataset_input $label_tensor_output $label_index 0
