# AdvPT Project WS 2023: Handwriting Recognition (MNIST)

This project template contains some files to get your project started.

As described in the project sheet, your task is to develop a fully-connected neural network in C++ for the recognition
of the handwritten digits found in the MNIST dataset.
We test your implementation by running the following shell scripts with our own (and not publicly available) datasets:

IMPLEMENTED * `build.sh`: Shall contain all necessary code to prepare/build your executable(s)

IMPLEMENTED * `read_dataset_images.sh`: Checks if you can successfully read in, convert and write an MNIST dataset image to file.

IMPLEMENTED * `read_dataset_labels.sh`: Checks if you can successfully read in and write an MNIST dataset label to file.

* `mnist.sh`: Triggers the training and testing of your neural network implementation.

Please note that these scripts are responsible for testing different aspects of  implementation and, thus, expect
different arguments.
You can find a more detailed description in the assignment sheet.

**The shell scripts must be at the top level of your repository! So do not move them somewhere else.**

Obviously you can also modify this README file to document your project.

Besides the shell scripts, we also have the following directories:

* `pytorch/`:
  Reference code and playground in Python to demonstrate the program flow of a fully-connected neural network.
  The results (e.g. the development of the loss) obtained by the Python script **shall not be seen as a ground truth**
  but should provide intuition how a NN of our topology should behave, e.g. what order of magnitude for the prediction
  accuracy can be achieved.
* `expected-results/`:
  Set of reference solutions used by the CI pipeline.
  Inspect the `.gitlab-ci.yml` to see which `expected-results` file belongs to which test.
* `mnist-datasets/`:
  Binary files containing image and label data of the MNIST dataset for training/testing the neural network.
* `mnist-configs/`:
  Configuration files passed to the `mnist.sh` script that provide input arguments to steer the program flow (e.g.
  hyperparameters) of the neural network.
* `src/tensor.hpp`:
  Reference solution for the tensor assignment. We recommend you to use this implementation as central datastructure for
  image/label data, the weights and biases for your network, etc. Keep in mind that this implementation is **slow** and
  potentially needs improvements to overcome the time limits of the evaluation.
* `src/matvec.hpp`: Reference solution for a matrix-vector multiplication using the tensor class. Can be also a
  potential target for optimizations.


