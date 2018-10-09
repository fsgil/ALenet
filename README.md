# ALenet
The idea is to create a simple model of a Lenet and train it on the cifar dataset

Full Description:

Construct and train a CNN based on the LeNet5 architecture1 (3 fully connected layers and 2 fully connected layers) using CIFAR10 as the dataset.

- Use receptive fields of size 5x5 for all their convolutional layers.

- Use ReLUs or ELUs as non linear activations (instead of sigmoids)



Other training details:

SGD optimization with learning rate of 0.001 and momentum of 0.9

Train for 10 epochs.

Note: if you find training parameters that yield a better training, feel free to use those.



Once the network has finish training, collect all the images that have been wrongly classified and use only those to train a second CNN with the same architecture as the first one for 10 epochs as well.

Report training and validation curves for all 10 epochs and for both CNNs.



The task should be completed using Keras (with TensorFlow as back-end) and once you're done, please include a report with the following:
