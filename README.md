# InterSleep, a PyTorch Lightning based library for Interpretable Sleep Classifiers

InterSleep is a versatile PyTorch Lightning library tailored for building, training, and explaining EEG sleep classifiers. Whether you're delving into sleep research or developing practical applications, InterSleep provides a seamless platform for your sleep stage classification projects.

## Supported Models and Architectures

InterSleep currently support the [Sequence-to-Sequence](https://iopscience.iop.org/article/10.1088/1361-6579/ac6049/meta)(SS) models framework. 

SS models take as input a sequence of L input sleep epochs and process them using two Encoders, an epoch Encoder and a sequence encoder. The epoch encoder acts as an epoch-wise feature extractor which transforms an input epoch in the input sequence into an epoch-feature vector for
representation. As a result, the input sequence is transformed into a sequence of feature vectors. Then the epochs sequence is processed using a sequence encoder which takes as input L-feature vectors and produce as output L-features vectors using a RNN aproach to represent intra-sequences information. Then the models provide as output L classes.

[![Sequence-to-Sequence framework](img/ss-framework.jpg)](https://iopscience.iop.org/article/10.1088/1361-6579/ac6049/meta)

The supported models in the SS framework are:
- [TinySleepNet](https://github.com/akaraspt/tinysleepnet)


TODO: example to build and train custom models 

## Supported Datasets

- [SleepEDF-20](https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/)
- [SleepEDF-50](https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/)

## Model Interpretability

Understanding the decisions made by your sleep classifier is crucial for both research and application purposes. 

InterSleep leverages the Captum library to provide interpretability, allowing you to explore and interpret the model's decisions in a transparent manner. Captum offers a suite of techniques for feature attribution, helping you uncover the significance of different input features in the classification process.

TODO: example for extract the explanations

## Contributing

We welcome contributions from the community to enhance and expand InterSleep. If you're interested in contributing, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure tests pass.
4. Submit a pull request.
