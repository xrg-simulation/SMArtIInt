# SMArtIInt
The SMArtIINt Library aims to support the usage of different artificial intelligence models (AI) in Modelica simulation tools. Currently, it supports TensorFlow models exported as TFLite models within Dymola. 

The repository contains a compiled version of the interface for usage in windows. __As a starting point you can run the script DymolaStartup.mos in the base folder. It will open the SMArtIInt-Library in Dymola. It contains some ready to run examples (SMartIInt.Tester) which demonstrate the usage.__ The corresponding python files which create the TF-Lite models are located in ExampleNeuralNets.

SMArtIInt uses other software - the source code is included as submodule + a compiled version for direct usage_
1. Tensorflow (https://github.com/tensorflow/tensorflow)
* License: https://github.com/tensorflow/tensorflow/blob/master/LICENSE
2. Bazel.exe (https://github.com/bazelbuild/bazel)
* License: https://github.com/bazelbuild/bazel/blob/master/LICENSE
3. ClaRa Delay (https://github.com/xrg-simulation/ClaRaDelay)
* License: https://github.com/xrg-simulation/ClaRaDelay/blob/main/LICENSE
