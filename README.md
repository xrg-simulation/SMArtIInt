# SMArtIInt
The SMArtIINt Library aims to support the usage of different artificial intelligence models (AI) in Modelica simulation tools. Currently, it supports TensorFlow models exported as TFLite models within Dymola. 

The repository contains a compiled version of the interface for usage in windows. __As a starting point you can run the script DymolaStartup.mos in the base folder. It will open the SMArtIInt-Library in Dymola. It contains some ready to run examples (SMartIInt.Tester) which demonstrate the usage.__ The corresponding python files which create the TF-Lite models are located in ExampleNeuralNets.

Currently, only a 64-bit version is available. If the variable Advanced.CompileWith64 is set on its default value 0, Dymola will automatically compile a 64-bit Dymosim.exe after giving a remark in the translate log file. In case Advanced.CompileWith64=2 a 64-bit dymosim.exe is created anyway and in case of Advanced.CompileWith64=1 compilation will fail.

SMArtIInt uses other software - the source code is included as submodule + a compiled version for direct usage_
1. Tensorflow (https://github.com/tensorflow/tensorflow)
* License: https://github.com/tensorflow/tensorflow/blob/master/LICENSE
2. Bazel.exe (https://github.com/bazelbuild/bazel)
* License: https://github.com/bazelbuild/bazel/blob/master/LICENSE
3. ClaRa Delay (https://github.com/xrg-simulation/ClaRaDelay)
* License: https://github.com/xrg-simulation/ClaRaDelay/blob/main/LICENSE
