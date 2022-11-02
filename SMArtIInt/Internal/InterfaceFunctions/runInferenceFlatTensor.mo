within SMArtIInt.Internal.InterfaceFunctions;
function runInferenceFlatTensor
  input SMArtIIntClass smartiint;
  input Real currentTime;
  input Real[:] flatInputTensor;
  input Integer n_out;
  output Real[n_out] flatOutputTensor;
  external"C" NeuralNet_runInferenceFlatTensor(smartiint, currentTime, flatInputTensor, size(flatInputTensor, 1), flatOutputTensor, n_out) annotation (
    Library="SMArtIInt",
    LibraryDirectory="modelica://SMArtIInt/Resources/Library");
end runInferenceFlatTensor;
