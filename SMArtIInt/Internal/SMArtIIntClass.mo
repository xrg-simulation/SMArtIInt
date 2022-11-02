within SMArtIInt.Internal;
class SMArtIIntClass
  extends ExternalObject;
  extends Modelica.Icons.SourcesPackage;

  function constructor
    input ModelicaUtilityHelper modelicaUtilityHelper;
    input String pathToTfLiteFile "String to file";
    input Integer n_inputDim "Number of dimensions of input array";
    input Integer[n_inputDim] inputSizes "Sizes in each dimensions of input array";
    input Integer n_outputDim;
    input Integer[n_outputDim] outputSizes "Sizes in each dimensions of input array";
    input Boolean stateful = false;
    input Real fixEvalStep = 0;
    output SMArtIIntClass smartiint;
    external "C" smartiint = NeuralNet_createObject(modelicaUtilityHelper, pathToTfLiteFile,
      n_inputDim, inputSizes, n_outputDim, outputSizes,
      stateful, fixEvalStep) annotation (
      Library="SMArtIInt",
      LibraryDirectory="modelica://SMArtIInt/Resources/Library");
  end constructor;

  function destructor
    input SMArtIIntClass smartiint;
  external"C" NeuralNet_destroyObject(smartiint) annotation (
      Library="SMArtIInt",
      LibraryDirectory="modelica://SMArtIInt/Resources/Library");
  end destructor;
end SMArtIIntClass;
