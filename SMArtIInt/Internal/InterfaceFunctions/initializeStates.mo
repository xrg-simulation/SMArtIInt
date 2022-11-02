within SMArtIInt.Internal.InterfaceFunctions;
function initializeStates
  input SMArtIIntClass smartiint;
  input Real[:] flatStateValues;
  external"C" NeuralNet_initializeStates(smartiint, flatStateValues, size(flatStateValues, 1)) annotation (
    Library="SMArtIInt",
    LibraryDirectory="modelica://SMArtIInt/Resources/Library");
end initializeStates;
