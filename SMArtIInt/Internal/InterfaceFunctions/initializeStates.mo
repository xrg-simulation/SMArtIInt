within SMArtIInt.Internal.InterfaceFunctions;
function initializeStates
  input SMArtIIntClass smartiint;
  input Real time_value;
  input Real[:] flatStateValues;
  external"C" NeuralNet_initializeStates(smartiint, time_value, flatStateValues, size(flatStateValues, 1)) annotation (
    Library={"SMArtIInt"},
    LibraryDirectory="modelica://SMArtIInt/Resources/Library");
end initializeStates;
