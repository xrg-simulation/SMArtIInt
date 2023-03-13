within SMArtIInt.Internal.Utilities;
model RunInterferenceRNN

  // general parameters
  parameter Integer nInputs=1 "Number of scalar inputs";
  parameter Integer nOutputs=1 "Number of scalar outputs";
  parameter Modelica.Units.SI.Time samplePeriod=0.1 "sampling period/interval";
  parameter Integer nHistoricElements=10 "Number of elements from sampling steps for each input fed to the neural net";

  parameter Boolean continuous=false;
  parameter Boolean useClaRaDelay = true "Switch between available delay types: Clara and MSL";
  parameter SubModels.RNNFlatteningMethod flatteningMethod=SubModels.RNNFlatteningMethod.OldFIrstInputSeq;
  parameter Boolean returnSequences = false;

  // instance of SMArtIInt class
  parameter Internal.SMArtIIntClass smartiint;

  Modelica.Blocks.Interfaces.RealInput u[nInputs]
    annotation (Placement(transformation(extent={{-120,-20},{-80,20}})));
  Modelica.Blocks.Interfaces.RealOutput y_flat[nOutputs]
    annotation (Placement(transformation(extent={{78,-20},{118,20}})));
  SubModels.RNNFlattenInput flattenedHistory(
    useClaRaDelay=useClaRaDelay,
    nInputs=nInputs,
    samplePeriod=samplePeriod,
    nHistoricElements=nHistoricElements,
    continuous=continuous,
    flatteningMethod=flatteningMethod,
    u=u) annotation (Placement(transformation(extent={{-10,20},{10,40}})));

  SubModels.RNNDeflattenOutput unflattenOutput(
    nOutputs=nOutputs,
    nHistoricElements=nHistoricElements) if returnSequences annotation (Placement(transformation(extent={{-10,-40},{10,-20}})));
equation
  y_flat[:] = InterfaceFunctions.runInferenceFlatTensor(
    smartiint,
    time,
    flattenedHistory.inputFlattenTensor,
    nOutputs);

  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)));
end RunInterferenceRNN;
