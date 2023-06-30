within SMArtIInt.Tester.ExamplePI;
model TF_PI_Stateful
  extends BaseClasses.BaseStatefulRecurrentNeuralNet(
    final batchSize=1,
    final numberOfOutputs=1,
    final numberOfInputs=1,
    final samplePeriod=10,
    final pathToTfLiteFile=Modelica.Utilities.Files.loadResource("modelica://SMArtIInt/Resources/ExampleNeuralNets/PI_stateful.tflite"));

  Modelica.Blocks.Interfaces.RealInput  u annotation (Placement(transformation(extent={{-126,-20},{-86,20}})));
  Modelica.Blocks.Interfaces.RealOutput y annotation (Placement(transformation(extent={{94,-10},{114,10}})));

initial equation

  // explicitly initialize states - if not done default value of 0 is used anyway!
  SMArtIInt.Internal.InterfaceFunctions.initializeStates(
    smartiint,
    {0, 0});

equation

  connect(u, array2DFlatteningModel.arrayIn[1, 1]) annotation (Line(points={{-106,0},{-40,0}}, color={0,0,0}));
  connect(array2DDeflatteningModel.arrayOut[1, 1], y) annotation (Line(points={{41.2,0},{104,0}}, color={0,0,0}));
  annotation (Documentation(info="<html>
<p>The model was created with the script ExampleNeuralNets\\PIController\\PI.py with setting rnn_type = RnnType.EXTSTATE in line 78. This </p>
</html>"));
end TF_PI_Stateful;
