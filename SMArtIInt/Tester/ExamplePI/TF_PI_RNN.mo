within SMArtIInt.Tester.ExamplePI;
model TF_PI_RNN

  Modelica.Blocks.Interfaces.RealOutput
                              y annotation (Placement(transformation(extent={{94,-10},{114,10}})));

  Blocks.EvaluateRecurrentNeuralNet         evaluateRecurrentNeuralNet(
    pathToTfLiteFile=Modelica.Utilities.Files.loadResource("modelica://SMArtIInt/Resources/ExampleNeuralNets/PI.tflite"),
    final samplePeriod=100,
    final numberOfInputs=1,
    final numberOfOutputs=1,
    final batchSize=1,
    final returnSequences=false,
    useClaRaDelay=true,
    final nHistoricElements=250,
    continuous=true)        annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  Modelica.Blocks.Interfaces.RealInput
                             u annotation (Placement(transformation(extent={{-126,-20},{-86,20}})));

equation

  connect(u, evaluateRecurrentNeuralNet.u[1]) annotation (Line(points={{-106,0},{-10,0}}, color={0,0,127}));
  connect(evaluateRecurrentNeuralNet.y[1], y) annotation (Line(points={{10,0},{104,0}}, color={0,0,127}));
  annotation (Documentation(info="<html>
<p>The model was created with the script <a href=\"modelica://SMArtIInt/Resources/ExampleNeuralNets/PI.py\">ExampleNeuralNets\\PIController\\PI.py</a> with setting rnn_type = RnnType.RNN in line 78.</p>
</html>"), Icon(graphics={                Bitmap(extent={{-100,-100},{100,100}},
          fileName="modelica://SMArtIInt/Resources/Images/Icon_Inference.png")}));
end TF_PI_RNN;
