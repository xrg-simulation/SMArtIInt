within SMArtIInt.Tester.ExamplePI;
model TF_PI_Stateful_onnx

  Modelica.Blocks.Interfaces.RealInput  u annotation (Placement(transformation(extent={{-126,-20},{-86,20}})));
  Modelica.Blocks.Interfaces.RealOutput y annotation (Placement(transformation(extent={{94,-10},{114,10}})));

  Blocks.EvaluateStatefulRecurrentNeuralNet controller(pathToTfLiteFile=Modelica.Utilities.Files.loadResource("modelica://SMArtIInt/Resources/ExampleNeuralNets/PIController/PI_stateful.onnx"), samplePeriod=10)   annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
equation

  connect(u, controller.u[1, 1]) annotation (Line(points={{-106,0},{-60,0},{-60,0},{-10,0}}, color={0,0,127}));
  connect(controller.y[1, 1], y) annotation (Line(points={{10,0},{104,0}}, color={0,0,127}));
  annotation (Documentation(info="<html>
<p>The model was created with the script <a href=\"modelica://SMArtIInt/Resources/ExampleNeuralNets/PIController/PI.py\">ExampleNeuralNets\\PIController\\PI.py</a> with setting rnn_type = RnnType.EXTSTATE in line 78. This </p>
</html>"), Icon(graphics={                Bitmap(extent={{-100,-100},{100,100}},
          fileName="modelica://SMArtIInt/Resources/Images/Icon_Inference.png")}));
end TF_PI_Stateful_onnx;
