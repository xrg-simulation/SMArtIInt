within SMArtIInt.Tester.PipeHeatTransferExample;
model PipeLocalHeatTransfer

  parameter Integer batchSize = 1 "number of simultaneous evaluations";

  Modelica.Blocks.Interfaces.RealInput Re[batchSize] annotation (Placement(transformation(extent={{-120,40},{-80,80}})));
  Modelica.Blocks.Interfaces.RealInput Pr[batchSize] annotation (Placement(transformation(extent={{-120,-20},{-80,20}})));
  Modelica.Blocks.Interfaces.RealInput dByL[batchSize] annotation (Placement(transformation(extent={{-120,-80},{-80,-40}})));
  Modelica.Blocks.Interfaces.RealOutput Nu[batchSize] annotation (Placement(transformation(extent={{90,-10},{110,10}})));
  Blocks.EvaluateSimpleFeedForwardNeuralNetwork evalNN(
    pathToTfLiteFile=Modelica.Utilities.Files.loadResource(
        "modelica://SMArtIInt//Resources//ExampleNeuralNets//model_large.tflite"),
    numberOfInputs=3,
    numberOfOutputs=1,
    batchSize=batchSize) annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
equation
  connect(Re, evalNN.arrayIn[:, 1]) annotation (Line(points={{-100,60},{-40,60},{-40,0},{-9.8,0}},  color={0,0,127}));
  connect(Pr, evalNN.arrayIn[:, 2]) annotation (Line(points={{-100,0},{-54,0},{-54,0},{-9.8,0}},
                                                                                  color={0,0,127}));
  connect(dByL, evalNN.arrayIn[:, 3]) annotation (Line(points={{-100,-60},{-40,-60},{-40,0},{-9.8,0}},  color={0,0,127}));
  connect(Nu, evalNN.arrayOut[:, 1]) annotation (Line(points={{100,0},{10,0}},   color={0,0,127}));
end PipeLocalHeatTransfer;
