within SMArtIInt.Tester.PipeHeatTransferExample;
model PipeLocalHeatTransfer_smallNN
  extends BaseClasses.BaseFeedForwardNeuralNet(final numberOfOutputs=1, final numberOfInputs=3,
  pathToTfLiteFile=Modelica.Utilities.Files.loadResource(
        "modelica://SMArtIInt//Resources//ExampleNeuralNets//model_small.tflite"));

  Modelica.Blocks.Interfaces.RealInput Re[batchSize] annotation (Placement(transformation(extent={{-120,40},{-80,80}})));
  Modelica.Blocks.Interfaces.RealInput Pr[batchSize] annotation (Placement(transformation(extent={{-120,-20},{-80,20}})));
  Modelica.Blocks.Interfaces.RealInput dByL[batchSize] annotation (Placement(transformation(extent={{-120,-80},{-80,-40}})));
  Modelica.Blocks.Interfaces.RealOutput Nu[batchSize] annotation (Placement(transformation(extent={{90,-10},{110,10}})));
equation
  connect(Re, array2DFlatteningModel.arrayIn[:, 1])
    annotation (Line(points={{-100,60},{-60,60},{-60,0},{-42,0}}, color={0,0,0}));
  connect(Pr, array2DFlatteningModel.arrayIn[:, 2]) annotation (Line(points={{-100,0},{-42,0}}, color={0,0,0}));
  connect(dByL, array2DFlatteningModel.arrayIn[:, 3])
    annotation (Line(points={{-100,-60},{-60,-60},{-60,0},{-42,0}}, color={0,0,0}));
  connect(array2DDeflatteningModel.arrayOut[:, 1], Nu) annotation (Line(points={{41.2,0},{100,0}}, color={0,0,0}));
end PipeLocalHeatTransfer_smallNN;
