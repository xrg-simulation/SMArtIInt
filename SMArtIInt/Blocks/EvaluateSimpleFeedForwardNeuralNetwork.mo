within SMArtIInt.Blocks;
model EvaluateSimpleFeedForwardNeuralNetwork
  extends BaseClasses.BaseFeedForwardNeuralNet;

  Modelica.Blocks.Interfaces.RealInput arrayIn[size(array2DFlatteningModel.arrayIn, 1),size(array2DFlatteningModel.arrayIn, 2)]
    annotation (Placement(transformation(extent={{-108,-10},{-88,10}}), iconTransformation(extent={{-108,-10},{-88,10}})));
  Modelica.Blocks.Interfaces.RealOutput arrayOut[size(array2DDeflatteningModel.arrayOut, 1),size(array2DDeflatteningModel.arrayOut,
    2)]
    annotation (Placement(transformation(extent={{90,-10},{110,10}}), iconTransformation(extent={{90,-10},{110,10}})));
equation
  connect(array2DFlatteningModel.arrayIn, arrayIn) annotation (Line(points={{-42,0},{-98,0}}, color={0,0,127}));
  connect(array2DDeflatteningModel.arrayOut, arrayOut) annotation (Line(points={{41.2,0},{100,0}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)),
    Documentation(info="<html>
<p>This is a specialized version of the EvaluateGenericNeuralNetwork. It can be used for neural networks which use several scalar inputs and outputs. The user has to create the wanted inputs and has to connect them to the input of the block. This input has the same shape [batchSize, numberOfInputs] of the input used in the tensorflow model. The individual input have to be fed into the last dimension. A batch size can be used simultaniously calculation.</p>
<p>The example <a href=\"modelica://SMArtIInt.Tester.PipeHeatTransferExample.PipeLocalHeatTransfer\">PipeLocalHeatTransfer</a> uses this block.</p>
</html>"));
end EvaluateSimpleFeedForwardNeuralNetwork;
