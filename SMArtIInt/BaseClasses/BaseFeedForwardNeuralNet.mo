within SMArtIInt.BaseClasses;
model BaseFeedForwardNeuralNet
  extends BaseGenericNeuralNet(
    final stateful=false,
    final outputSizes={batchSize, numberOfOutputs},
    final outputDimensions=2,
    final inputSizes={batchSize, numberOfInputs},
    final inputDimensions=2,
    final samplePeriod=0);

  parameter Integer numberOfInputs "Number of Real Inputs";
  parameter Integer numberOfOutputs "Number of Real Outputs";

  parameter Integer batchSize=1 "Number of parallel batched simulations";

  Internal.Utilities.SubModels.Array2DFlatteningModel array2DFlatteningModel(final numberOfInputs=numberOfInputs, final batchSize=batchSize) annotation (Placement(transformation(extent={{-42,-10},{-22,10}})));
  Internal.Utilities.SubModels.Array2DDeflatteningModel array2DDeflatteningModel(final numberOfOutput=numberOfOutputs,
      final batchSize=batchSize)                                                                                                      annotation (Placement(transformation(extent={{20,-10},{40,10}})));
equation
  connect(array2DFlatteningModel.flatArray, runInference.u)
    annotation (Line(points={{-22,0},{-9.8,0}}, color={0,0,127}));
  connect(runInference.y, array2DDeflatteningModel.flatArray) annotation (Line(points={{10,0},{20.2,0}}, color={0,0,127}));
  annotation (Documentation(info="<html>
<p>This is a specialized version of the BaseGenericNeuralNet. It can be used for neural networks which use several scalar inputs and outputs. The user has to create the wanted inputs and has to connect them to the input of the block array2DFlatteningModel. This input has the same shape [batchSize, numberOfInputs] of the input used in the tensorflow model. The individual input have to be fed into the last dimension. A batch size can be used simultaneously calculation.</p>
<p>The example <a href=\"modelica://SMArtIInt.Tester.PipeHeatTransferExample.PipeLocalHeatTransfer_smallNN\">PipeLocalHeatTransfer_smallNN </a>extends this model.</p>
</html>"));
end BaseFeedForwardNeuralNet;
