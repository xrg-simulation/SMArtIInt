within SMArtIInt.BaseClasses;
partial model BaseNeuralNet

  parameter String pathToTfLiteFile="" "Choose path to tflite file" annotation (Dialog(group="Selected Model"));

  parameter Integer inputDimensions "Number of input dimension" annotation (Dialog(group="Tensor sizing"));
  parameter Integer[inputDimensions] inputSizes "Vector with size of tensor in each dimension" annotation (Dialog(group="Tensor sizing"));
  final parameter Integer nInputElements=product(inputSizes);

  parameter Integer outputDimensions "Number of output dimension" annotation (Dialog(group="Tensor sizing"));
  parameter Integer[outputDimensions] outputSizes "Vector with size of tensor in each dimension" annotation (Dialog(group="Tensor sizing"));
  final parameter Integer nOutputElements=product(outputSizes);

  parameter Boolean stateful=false "Activate state handling for RNN with state in-/outputs" annotation (Dialog(group="RNN Timing Settings"));
  parameter Real samplePeriod=0 "Fixed sample period for RNNs" annotation (Dialog(group="RNN Timing Settings"));

protected
  final parameter SMArtIInt.Internal.ModelicaUtilityHelper modelicaUtilityHelper=SMArtIInt.Internal.ModelicaUtilityHelper();

  final parameter SMArtIInt.Internal.SMArtIIntClass smartiint=SMArtIInt.Internal.SMArtIIntClass(
      modelicaUtilityHelper,
      pathToTfLiteFile,
      inputDimensions,
      inputSizes,
      outputDimensions,
      outputSizes,
      stateful,
      samplePeriod);

  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false), graphics={Rectangle(
          extent={{-100,100},{100,-100}},
          lineColor={28,108,200},
          pattern=LinePattern.None,
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid), Bitmap(extent={{-100,-100},{100,100}},
          fileName="modelica://SMArtIInt/Resources/Images/Icon_Inference.svg")}),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    Documentation(info="<html>
<p>This base class defines the parameter interface for all classes using the TfLite interface. This class does not contain any evaluation call of a TfLite model and therefore it should not be used.</p>
</html>"));
end BaseNeuralNet;
