within SMArtIInt.Tester.ExamplePI.ONNX;
model StepTest_Stateful_onnx
  extends TFLite.StepTest_RNN_tflite(redeclare ONNX.TF_PI_Stateful_onnx controller);

  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=36000,
      Interval=1,
      __Dymola_Algorithm="Dassl"));
end StepTest_Stateful_onnx;
