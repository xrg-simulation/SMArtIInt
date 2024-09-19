within SMArtIInt.Tester.ExamplePI.TFLite;
model StepTest_Stateful_tflite
  extends TFLite.StepTest_RNN_tflite(redeclare TFLite.TF_PI_Stateful_tflite controller);

  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=36000,
      Interval=1,
      __Dymola_Algorithm="Dassl"));
end StepTest_Stateful_tflite;
