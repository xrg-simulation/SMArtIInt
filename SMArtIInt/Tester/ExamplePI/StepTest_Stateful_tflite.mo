within SMArtIInt.Tester.ExamplePI;
model StepTest_Stateful_tflite
  extends StepTest_RNN_tflite(redeclare TF_PI_Stateful_tflite controller);

  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=36000,
      Interval=1,
      __Dymola_Algorithm="Dassl"));
end StepTest_Stateful_tflite;
