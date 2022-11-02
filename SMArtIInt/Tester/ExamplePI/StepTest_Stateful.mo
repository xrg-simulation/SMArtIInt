within SMArtIInt.Tester.ExamplePI;
model StepTest_Stateful
  extends StepTest_RNN(redeclare TF_PI_Stateful controller);

  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=36000,
      Interval=1,
      __Dymola_Algorithm="Dassl"));
end StepTest_Stateful;
