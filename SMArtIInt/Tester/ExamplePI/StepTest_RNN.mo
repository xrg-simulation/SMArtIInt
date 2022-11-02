within SMArtIInt.Tester.ExamplePI;
model StepTest_RNN
  extends Modelica.Icons.Example;
  replaceable TF_PI_RNN      controller constrainedby Modelica.Blocks.Interfaces.SISO
    annotation (Placement(transformation(extent={{-10,-10},{10,10}})), choicesAllMatching=true);
  Modelica.Blocks.Sources.Step step(startTime=10)   annotation (Placement(transformation(extent={{-82,-10},{-62,10}})));
equation
  connect(step.y, controller.u) annotation (Line(points={{-61,0},{-10.6,0}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=36000,
      Interval=1,
      __Dymola_Algorithm="Dassl"));
end StepTest_RNN;
