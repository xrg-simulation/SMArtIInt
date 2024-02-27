within SMArtIInt.Tester.PipeHeatTransferExample;
model TestModelEvaluation
  extends Modelica.Icons.Example;

  replaceable package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater annotation (choicesAllMatching=true);

  Medium.BaseProperties[n] mediums;

  parameter Integer n=100;
  parameter Modelica.Units.SI.Length ds=10e-3;
  parameter Modelica.Units.SI.Length lengths=100/n;

  final parameter Real acc(unit="m/s2") = 1;
  Modelica.Units.SI.Velocity vs;

  replaceable Tester.PipeHeatTransferExample.NNHeatTransfer heatTransfer constrainedby Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.PartialPipeFlowHeatTransfer(
    redeclare final package Medium = Medium,
    final n=n,
    final nParallel=1,
    final surfaceAreas=fill(ds*Modelica.Constants.pi*lengths, n),
    final lengths=fill(lengths, n),
    final dimensions=fill(ds, n),
    final roughnesses=fill(0.025e-3, n),
    final states=mediums.state,
    final vs=fill(vs, n),
    final use_k=false) annotation (choicesAllMatching=true);

equation
  mediums.T = fill(300, n);
  mediums.p = fill(1e5, n);
  vs = acc*time;

  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=100,
      __Dymola_NumberOfIntervals=5000,
      __Dymola_Algorithm="Dassl"));
end TestModelEvaluation;
