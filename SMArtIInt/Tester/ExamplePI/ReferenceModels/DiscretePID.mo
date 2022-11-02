within SMArtIInt.Tester.ExamplePI.ReferenceModels;
model DiscretePID
  extends Modelica.Blocks.Interfaces.SISO;
  import gdv = SMArtIInt.Internal.ClaRaDelay.getDelayValuesAtTime;
  import Modelica.Units.SI;

  parameter Real k(unit="1") = 1 "Gain";
  parameter SI.Time T(start=1, min=Modelica.Constants.small) "Time Constant (T>0 required)";

  Modelica.Clocked.RealSignals.Sampler.SampleClocked
                                            sample2
    annotation (Placement(transformation(extent={{-50,30},{-38,42}})));
  Modelica.Clocked.RealSignals.Sampler.Hold
                                   hold1
    annotation (Placement(transformation(extent={{30,30},{42,42}})));
  Modelica.Clocked.RealSignals.NonPeriodic.PI
                                     PI(
    x(fixed=true),
    T=T,
    k=k)   annotation (Placement(transformation(extent={{-2,26},{18,46}})));
Modelica.Clocked.ClockSignals.Clocks.PeriodicRealClock
                                              periodicClock(period=period)
    annotation (Placement(transformation(extent={{-80,-34},{-68,-22}})));
  parameter SI.Time period=0.1 "Period of clock (defined as Real number)";
equation

  connect(PI.y,hold1. u) annotation (Line(
      points={{19,36},{28.8,36}},
      color={0,0,127}));
connect(periodicClock.y,sample2. clock) annotation (Line(
    points={{-67.4,-28},{-44,-28},{-44,28.8}},
    color={175,175,175},
    pattern=LinePattern.Dot,
    thickness=0.5));
  connect(sample2.y, PI.u) annotation (Line(points={{-37.4,36},{-4,36}}, color={0,0,127}));
  connect(sample2.u, u) annotation (Line(points={{-51.2,36},{-82,36},{-82,0},{-120,0}}, color={0,0,127}));
  connect(hold1.y, y) annotation (Line(points={{42.6,36},{68,36},{68,0},{110,0}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)));
end DiscretePID;
