within SMArtIInt.Tester.PipeHeatTransferExample;
model TestPipe
  extends Modelica.Icons.Example;

  replaceable package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater annotation (choicesAllMatching=true);

  Modelica.Fluid.Pipes.DynamicPipe pipe(redeclare package Medium = Medium,
    length=10,
    diameter=10e-3,
    nNodes=100,
    modelStructure=Modelica.Fluid.Types.ModelStructure.av_b,
    use_HeatTransfer=true,
    redeclare replaceable model HeatTransfer = Tester.PipeHeatTransferExample.NNHeatTransfer)
                                                                           annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  Modelica.Fluid.Sources.Boundary_pT sink(redeclare package Medium = Medium, nPorts=1)
                                                                             annotation (Placement(transformation(extent={{68,-10},{48,10}})));
  Modelica.Fluid.Sources.MassFlowSource_T source(redeclare package Medium = Medium,
    m_flow=1,                                                                       nPorts=1) annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
  inner Modelica.Fluid.System system annotation (Placement(transformation(extent={{-78,60},{-58,80}})));

  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow fixedHeatFlow[pipe.n](each Q_flow=1000) annotation (Placement(transformation(extent={{-36,22},{-16,42}})));
equation
  connect(source.ports[1], pipe.port_a) annotation (Line(points={{-60,0},{-10,0}}, color={0,127,255}));
  connect(fixedHeatFlow.port, pipe.heatPorts) annotation (Line(points={{-16,32},{0.1,32},{0.1,4.4}}, color={191,0,0}));
  connect(sink.ports[1], pipe.port_b) annotation (Line(points={{48,0},{10,0}}, color={0,127,255}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)));
end TestPipe;
