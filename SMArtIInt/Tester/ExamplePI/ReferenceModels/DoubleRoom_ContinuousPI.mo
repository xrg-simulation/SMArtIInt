within SMArtIInt.Tester.ExamplePI.ReferenceModels;
model DoubleRoom_ContinuousPI
    extends Modelica.Icons.Example;

    parameter Modelica.Units.SI.Time timeScale=30000 "Time scale of first table column";

    replaceable package Medium = Modelica.Media.Air.DryAirNasa annotation (choicesAllMatching=true);

  Modelica.Fluid.Sources.MassFlowSource_T boundary(
    redeclare package Medium = Medium,
    use_m_flow_in=true,
    m_flow=50/3600,
    T=293.15,
    nPorts=1) annotation (Placement(transformation(extent={{-70,-10},{-50,10}})));

  Modelica.Fluid.Vessels.ClosedVolume room_1(
    redeclare package Medium = Medium,
    energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    T_start=293.15,
    use_portsData=false,
    use_HeatTransfer=true,
    redeclare model HeatTransfer =
        Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.ConstantHeatTransfer (                           alpha0=100),
    V=10,
    nPorts=2) annotation (Placement(transformation(extent={{-38,10},{-18,30}})));
  Modelica.Fluid.Vessels.ClosedVolume room_2(
    redeclare package Medium = Medium,
    energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    T_start=293.15,
    use_portsData=false,
    use_HeatTransfer=true,
    redeclare model HeatTransfer = Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.ConstantHeatTransfer (alpha0=100),
    V=10,
    nPorts=2) annotation (Placement(transformation(extent={{10,6},{30,26}})));
  Modelica.Fluid.Fittings.GenericResistances.VolumeFlowRate volumeFlowRate(
    redeclare package Medium = Medium,
    a=0,
    b=1000/10) annotation (Placement(transformation(extent={{-20,-10},{0,10}})));
  Modelica.Fluid.Sources.Boundary_pT sink(
    redeclare package Medium = Medium,
    T=293.15,
    nPorts=1) annotation (Placement(transformation(extent={{102,-10},{82,10}})));
  Modelica.Fluid.Fittings.GenericResistances.VolumeFlowRate volumeFlowRate1(
    redeclare package Medium = Medium,
    a=0,
    b=1000/10) annotation (Placement(transformation(extent={{30,-10},{50,10}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C=10000, T(
      displayUnit="K",
      start=280,
      fixed=true))                                                                                  annotation (Placement(transformation(extent={{-6,72},
            {14,92}})));
  Modelica.Thermal.HeatTransfer.Components.ConvectiveResistor convectiveResistor annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=270,
        origin={4,50})));
  Modelica.Blocks.Sources.RealExpression expr_convectResistance(y=0.006) annotation (Placement(transformation(extent={{74,28},{54,48}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow controlHeatFlow annotation (Placement(transformation(extent={{-72,42},{-52,62}})));
  replaceable Modelica.Blocks.Continuous.PI controller(
    k=30,
    T=1600,
    initType=Modelica.Blocks.Types.Init.InitialState,
    y_start=100)                                      constrainedby Modelica.Blocks.Interfaces.SISO annotation (Placement(transformation(extent={{18,-50},{-2,-30}})), choicesAllMatching=true);
  Modelica.Fluid.Sensors.TemperatureTwoPort temperature(redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{56,10},{76,-10}})));
  Modelica.Blocks.Sources.CombiTimeTable combiTimeTable(
    table=[0.0,293.15,50/3600; 1,303.15,50/3600; 2,308.15,50/3600; 3,323.15,50/3600; 4,293.15,50/3600; 5,308.15,25/3600; 6,323.15,25/3600; 7,323.15,25/3600; 8,293.15,25/3600; 9,293.15,25/3600],                                                                                         smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
    timeScale=timeScale)                                                                                                                                                                                annotation (Placement(transformation(extent={{96,-74},{76,-54}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature
                                                           lossHeatFlow annotation (Placement(transformation(extent={{70,62},{50,82}})));
  Modelica.Blocks.Noise.UniformNoise lossNoise(
    samplePeriod=7200,
    enableNoise=true,
    y_off=293.15,
    startTime=1000,
    y_min=280,
    y_max=280)
             annotation (Placement(transformation(extent={{100,62},{80,82}})));
  Modelica.Blocks.Math.Add calcCntrlDev(k1=-1, k2=1)
                                        annotation (Placement(transformation(extent={{54,-50},{34,-30}})));

  Modelica.Thermal.HeatTransfer.Components.ThermalResistor    thermalResistor(R=0.1)
                                                                                 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={30,72})));
  Modelica.Blocks.Sources.Step step(
    height=150,
    offset=100,
    startTime=10000) annotation (Placement(transformation(extent={{-112,44},{-92,64}})));
equation
  connect(boundary.ports[1],room_1. ports[1]) annotation (Line(points={{-50,0},{-29,0},{-29,10}}, color={0,127,255}));
  connect(room_1.ports[2], volumeFlowRate.port_a)
    annotation (Line(points={{-27,10},{-30,10},{-30,0},{-20,0}}, color={0,127,255}));
  connect(volumeFlowRate.port_b, room_2.ports[1]) annotation (Line(points={{0,0},{19,0},{19,6}}, color={0,127,255}));
  connect(room_2.ports[2], volumeFlowRate1.port_a) annotation (Line(points={{21,6},{21,0},{30,0}}, color={0,127,255}));
  connect(heatCapacitor.port, convectiveResistor.solid) annotation (Line(points={{4,72},{4,60}}, color={191,0,0}));
  connect(convectiveResistor.fluid, room_2.heatPort) annotation (Line(points={{4,40},{2,40},{2,16},{10,16}}, color={191,0,0}));
  connect(convectiveResistor.Rc, expr_convectResistance.y) annotation (Line(points={{14,50},{48,50},{48,38},{53,38}}, color={0,0,127}));
  connect(controlHeatFlow.port, room_1.heatPort) annotation (Line(
      points={{-52,52},{-46,52},{-46,20},{-38,20}},
      color={167,25,48},
      thickness=0.5));
  connect(volumeFlowRate1.port_b, temperature.port_a) annotation (Line(points={{50,0},{56,0}}, color={0,127,255}));
  connect(temperature.port_b, sink.ports[1]) annotation (Line(points={{76,0},{82,0}}, color={0,127,255}));
  connect(controller.u, calcCntrlDev.y) annotation (Line(points={{20,-40},{33,-40}}, color={0,0,127}));
  connect(calcCntrlDev.u2, combiTimeTable.y[1]) annotation (Line(points={{56,-46},{66,-46},{66,-64},{75,-64}}, color={0,0,127}));
  connect(calcCntrlDev.u1, temperature.T) annotation (Line(points={{56,-34},{66,-34},{66,-11}}, color={0,0,127}));
  connect(combiTimeTable.y[2], boundary.m_flow_in) annotation (Line(points={{75,-64},{-90,-64},{-90,8},{-70,8}}, color={0,0,127}));
  connect(heatCapacitor.port, thermalResistor.port_a) annotation (Line(points={{4,72},{20,72}}, color={191,0,0}));
  connect(thermalResistor.port_b, lossHeatFlow.port) annotation (Line(points={{40,72},{50,72}}, color={191,0,0}));
  connect(lossHeatFlow.T, lossNoise.y) annotation (Line(points={{72,72},{79,72}}, color={0,0,127}));
  connect(controller.y, controlHeatFlow.Q_flow) annotation (Line(points={{-3,-40},{-80,-40},{-80,52},{-72,52}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=300000,
      __Dymola_NumberOfIntervals=5000,
      Tolerance=1e-06,
      __Dymola_Algorithm="Dassl"));
end DoubleRoom_ContinuousPI;
