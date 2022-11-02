within SMArtIInt.Tester.PipeHeatTransferExample;
model NNHeatTransfer
  extends Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.PartialPipeFlowHeatTransfer;

  replaceable PipeLocalHeatTransfer pipeLocalHeatTransfer(batchSize=n) annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  Modelica.Blocks.Sources.RealExpression Expr_Res[n](y=Res) annotation (Placement(transformation(extent={{-74,4},{-54,24}})));
  Modelica.Blocks.Sources.RealExpression Expr_Prs[n](y=Prs) annotation (Placement(transformation(extent={{-74,-12},{-54,8}})));
  Modelica.Blocks.Sources.RealExpression Expr_dByLs[n](y={diameters[i]/lengths[i]/(if vs[i] >= 0 then (i - 0.5) else (n - i + 0.5)) for i in 1:n}) annotation (Placement(transformation(extent={{-74,-30},{-54,-10}})));
equation
  Nus = pipeLocalHeatTransfer.Nu[:];

  connect(Expr_Res.y, pipeLocalHeatTransfer.Re) annotation (Line(points={{-53,14},{-24,14},{-24,6},{-10,6}}, color={0,0,127}));
  connect(Expr_Prs.y, pipeLocalHeatTransfer.Pr) annotation (Line(points={{-53,-2},{-10,-2},{-10,0}}, color={0,0,127}));
  connect(Expr_dByLs.y, pipeLocalHeatTransfer.dByL) annotation (Line(points={{-53,-20},{-24,-20},{-24,-10},{-10,-10},{-10,-6}}, color={0,0,127}));
end NNHeatTransfer;
