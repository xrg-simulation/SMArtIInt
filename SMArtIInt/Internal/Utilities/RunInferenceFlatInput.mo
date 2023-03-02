within SMArtIInt.Internal.Utilities;
model RunInferenceFlatInput
  parameter Integer nTotalInputsElements;
  parameter Integer nTotalOutputElements;

  parameter SMArtIIntClass smartiint;

  Modelica.Blocks.Interfaces.RealInput u[nTotalInputsElements]
    annotation (Placement(transformation(extent={{-120,-20},{-80,20}})));
  Modelica.Blocks.Interfaces.RealOutput y[nTotalOutputElements] annotation (Placement(transformation(extent={{80,-20},{120,20}})));

equation
  y[:] =InterfaceFunctions.runInferenceFlatTensor(
    smartiint,
    time,
    u,
    nTotalOutputElements);

  annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={Rectangle(
          extent={{-100,100},{100,-100}},
          pattern=LinePattern.None,
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid),                                 Bitmap(extent={{-102,-100},{102,100}}, fileName="modelica://SMArtIInt/Resources/Images/Icon_Inference.svg")}), Diagram(coordinateSystem(preserveAspectRatio=false)));
end RunInferenceFlatInput;
