within SMArtIInt.Internal.Utilities;
model RunInferenceFlatInputStatefulRNN
  parameter Integer nTotalInputsElements;
  parameter Integer nTotalOutputElements;

  parameter SMArtIIntClass smartiint;

  parameter Boolean continuous=true;
  parameter Real samplePeriod=0 "Fixed sample period for RNNs";

  final parameter Modelica.Units.SI.Time startTime(fixed=false);
  Boolean sampleTrigger;

  Interfaces.RealVectorInput realVectorInput[nTotalInputsElements]
    annotation (Placement(transformation(extent={{-118,-20},{-78,20}})));
  Interfaces.RealVectorOutput y[nTotalOutputElements] annotation (Placement(transformation(extent={{80,-20},{120,20}})));

initial equation
  startTime = time;
equation
  if continuous then
    sampleTrigger = true;
    y[:] = InterfaceFunctions.runInferenceFlatTensor(
      smartiint,
      time,
      realVectorInput,
      nTotalOutputElements);
  else
    sampleTrigger = sample(startTime, samplePeriod);

    when {sampleTrigger, initial()} then
      y[:] = InterfaceFunctions.runInferenceFlatTensor(
        smartiint,
        time,
        realVectorInput,
        nTotalOutputElements);
    end when;


  end if;

  annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={Rectangle(
          extent={{-100,100},{100,-100}},
          pattern=LinePattern.None,
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid), Bitmap(extent={{-102,-100},{102,100}}, fileName="modelica://SMArtIInt/Resources/Images/Icon_Inference.svg")}),
      Diagram(coordinateSystem(preserveAspectRatio=false)));
end RunInferenceFlatInputStatefulRNN;
