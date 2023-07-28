within SMArtIInt.Internal.Utilities.SubModels;
model Array2DFlatteningModel
  parameter Integer numberOfInputs = 1 "Number of Real Inputs";
  parameter Integer batchSize=1 "Number of parallel batched inqueries";

  parameter Boolean useRowMajor = true "use true for row major flattening and false for column major flattening" annotation(Evaluate=true);

  Modelica.Blocks.Interfaces.RealOutput flatArray[batchSize*numberOfInputs] annotation (Placement(transformation(extent={{80,-20},{120,20}})));
  Modelica.Blocks.Interfaces.RealInput arrayIn[batchSize, numberOfInputs] annotation (Placement(transformation(extent={{-120,-20},{-80,20}})));
equation

  if useRowMajor then
    for i in 1:size(arrayIn,1) loop
      for j in 1:size(arrayIn,2) loop
        flatArray[(i-1)*size(arrayIn,2) + j] = arrayIn[i, j];
      end for;
    end for;
  else
    for i in 1:size(arrayIn,2) loop
      for j in 1:size(arrayIn,1) loop
        flatArray[(i-1)*size(arrayIn,1) + j] = arrayIn[j, i];
      end for;
    end for;
  end if;

  annotation (Icon(graphics={Rectangle(
          extent={{-100,100},{100,-100}},
          lineColor={0,0,0},
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid),
                             Bitmap(extent={{-80,-80},{80,80}}, fileName=
              "modelica://SMArtIInt/Resources/Images/Icon_Flattening.png")}));
end Array2DFlatteningModel;
