within SMArtIInt.Internal.Utilities.SubModels;
model Array2DDeflatteningModel
  parameter Integer numberOfOutput "Number of Real Outputs";
  parameter Integer batchSize=1 "Number of parallel batched inqueries";

  parameter Boolean useRowMajor = true "use true for row major flattening and false for column major flattening" annotation(Evaluate=true);

  Modelica.Blocks.Interfaces.RealInput  flatArray[batchSize*numberOfOutput] annotation (Placement(transformation(extent={{-118,-20},{-78,20}})));
  Modelica.Blocks.Interfaces.RealOutput arrayOut         [batchSize, numberOfOutput] annotation (Placement(transformation(extent={{92,-20},{132,20}})));
equation

  if useRowMajor then
    for i in 1:batchSize loop
      for j in 1:numberOfOutput loop
        flatArray[(i-1)*numberOfOutput + j] = arrayOut[i, j];
      end for;
    end for;
  else
    for i in 1:numberOfOutput loop
      for j in 1:batchSize loop
        flatArray[(i-1)*batchSize + j] = arrayOut[j, i];
      end for;
    end for;
  end if;

  annotation (Icon(graphics={Rectangle(
          extent={{-100,100},{100,-100}},
          lineColor={0,0,0},
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid), Bitmap(extent={{-80,-80},{80,80}}, fileName=
              "modelica://SMArtIInt/Resources/Images/Icon_Deflattening.svg")}));
end Array2DDeflatteningModel;
