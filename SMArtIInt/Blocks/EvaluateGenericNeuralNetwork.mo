within SMArtIInt.Blocks;
model EvaluateGenericNeuralNetwork
  extends BaseClasses.BaseGenericNeuralNet;

  Modelica.Blocks.Interfaces.RealInput u[nInputElements] annotation (Placement(transformation(extent={{-110,-10},{-90,10}}),
        iconTransformation(extent={{-110,-10},{-90,10}})));
  Modelica.Blocks.Interfaces.RealOutput               y
                                               [nOutputElements] annotation (Placement(transformation(extent={{90,-10},
            {110,10}}), iconTransformation(extent={{90,-10},{110,10}})));
equation
  connect(runInference.u, u) annotation (Line(points={{-10,0},{-100,0}},  color={0,0,127}));
  connect(runInference.y, y) annotation (Line(points={{10,0},{100,0}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)),
    Documentation(info="<html>
<p>This is the most generic block to include neural networks within Modelica. It extends the BaseGenericNeuralNet and provides generic in- and outputs. It can be used for any neural network. For easier handling the specialized versions EvaluateFeedForwardNeuralNet, EvaluateRecurrentNeuralNet and EvaluateStatefulRecurrentNeuralNet are available.</p>
<p>This most likely use case of this model is with a multi-layer perceptron neural network. </p>
<p>In order to include a neural network in Model, place this block in your own model. You have to </p>
<ul>
<li>give the path of the TFLite model</li>
<li>specify the number of dimensions for in and output</li>
<li>specify the vector sizes in each input and output dimension</li>
<li>create input and output connectors</li>
<li>connect input and output connectors to the single input and single output vector of the runInference submodel</li>
</ul>
<p>The runInference model uses a flattened vectors for input and output. The total number of elements equals the product of all input or output sizes, respectively. The user has to connect the defined input and output to the flattened vectors in the same order as they are used within created neural network. </p>
</html>"));
end EvaluateGenericNeuralNetwork;
