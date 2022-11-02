within SMArtIInt.BaseClasses;
partial model BaseGenericNeuralNet
  extends BaseNeuralNet;

  Internal.Utilities.RunInferenceFlatInput runInference(
    final nTotalInputsElements= nInputElements,
    final nTotalOutputElements=nOutputElements,
    final smartiint=smartiint) annotation (Placement(transformation(extent={{-10,-10},{10,10}})));

  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)),
    Documentation(info="<html>
<p>This is the most generic base class to include neural networks within Modelica. It can be used for any neural network. For easier handling the specialized versions BaseFeedForwardNeuralNet, BaseRecurrentNeuralNet and BaseStatefulRecurrentNeuralNet are available.</p>
<p>The most likely use case of this model is with a multi-layer perceptron neural network. </p>
<p>In order to include a neural network in Model, extend this base class in your own model. After extending you have to</p>
<ul>
<li>give the path of the TFLite model</li>
<li>specify the number of dimensions for in and output</li>
<li>specify the vector sizes in each input and output dimension</li>
<li>create input and output connectors</li>
<li>connect input and output connectors to the single input and single output vector of the runInference submodel</li>
</ul>
<p>The runInference model uses flattened vectors for input and output. The total number of elements equals the product of all input or output sizes, respectively. The user has to connect the defined input and output to the flattened vectors in the same order as they are used within the created neural network. </p>
</html>"));
end BaseGenericNeuralNet;
