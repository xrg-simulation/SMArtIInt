within SMArtIInt;
package Blocks
  extends Modelica.Icons.VariantsPackage;

annotation (Documentation(info="<html>
<p>This package provides several blocks which can be used to include different types of neural networks within Modelica models. These blocks can be included in models, but their in- and outputs are still generic arrays. The user has to fill the arrays in the same manner as they are used during training in python.</p>
<p>Steps to include a model:</p>
<ol>
<li>Create and train a model in TensorFlow</li>
<li>Export a trained TensorFlow model as TfLite model</li>
<li>Place the corresponding block in your model</li>
<li>Parametrize the block (provide path, number of in- and outputs, etc.)</li>
<li>Connect the in- and outputs of the block. The arrays have the same structure as those in TensorFlow: the inputs have to be connected in the same manner as they are used in the neural network during training.</li>
</ol>
<p><br>The examples <a href=\"modelica://SMArtIInt.Tester.PipeHeatTransferExample.PipeLocalHeatTransfer\">PipeLocalHeatTransfer</a> and <a href=\"modelica://SMArtIInt.Tester.ExamplePI.TF_PI_RNN\">TF_PI_Stateful</a> use this approach.</p>
</html>"));
end Blocks;
