within SMArtIInt;
package BaseClasses
  extends Modelica.Icons.BasesPackage;

annotation (Documentation(info="<html>
<p>The package containes templates to include different types of neural networks in Modelica. One find blocks within the model which have to be connected with the dessired in- and outputs. The user should extend the model, give all parameters and give a interface which has to be connected to the inner blocks.</p>
<p><br>Steps to include a model:</p>
<ol>
<li>Create and train a model in TensorFlow</li>
<li>Export a trained TensorFlow model as TfLite model</li>
<li>Extends the appropriate base class</li>
<li>Parametrize the model (provide path, number of in- and outputs, etc.)</li>
<li>Define the interface and connect the in- and outputs of the blocks. The arrays have the same structure as those in TensorFlow: the inputs have to be connected in the same manner as they are used in the neural network during training.</li>
</ol>
<p><br>The examples <a href=\"modelica://SMArtIInt.Tester.PipeHeatTransferExample.PipeLocalHeatTransfer_smallNN\">PipeLocalHeatTransfer_smallNN</a> and <a href=\"modelica://SMArtIInt.Tester.ExamplePI.TF_PI_Stateful\">TF_PI_Stateful</a> uses this approach to create a model.</p>
</html>"));
end BaseClasses;
