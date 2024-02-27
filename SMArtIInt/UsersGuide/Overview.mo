within SMArtIInt.UsersGuide;
model Overview
  extends Modelica.Icons.Information;
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)),
    Documentation(info="<html>
    <p>The SMArtIINt Library aims to support the usage of different artificial intelligence models (AI) in Modelica simulation tools. Currently, it supports TensorFlow models exported as TFLite models within Dymola. There are two different ways to inlcude neural networks within a model. One could either use the <a href=\"modelica://SMArtIInt.BaseClasses\">Base Classes</a> (1) or the <a href=\"modelica://SMArtIInt.Blocks\">Blocks</a> (2). Both packages provide templates for the usage of different types of neural networks. The usage is explained in the package and the models.</p>
<p>SMArtIInt Library comes along with some <a href=\"modelica://SMArtIInt.Tester\">Examples</a>. The python scripts which created the used TFLite models can be found <a href=\"modelica://SMArtIInt/Resources/ExampleNeuralNets/\">here</a>.</p>
</html>"));
end Overview;
