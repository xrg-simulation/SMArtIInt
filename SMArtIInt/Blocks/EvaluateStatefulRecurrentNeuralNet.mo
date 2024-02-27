within SMArtIInt.Blocks;
model EvaluateStatefulRecurrentNeuralNet
  extends BaseClasses.BaseStatefulRecurrentNeuralNet;
  Modelica.Blocks.Interfaces.RealInput u[size(array2DFlatteningModel.arrayIn, 1),size(array2DFlatteningModel.arrayIn,
    2)] annotation (Placement(transformation(extent={{-110,-10},{-90,10}}), iconTransformation(extent={{-110,-10},{-90,10}})));
  Modelica.Blocks.Interfaces.RealOutput y[size(array2DDeflatteningModel.arrayOut, 1),size(array2DDeflatteningModel.arrayOut, 2)]
    annotation (Placement(transformation(extent={{90,-10},{110,10}})));
equation
  connect(array2DFlatteningModel.arrayIn, u)
    annotation (Line(points={{-40,0},{-100,0}}, color={0,0,127}));
  connect(array2DDeflatteningModel.arrayOut, y) annotation (Line(points={{41.2,0},{100,0}}, color={0,0,127}));
  annotation (Documentation(info="<html>
<p>Use this block if you want to include a recurrent neural network in Modelica which has been created with the flag stateful=True in TensorFlow. Please notice, that TFLite is not capable to handle the stateful states internally. Therefore, the neural network has to be created with access to all states as additional inputs and outputs. In this context the inputs and outputs have to be additional access points to the neural networks. SMArtIInt will handle the updates of the states by storing the values of the state outputs and feed them back into the state inputs. Therefore, for all states matching in- and output have to exist. When creating the neural network the user has to take care of this. The stateful PI controller created in the script <a href=\"modelica://SMArtIInt/Resources/ExampleNeuralNets/PI.py\">ExampleNeuralNets/PIController/PI.py</a> gives an example how to expose the states as in- an outputs.</p>
<p>Please place this block in your own model. After that</p>
<ul>
<li>give the path to the TFLite model</li>
<li>specify the number of inputs</li>
<li>specify the sampling interval</li>
<li>Provide values for the input of the block in the same way as they are given in the training</li>
</ul>
<p>Most likely, the stateful RNN will be trained for time discret data and therefore it has to be called only at discrete time instances. The user has to provide the samplingInterval for the discrete time instances. If continuous = false, the model will create events at each of the time instances and will only call the model at these instances. The many events have an impact on simulation performance. To increase performance the user can set continuous = true. In that case the model can be called for any times as it is demanded by the solver. SMArtIInt will internally call the neural network only at the time sampled time points. Hence, no events are created, but the inputs for the neural network have to be interpolated. Additionally, solution accuracy for the states of the neural network and the impact of the interpolation of the inputs cannot be controlled by the solver directly. Only the impact on the states in the model can be evaluated. Therefore, this approach might created inaccurate solutions especially if mult-step solver like DASSL and Cvode are used with a high tolerance value. The user should compare the simulation results to those with continuous = false or to results with continuous = true and lower tolerance and/or single step solver like (Radau).</p>
<p>For an example take a look at the model <a href=\"modelica://SMArtIInt/Tester.ExamplePI.TF_PI_Stateful\">Tester.ExamplePI.TF_PI_Stateful</a>. This example does not use this block directly but the usage is similar as it extends the model which is extended by this block.</p>
</html>"));
end EvaluateStatefulRecurrentNeuralNet;
