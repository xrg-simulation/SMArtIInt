within SMArtIInt.BaseClasses;
partial model BaseStatefulRecurrentNeuralNet
  extends BaseNeuralNet(
    final stateful=true,
    final outputSizes={batchSize,numberOfOutputs},
    final outputDimensions=2,
    final inputSizes={batchSize,1,numberOfInputs},
    final inputDimensions=3);

  parameter Integer numberOfInputs=1 "Number of input values" annotation (Dialog(group="Input/Output Sizing"));
  parameter Integer numberOfOutputs=1 "Number of output values" annotation (Dialog(group="Input/Output Sizing"));
  parameter Integer batchSize=1 "Number of parallel batched simulations"
    annotation (Dialog(group="Input/Output Sizing"));

  parameter Boolean continuous=true "=true: model operates continuously; input values are delayed"
    annotation (Dialog(group="RNN Timing Settings"));


  Internal.Utilities.RunInferenceFlatInputStatefulRNN
                                           runInferenceFlatInputStatefulRNN(
    final nTotalInputsElements=nInputElements,
    final nTotalOutputElements=nOutputElements,
    final smartiint=smartiint,
    final continuous=continuous,
    final samplePeriod=samplePeriod)
                               annotation (Placement(transformation(extent={{-10,-10},{10,10}})));

  Internal.Utilities.SubModels.Array2DFlatteningModel array2DFlatteningModel(final numberOfInputs=numberOfInputs,
      final batchSize=batchSize) annotation (Placement(transformation(extent={{-40,-10},{-20,10}})));
  Internal.Utilities.SubModels.Array2DDeflatteningModel array2DDeflatteningModel(numberOfOutput=numberOfOutputs,
      batchSize=batchSize) annotation (Placement(transformation(extent={{20,-10},{40,10}})));

equation

  connect(array2DFlatteningModel.flatArray, runInferenceFlatInputStatefulRNN.u)
    annotation (Line(points={{-20,0},{-9.8,0}}, color={0,0,127}));
  connect(runInferenceFlatInputStatefulRNN.y, array2DDeflatteningModel.flatArray)
    annotation (Line(points={{10,0},{20.2,0}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    Documentation(info="<html>
    <p>Use this base class if you want to include a recurrent neural network in Modelica, which has been created with the flag stateful=True in TensorFlow. Please notice, that TFLite is not capable of handling the stateful states internally. Therefore, the neural network has to be created with access to all states as additional inputs and outputs. In this context, the inputs and outputs have to be additional access points to the neural networks. SMArtIInt will handle the updates of the states by storing the values of the state outputs, and feeding them back into the state inputs. Therefore, for all states, matching in- and output have to exist. When creating the neural network the user has to take care of this. The stateful PI controller created in the script <a href=\"modelica://SMArtIInt/Resources/ExampleNeuralNets/PI.py\">ExampleNeuralNets/PIController/PI.py</a> gives an example of how to expose the states as in- an outputs.</p>
<p>Please extend this model in your own model. After extending you have to</p>
<ul>
<li>give the path to the TFLite model</li>
<li>specify the number of inputs</li>
<li>specify the sampling interval</li>
<li>create input and output connectors</li>
<li>connect input and output connectors to the single input and single output vector of the runInference submodel</li>
</ul>
<p>Most likely, the stateful RNN will be trained for time discrete data and therefore it has to be called only at discrete time instances. The user has to provide the sampling interval for the discrete time instances. If continuous = false, the model will create events at each of the time instances and will only call the model at these instances. The many events have an impact on simulation performance. To increase performance the user can set continuous = true. In that case the model can be called for any times, as it is demanded by the solver. SMArtIInt will internally call the neural network only at the time sampled time points. Hence, no events are created, but the inputs for the neural network have to be interpolated. Additionally, the solution accuracy for the states of the neural network and the impact of the interpolation of the inputs cannot be controlled by the solver directly. Only the impact on the states in the model can be evaluated. Therefore, this approach might create inaccurate solutions, especially if a multi-step solver like DASSL and Cvode are used with a high tolerance value. The user should compare the simulation results to those with continuous = false or to results with continuous = true and lower tolerance and/or a single step solver like (Radau).</p>
<p>For an example take a look at the model <a href=\"modelica://SMArtIInt.Tester.ExamplePI.TF_PI_Stateful\">Tester.ExamplePI.TF_PI_Stateful</a></p>
</html>"));
end BaseStatefulRecurrentNeuralNet;
