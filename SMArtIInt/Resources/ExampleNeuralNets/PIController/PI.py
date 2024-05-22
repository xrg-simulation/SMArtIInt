# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from enum import Enum

class RnnType(Enum):
    """available types of RNN"""
    RNN = 0
    EXTSTATE = 1
    STATE = 2


def generateAiPi(rnnType: RnnType, window_size=1, k=30, T=1600, tau=100):
    stateful = rnnType == RnnType.STATE

    if rnnType == RnnType.EXTSTATE:
        modelInput = tf.keras.Input(shape=(1, 1), name="FeatureInput")
    elif rnnType == RnnType.STATE:
        modelInput = tf.keras.Input(batch_shape=(1, 1, 1), name="FeatureInput")
    else:
        modelInput = tf.keras.Input(shape=(window_size, 1), name="FeatureInput")

    layer = tf.keras.layers.SimpleRNN(units=2,
                                      activation='linear',
                                      use_bias=False,
                                      stateful=stateful,
                                      return_state=True,
                                      name="ZPassAndIntegrate",
                                      )
    state_input = tf.keras.Input(shape=(2,))
    if rnnType == RnnType.EXTSTATE:
        out, state = layer(modelInput, initial_state=[state_input])
    else:
        out, state = layer(modelInput, )
    out = tf.keras.layers.Dense(units=1,
                                activation='linear',
                                use_bias=False,
                                name="Weight")(out)
    if rnnType == RnnType.EXTSTATE:
        model = tf.keras.Model([modelInput, state_input], [out, state])
    else:
        model = tf.keras.Model([modelInput], [out])
    # model.build()

    # manipulate weights to mimic PI controller
    model.layers[-1].set_weights([np.array([[k / T * tau], [k]])])
    model.layers[-2].set_weights([
        np.array([[0, 1]]),
        np.array([[1, 0],
                  [1, 0]])
    ])
    return model


def step(height=1.0, duration=3600, start=500, tau=100, window_size=500):
    times = np.arange(0, duration, tau)
    step_data = np.zeros_like(times)
    step_data[times >= start] = height
    unrld_data = np.zeros([len(step_data), window_size, 1])
    for i in range(len(step_data)):
        start_idx = max(0, i - window_size + 1)
        n_elements = i - start_idx + 1
        unrld_data[i, -n_elements:, 0] = step_data[start_idx: i + 1]

    return times, step_data, unrld_data


### ----------------------Settings for the PI ----------------------
k = 30  # proportional gain
T = 1600  # integrator time constant
window_size = 250  # number of past elements used for non-state variant
rnn_type = RnnType.EXTSTATE  # define type of used RNN
if rnn_type == RnnType.RNN:
    tau = 100  # sampling rate
else:
    tau = 10
testTFLiteModel = False
### Test data
test_model = False
stepTime = 100

# create model and test data
model = generateAiPi(rnn_type, window_size=window_size, k=k, T=T, tau=tau)
times, step_data, unrld_data = step(height=1, duration=3600, start=500,
                                    tau=tau, window_size=window_size)

# test model
if test_model:
    if rnn_type == RnnType.RNN:
        results = model.predict(unrld_data).flatten()
    elif rnn_type == RnnType.STATE:
        results = []
        for val in step_data:
            results.append(model.predict([[val]]).flatten())
    else:
        states = np.array([[0, 0]])
        results = []
        for val in step_data:
            temp = model.predict([np.array([val]), states])
            states = temp[1]
            results.append(temp[0].flatten())

    fig = plt.Figure()
    plt.title('Step Answer of TF Model')
    plt.xlabel('Time in sec')
    plt.ylabel('Model Output')
    plt.ylim([0, 100])
    plt.plot(times, results)
    plt.show()

### --------- Export To TFLite ------------------------------
if rnn_type != RnnType.STATE:
    # export the model as tflite model

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    if rnn_type == RnnType.EXTSTATE:
        path = os.path.join('.', "PI_stateful.tflite")
    else:
        path = os.path.join('.', "PI.tflite")
    with open(path, 'wb') as f:
        f.write(tflite_model)

    if testTFLiteModel:

        interpreter = tf.lite.Interpreter(
            model_path=path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        states = np.array([[0, 0]])
        results = []

        for i, data_point in enumerate(step_data):
            if rnn_type == RnnType.EXTSTATE:
                np_features = np.array([[[data_point]]])
                np_features = np_features.astype(input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], np_features)
                states = states.astype(input_details[1]['dtype'])
                interpreter.set_tensor(input_details[1]['index'], states)
                interpreter.invoke()
                results.append(interpreter.get_tensor(
                    output_details[0]['index'])[0][0])
                states = interpreter.get_tensor(output_details[1]['index'])
            else:
                np_features = np.array([unrld_data[i, :, :]])
                np_features = np_features.astype(input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], np_features)
                interpreter.invoke()
                results.append(interpreter.get_tensor(
                    output_details[0]['index'])[0][0])

        fig = plt.Figure()
        plt.title('Step Answer of TFLite Model')
        plt.xlabel('Time in sec')
        plt.ylabel('Model Output')
        plt.ylim([0, 100])
        plt.plot(times, results)
        plt.show()
else:
    print("Stateful model cannot be exported as TFLite model!")

### --------- Export To ONNX ------------------------------
model.summary()
model.save("savedmodel")
model.save("savedmodel.h5")
if rnn_type == RnnType.EXTSTATE:
    os.system("python -m tf2onnx.convert --saved-model savedmodel --output PI_stateful.onnx --opset 13")
              #"--inputs FeatureInput:0,input_1:0 --outputs Weight:0,PassAndIntegrate:0")
elif rnn_type == RnnType.RNN:
    os.system("python -m tf2onnx.convert --saved-model savedmodel --output PI.onnx --opset 13")
