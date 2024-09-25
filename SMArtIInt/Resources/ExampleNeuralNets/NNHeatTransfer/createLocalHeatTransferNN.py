import os
import numpy as np
import pyDOE2 as doe
import matplotlib.pylab as plt
import tensorflow as tf
import datetime
import plotly.graph_objects as go
import plotly.io as pio
import pickle

# plotly config
pio.renderers.default = 'browser'


# the following two function are used for data generation and reproduce
# the local Nusselt value calculation for the pipe model in the Modelica
# Standard Library 
def spliceFunction(pos, neg, x, deltax):
    # this function is taken from the Modelica Standard Library as found in
    # https://github.com/modelica/ModelicaStandardLibrary and modified to use
    # it in python. The original was pupliced with the following copyright
    # notice: 
    # Copyright (c) 1998-2020, Modelica Association and contributors
    # All rights reserved.

    scaledX1 = x / deltax
    scaledX = scaledX1 * np.arcsin(1)
    y = (np.tanh(np.tan(scaledX)) + 1) / 2
    y[scaledX1 <= -0.999999999] = 0
    y[scaledX1 >= 0.999999999] = 1
    return pos * y + (1 - y) * neg


def calc_nusselt(Res, Prs, dByLs):
    # this function is taken from the Modelica Standard Library as found in
    # https://github.com/modelica/ModelicaStandardLibrary and modified to use
    # it in python. The original was pupliced with the following copyright
    # notice: 
    # Copyright (c) 1998-2020, Modelica Association and contributors
    # All rights reserved.

    Res = np.array(Res)
    Prs = np.array(Prs)
    dByLs = np.array(dByLs)

    Xis = (1.8 * np.log10(Res) - 1.5) ** (-2)

    Nu_1 = 3.66
    Nus_2 = 1.077 * (Res * Prs * dByLs) ** (1.0 / 3)

    Nus_lam = (Nu_1 ** 3 + 0.7 ** 3 + (Nus_2 - 0.7) ** 3) ** (1.0 / 3)

    Nus_turb = (Xis / 8) * Res * Prs / (1 + 12.7 * (Xis / 8) ** 0.5 * (Prs ** (2 / 3) - 1)) * \
               (1 + 1 / 3 * (dByLs) ** (2 / 3))

    return spliceFunction(Nus_turb, Nus_lam, Res - 6150, 3850)


def generate_nusselt_data(N):
    # define the ranges for the input variables
    Prs_range = [0.5, 15]
    Res_range = [0, 1e5]
    dByLs_range = [0, 10]

    # generate the data with latin hypercube sampling 
    doe_sampling = doe.lhs(3, N)
    Prs = Prs_range[0] + (Prs_range[1] - Prs_range[0]) * doe_sampling[:, 0]
    Res = Res_range[0] + (Res_range[1] - Res_range[0]) * (np.exp(doe_sampling[:, 1]) - 1)
    dByLs = dByLs_range[0] + (dByLs_range[1] - dByLs_range[0]) * doe_sampling[:, 2]

    return Res, Prs, dByLs, calc_nusselt(Res, Prs, dByLs)


# as first step we generate the data
n = 5000
preset = "large"
filename = f"data_{n}.pckl"

if os.path.exists(filename):
    with open(filename, 'rb') as file:
        load = pickle.load(file)
    Res = load[0]
    Prs = load[1]
    dByLs = load[2]
    Nus = load[3]
else:
    with open(filename, 'wb') as file:
        Res, Prs, dByLs, Nus = generate_nusselt_data(n)
        pickle.dump([Res, Prs, dByLs, Nus], file)
    # we create some plots to check the data
    plt.Figure()
    sc = plt.scatter(Res, Nus, c=Prs)
    plt.xlabel('Re')
    plt.ylabel('Nu')
    plt.colorbar(sc).set_label('Pr')
    plt.show()

    plt.Figure()
    sc = plt.scatter(Res, Nus, c=dByLs)
    plt.xlabel('Re')
    plt.ylabel('Nu')
    plt.colorbar(sc).set_label('dByLs')
    plt.show()

# we create a combined array
data = np.array([Res, Prs, dByLs]).T
bool_filter = [True] * len(Res)
data = data[bool_filter, :]
Nus = Nus[bool_filter]
Res = Res[bool_filter]
Prs = Prs[bool_filter]
dByLs = dByLs[bool_filter]

# data = np.array([Res]).T
# create an array to split the data in to training and test
# validation split will be done in the call of train
splits = np.random.choice([0, 1], p=[0.9, 0.1], size=len(Nus))
train_data = data[splits == 0]
test_data = data[splits == 1]
Nus_train = Nus[splits == 0]
Nus_test = Nus[splits == 1]

# create a neural network which will be used to represent the data
config_name = preset + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(config_name)
if preset == "large":
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=1 / np.max(train_data, axis=0), input_shape=(data.shape[1],),
        name="Scaler"))
    model.add(tf.keras.layers.Dense(units=256, activation='relu',
                                    name="Hidden_1"))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=128, activation='relu',
                                    name="Hidden_2"))
    model.add(tf.keras.layers.Dense(units=64, activation='relu',
                                    name="Hidden_3"))
    model.add(tf.keras.layers.Dense(units=32, activation='relu',
                                    name="Hidden_4"))
    model.add(tf.keras.layers.Dense(units=16, activation='relu',
                                    name="Hidden_5"))
    model.add(tf.keras.layers.Dense(1, activation='tanh',
                                    name="Out_1"))
    model.add(tf.keras.layers.Dense(1, activation='linear',
                                    name="Out_2"))
    model.layers[-1].set_weights([np.array([[max(Nus)]]), np.array([0])])

    model.compile(loss='MeanAbsolutePercentageError',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["RootMeanSquaredError",
                           "MeanAbsolutePercentageError"])
else:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=1 / np.max(train_data, axis=0), input_shape=(data.shape[1],),
        name="Scaler"))
    model.add(tf.keras.layers.Dense(units=64, activation='relu',
                                    name="Hidden_1"))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=32, activation='relu',
                                    name="Hidden_2"))
    model.add(tf.keras.layers.Dense(units=16, activation='relu',
                                    name="Hidden_3"))
    model.add(tf.keras.layers.Dense(1, activation='tanh',
                                    name="Out_1"))
    model.add(tf.keras.layers.Dense(1, activation='linear',
                                    name="Out_2"))
    model.layers[-1].set_weights([np.array([[max(Nus)]]), np.array([0])])

model.compile(loss='MeanAbsolutePercentageError',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["RootMeanSquaredError",
                       "MeanAbsolutePercentageError"])
cb_es = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0,
    patience=500,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                 patience=5, min_lr=0.001)
log_dir = "ht_log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_data, Nus_train,
                    verbose=1, batch_size=256,
                    epochs=10000, validation_split=0.1,
                    callbacks=[cb_es, tb_callback, reduce_lr])

model.save(os.path.join(config_name, 'model.h'))
with open(os.path.join(config_name, 'history.pckl'), 'wb') as file:
    pickle.dump(history.history, file)

Nus_predict_train = model.predict(train_data).flatten()
Nus_predict_test = model.predict(test_data).flatten()
Nus_predict = model.predict(data).flatten()

path = config_name

plt.Figure()
plt.scatter(Res, Nus, c='blue', marker='o')
plt.scatter(Res, Nus_predict, c='red', marker='x')
plt.xlabel('Reynolds Number')
plt.ylabel('Nusselt Number')
plt.legend(["Data", "Model"])
plt.savefig(os.path.join(path, 'Nu-Re.png'))
plt.show()

plt.Figure()
plt.scatter(Nus_train, abs(Nus_predict_train / Nus_train - 1), c='black', marker='x')
plt.scatter(Nus_test, abs(Nus_predict_test / Nus_test - 1), c='green', marker='x')
plt.xlabel('Nusselt Number')
plt.ylabel('Relative Deviation')
plt.legend(["Training + Validation", "Test"])
plt.yscale('log')
plt.savefig(os.path.join(path, 'NuErr-Re.png'))
plt.show()

tf.keras.utils.plot_model(model, to_file=
os.path.join(path, "model.png"),
                          show_shapes=True)

plt.Figure()
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.xlabel('Epochs')
plt.ylabel('Root Mean Squared Error')
plt.legend(["Training", "Validation"])
plt.yscale('log')
plt.xscale('log')  #
plt.savefig(os.path.join(path, 'history.png'))
plt.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=Res, y=Nus, line=None, mode='markers', name='Data'))
fig.add_trace(go.Scatter(x=Res, y=Nus_predict, line=None, mode='markers', name='Model'))
fig.show()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(path, "model.tflite"), 'wb') as f:
    f.write(tflite_model)

### --------- Export To ONNX ------------------------------
model.summary()
model.save("savedmodel")
model.save("savedmodel.h5")

os.system("python -m tf2onnx.convert --saved-model savedmodel --output model.onnx --opset 13")
