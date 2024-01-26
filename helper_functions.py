from kneed import KneeLocator
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import sequence 
from tf_keras_vis.gradcam import Gradcam

### 1. get degradation knee point ###
def get_knee_point(discharge_agg_data):
    """
    Takes a discharge capacity degradation curve,
    apply smoothing and locate the degradation knee point,
    returns the knee point

    Parameters:
        discharge_agg_data: pd.DataFrame, the discharge capacity degradation curve
        
    Returns:
        knee: int, the knee point on the degradation curve
    """
    fig, ax = plt.subplots()
    org = discharge_agg_data.query("capacity_retention>0.75")
    org.plot("cycle_number", "capacity_retention", ax=ax, label="org")
    smoothed = pd.Series(
        sm.nonparametric.lowess(
            org["capacity_retention"], org["cycle_number"], 
            frac=0.25
        )[:, 1]
    )
    smoothed.plot(ax=ax, label="smoothed")
    knee = KneeLocator(
        smoothed.index, smoothed.values, 
        S=1, curve="concave", direction="decreasing", interp_method="interp1d"
    ).elbow
    ax.axvline(knee, color="r", label="knee")
    ax.legend()
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Capacity retention")
    return knee

### 2. CNN model train and visualization ###
def CNN1D_train(X_train, y_train, X_test, y_test, epochs, batch_size, n_cycle=10):
    """
    Takes training data and a few model hyperparameters,
    train a 1DCNN model and plot the results
    returns the trained model

    Parameters:
        X_train: numpy.array, the feature values in training data
        y_train: numpy.array, the target values in training data
        X_test: numpy.array, the feature values in test data
        y_test: numpy.array, the target values in test data
        epochs: int, number of epochs to train
        batch_size: int, number of batch size
        n_cycle: int, default=10, use the data from the first 10 cycles
        
    Returns:
        model: tf.Model, the trained model
    """
    model = Sequential()
    model.add(Conv1D(
        filters=4, kernel_size=10, input_shape=(X_train.shape[1], X_train.shape[2]), 
        activation="relu",
    ))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(20, activation="relu"))
    model.add(Dense(y_train.shape[1]))
    model.compile(loss="mean_absolute_error", optimizer="adam")
    # Fit network
    model.fit(
        X_train, y_train, 
        epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2
    )
    # Evaluate model
    train_pred, test_pred = (model.predict(X_train), model.predict(X_test))
    train_err, test_err = (
        np.mean((np.abs((n_cycle/train_pred - n_cycle/y_train) / (n_cycle/y_train))).flatten()),
        np.mean((np.abs((n_cycle/test_pred - n_cycle/y_test) / (n_cycle/y_test))).flatten()),
    )

    fig, ax = plt.subplots()
    lim = (0., 70)
    ax.plot(
        n_cycle/y_train.flatten(), n_cycle/train_pred.flatten(), marker="o", linewidth=0,
        label="Train: %.2f" %train_err
    )
    ax.plot(
        n_cycle/y_test.flatten(), n_cycle/test_pred.flatten(), marker="o", linewidth=0,
        label="Test: %.2f" %test_err
    )
    ax.plot(lim, lim, color="k")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend()
    ax.set_xlabel("Actual knee")
    ax.set_ylabel("Predicted knee")
    return model

def plot_saliency(model, X, discharge_data, n_cycle=10):
    """
    Takes a trained 1DCNN model and the discharge curve data,
    plot the saliency map and highlight the most informative region

    Parameters:
        model: tf.Model, the trained 1DCNN model
        X: numpy.array, the feature values
        discharge_data: pd.Series, the discharge curve data
        n_cycle: int, default=10, use the data from the first 10 cycles
    """
    def _model_modifier_function(cloned_model):
        cloned_model.layers[-1].activation = tf.keras.activations.linear

    gradcam = Gradcam(
        model,
        model_modifier=_model_modifier_function,
        clone=True,
    )

    fig, ax = plt.subplots()
    tempt = pd.DataFrame(discharge_data, columns=["U"]).reset_index()
    tempt["cycle_number"] = tempt["discharge_cycle"].apply(int)
    s_res = gradcam(
        lambda x: x, X, 
        seek_penultimate_conv_layer=False,
        penultimate_layer=model.layers[0].name
    )
    ax.pcolor([s_res.flatten()]*5)
    for n in range(n_cycle):
        tempt.query("cycle_number==%d" %n).reset_index()["U"].iloc[1:].plot(
            ax=ax, color="w", 
        )
    ax.set_ylim((discharge_data.min(), discharge_data.max()))
    ax.set_xticks([])
    ax.set_xlabel("Normalized capacity")
    ax.set_ylabel("U (V)")
    return

### 3. get curve variance ###
def get_area_var(
    end_perc, discharge_data, start=0, n_cycle=10,
):
    """
    Takes discharge voltage curves and end location,
    returns the averaged curve variation
    
    Parameters:
        end_perc: float, the end location on discharge voltage curve
        discharge_data: pd.DataFrame, the discharge curve data
        start: int, default=0, use the data starting from the 1st cycle
        n_cycle: int, default=10, use the data from the first 10 cycles
        
    Returns: 
        area_var: float, the averaged curve variation
    """    
    tempt = []
    n = _get_number_pt(discharge_data, 0.1)
    for cycle in range(start, start+n_cycle):
        if end_perc == 0.1:
            ts = discharge_data.query(
                "cycle_number==%d" %cycle
            ).iloc[:n]["U"].to_numpy().reshape(-1, 1)
        else:
            t = discharge_data.query("cycle_number==%d" %cycle)
            ts = t.iloc[
                int(len(t)*end_perc)-n:int(len(t)*end_perc)
            ]["U"].to_numpy().reshape(-1, 1)
        tempt.append(ts)
    X = np.column_stack(tempt)
    return np.mean(_diff_area(X[:, 0], X[:, 1:]))

def _get_number_pt(df, perc):
    return int(perc * len(df.query("cycle_number==0")))

def _diff_area(base, X):
    return [np.sum(np.abs(X[:, i]-base)) for i in range(X.shape[1])]