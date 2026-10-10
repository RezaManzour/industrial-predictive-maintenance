"""Keras MLP mirror: same numeric inputs and split, Type as one-hot (no embedding)."""

from __future__ import annotations

import numpy as np

SEED = 42


def fit_keras_mlp(
    X_tr: np.ndarray,
    t_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    t_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 200,
    patience: int = 15,
):
    """Return the best-val Keras model (by PR-AUC) and its val scores."""
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(SEED)
    x_tr = np.hstack([X_tr, np.eye(3, dtype=np.float32)[t_tr]])
    x_val = np.hstack([X_val, np.eye(3, dtype=np.float32)[t_val]])
    model = keras.Sequential(
        [
            keras.Input(shape=(x_tr.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    weight = {0: 1.0, 1: float((y_tr == 0).sum() / (y_tr == 1).sum())}
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc", mode="max", factor=0.5, patience=5
        ),
    ]
    model.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        batch_size=256,
        class_weight=weight,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    return model, model.predict(x_val, verbose=0).ravel()


def predict_keras_mlp(model, X: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = np.hstack([X, np.eye(3, dtype=np.float32)[t]])
    return model.predict(x, verbose=0).ravel()
