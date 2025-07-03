from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from airontools.constructors.layers import layer_constructor
import keras

if __name__ == "__main__":
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build a simple neural network
    model_name = "binary_classification"
    n_units = [3, 2]
    kernel_regularizer_l1 = 0.001
    kernel_regularizer_l2 = 0.001
    bias_regularizer_l1 = 0.001
    bias_regularizer_l2 = 0.001
    dropout_rate = 0.1
    normalization_type = "bn"
    inputs = keras.layers.Input(
        shape=(X_train.shape[-1],), name=f"{model_name}_input_layer"
    )
    outputs = None
    if len(n_units) > 0:
        for hidden_layer_i, n_units_i in enumerate(n_units):
            _inputs = inputs if hidden_layer_i == 0 else outputs
            _dropout_rate = 0.0 if hidden_layer_i == 0 else dropout_rate
            _activation = "tanh" if hidden_layer_i == 1 else "prelu"
            outputs = layer_constructor(
                _inputs,
                name=f"{model_name}_hidden_layer_{hidden_layer_i}",
                units=n_units_i,
                activation=_activation,
                kernel_regularizer_l1=kernel_regularizer_l1,
                kernel_regularizer_l2=kernel_regularizer_l2,
                bias_regularizer_l1=bias_regularizer_l1,
                bias_regularizer_l2=bias_regularizer_l2,
                dropout_rate=_dropout_rate,
                normalization_type=normalization_type,
            )
    else:
        outputs = inputs
    outputs = layer_constructor(
        outputs,
        name=f"{model_name}_outputs",
        units=1,
        activation="sigmoid",
        kernel_regularizer_l1=kernel_regularizer_l1,
        kernel_regularizer_l2=kernel_regularizer_l2,
        bias_regularizer_l1=bias_regularizer_l1,
        bias_regularizer_l2=bias_regularizer_l2,
        dropout_rate=dropout_rate,
        normalization_type=normalization_type,
    )
    model = keras.models.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name,
    )

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1)

    # Evaluate on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")
