from tensorflow.keras import Model, layers


def create_my_model():
    """Creates keras model that is used in all scenarios.
    CNN from cats vs. dogs Colab notebook

    Returns:
        tensorflow.python.keras.engine.functional.Functional: model
    """
    img_input = layers.Input(shape=(150, 150, 3))

    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    # x = layers.BatchNormalization()(img_input)
    x = layers.Conv2D(16, 3, activation="relu")(img_input)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    # x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    # x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 256 hidden units
    x = layers.Dense(256, activation="relu")(x)

    # Create output layer with a single node and sigmoid activation
    output = layers.Dense(1, activation="sigmoid")(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully
    # connected layer + sigmoid output layer
    return Model(img_input, output)
