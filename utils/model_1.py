import tensorflow as tf
from tensorflow import keras
def Conv1D(input_shape,num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=512, kernel_size=3, padding="valid", activation="relu")(input_layer)
    drop1=keras.layers.Dropout(0.5)(conv1)
    
    conv2 = keras.layers.Conv1D(filters=512, kernel_size=3, padding="valid", activation="relu")(conv1)
    drop2=keras.layers.Dropout(0.5)(conv2)

    conv3 = keras.layers.Conv1D(filters=256, kernel_size=2, padding="valid", activation="relu")(drop2) # type: ignore
    drop3 = keras.layers.Dropout(0.5)(conv2) # type: ignore
    gap=keras.layers.MaxPooling1D(pool_size=2)(conv2)

    
#    gap = keras.layers.GlobalAveragePooling1D()
    flat1=keras.layers.Flatten()(gap)
    dens1=keras.layers.Dense(256,activation="relu")(flat1)
    dens2=keras.layers.Dense(128,activation="relu")(dens1)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(dens2)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def make_model(train_features, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    return keras.Sequential(
        [
            keras.layers.Dense(
                16,
                activation='relu',
                input_shape=
                    train_features,
            ),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(
                1, activation='sigmoid', bias_initializer=output_bias
            ),
        ]
    )

def LSTM_Model(input_shape,num_classes):
    model=keras.models.Sequential()
    model.add(keras.layers.LSTM(units = 100, return_sequences = True, input_shape = input_shape ))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(units = 100, return_sequences = True))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(units = 100, return_sequences = True))
    model.add(keras.layers.Dropout(0.25))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(keras.layers.LSTM(units = 100))
    model.add(keras.layers.Dropout(0.25))

    #Adding Batch Normalisation
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Adding the output layer
    model.add(keras.layers.Dense(units = num_classes, activation = 'softmax'))
    
    return model


def LSTM_Model_Conv(input_shape,num_classes):
    model=keras.models.Sequential()
    #model.add(keras.layers.Conv1D(2056, 5,activation = 'relu',input_shape = input_shape))
    #model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(units = 100, return_sequences = True,input_shape = input_shape ))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(units = 100, return_sequences = True))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(units = 100, return_sequences = True))
    model.add(keras.layers.Dropout(0.25))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(keras.layers.LSTM(units = 100))
    model.add(keras.layers.Dropout(0.25))

    #Adding Batch Normalisation
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Adding the output layer
    model.add(keras.layers.Dense(units = num_classes, activation = 'softmax'))
    
    return model