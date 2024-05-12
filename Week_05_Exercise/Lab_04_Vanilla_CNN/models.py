from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, \
                                    Flatten, \
                                    Dropout, \
                                    BatchNormalization, \
                                    Conv2D, \
                                    MaxPooling2D


def create_mlp(hidden_layers, n_classes):
  """Function to create a Multi layer perceptron

  Args:
      hidden_layers (np.array): array with the neurons
      for each hidden layer
      n_classes (int): number of classes

  Returns:
      Kearas model: a Keras model created using the
      Sequential API
  """
  model = Sequential()
  # Input layer
  model.add(Flatten(input_shape=(256,256,3)))
  # Hidden layers
  for i in hidden_layers:
    model.add(Dense(i, 
                    activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
  # Output layer
  model.add(Dense(n_classes, 
                  activation="softmax"))
  
  # Compile the model
  model.compile(optimizer='adam',
                loss="categorical_crossentropy",
                metrics=['accuracy']
                )
  return model


def create_cnn(filters, k, input_shape, n_classes):
  """Function to create a Convolutional neural network

  Args:
      filters (np.array): array with the number of filters
        used in each convolutional layer.
      k (int): kernel size used in each convolutional layer.
      input_shape (): shape of the input images [width,height,channels].
      n_classes (int): number of classes.

  Returns:
      Keras model: a Keras model create using the Sequential API.
  """
  
  model = Sequential()
  # Convolutional block 1
  model.add(Conv2D(filters[0], 
                   (k,k), 
                   input_shape=input_shape, 
                   activation="relu"))    
  model.add(BatchNormalization())
  model.add(MaxPooling2D(2,2))  
  model.add(Dropout(rate=0.2))
  
  # More convolutional blocks
  for n in filters[1:]:
    model.add(Conv2D(n, 
                     (k,k), 
                     activation="relu"))        
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))    
    model.add(Dropout(rate=0.2))
  
  # Flatten and output layer
  model.add(Flatten())
  model.add(Dense(100, 
                  activation="relu"))  
  # Output layer
  model.add(Dense(n_classes, 
                  activation="softmax"))
  
  # Compile the model
  model.compile(optimizer='adam',
                loss="categorical_crossentropy",
                metrics=['accuracy']
                )
  return model