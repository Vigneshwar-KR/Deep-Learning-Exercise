from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D


def get_ae(encoding_dim):    
  input_img = Input(shape=(784,))
  encoded = Dense(encoding_dim, activation="relu")(input_img)
  decoded = Dense(784, activation="sigmoid")(encoded)

  autoencoder = Model(inputs=input_img, 
                      outputs=decoded)

  autoencoder.compile(optimizer="adam",
                      loss="binary_crossentropy")
  return autoencoder

def get_deep_ae():
  input_img = Input(shape=(784,))
  encoded = Dense(128, activation="relu")(input_img)
  encoded = Dense(64, activation="relu")(encoded)
  encoded = Dense(32, activation="relu")(encoded)

  decoded = Dense(64, activation="relu")(encoded)
  decoded = Dense(128, activation="relu")(decoded)
  decoded = Dense(784, activation="sigmoid")(decoded)

  autoencoder_deep = Model(inputs=input_img, 
                            outputs=decoded)

  autoencoder_deep.compile(optimizer="adam",
                          loss="binary_crossentropy")
  return autoencoder_deep

def get_conv_ae():
  input_img = Input(shape=(28, 28, 1))

  x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2), padding='same')(x)

  # at this point the representation is (4, 4, 8) i.e. 128-dimensional

  x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(16, (3, 3), activation='relu')(x)
  x = UpSampling2D((2, 2))(x)
  decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

  autoencoder_conv = Model(inputs=input_img,
                          outputs=decoded)

  autoencoder_conv.compile(optimizer='adam', 
                          loss='binary_crossentropy')
  return autoencoder_conv