---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region colab_type="text" id="F2MV_JxL3O-y" -->
### <font color="blue"> Import Libraries </font>
<!-- #endregion -->

```python colab={} colab_type="code" id="2jX3tF9t3MFP"
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
```

```python colab={} colab_type="code" id="F3UMn0Sr-aG3"
num_classes = 10
input_shape = (28, 28, 1)
```

<!-- #region colab_type="text" id="bJvofKpd3cAy" -->
### <font color='blue'> Download the training data and test data</font>


<!-- #endregion -->

```python colab={} colab_type="code" id="kRqvK1VO-iRW"
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

<!-- #region colab_type="text" id="-4jbqR4n3mrr" -->
### <font color="blue"> Normalizing the data </font>


<!-- #endregion -->

```python colab={} colab_type="code" id="SwrHbEFG-lHy"
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
```

<!-- #region colab_type="text" id="UePztaPP1VY_" -->
### <font color='blue'> Synthesis Noisy imaage </font>
<!-- #endregion -->

```python colab={} colab_type="code" id="ILabWz8f-ofI"
noise_factor=.5
x_train_noicy=x_train+noise_factor*np.random.normal(loc=0,scale=1.0,size=x_train.shape)
x_test_noicy=x_test+noise_factor*np.random.normal(loc=0,scale=1.0,size=x_test.shape)
x_train_noicy=np.clip(x_train_noicy,0.,1.)
x_test_noicy=np.clip(x_test_noicy,0.,1.)
```

<!-- #region colab_type="text" id="gXj81CXd3uxp" -->
### <font color='blue'> Print the shape of the datasets</font>


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="iki6G1cJ-sMq" outputId="567cd5f9-aa0d-4700-a94a-8dbcdb61bc87"
print("shape of x_train:",x_train.shape)
print("shape of y_train:",y_train.shape)
print("shape of x_test:",x_test.shape)
print("shape of y_test:",y_test.shape)
```

<!-- #region colab_type="text" id="RWhYR8S238Co" -->
### <font color="blue"> Reshape the data </font>
<!-- #endregion -->

```python colab={} colab_type="code" id="Q1gjXjsA-vd-"
x_train_noicy = np.expand_dims(x_train_noicy, -1) # Make sure images have shape (28, 28, 1)
x_test_noicy = np.expand_dims(x_test_noicy, -1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="b9SXnh2N-6HZ" outputId="4e808984-f93c-48f7-ee7a-841366d26ee0"
print("shape of x_train:",x_train_noicy.shape)
print("shape of y_train:",y_train.shape)
print("shape of x_test:",x_test.shape)
print("shape of y_test:",y_test.shape)
```

<!-- #region colab_type="text" id="GXUNYYlw4A31" -->
### <font color='blue'> Create a binary class matrix </font>


<!-- #endregion -->

```python colab={} colab_type="code" id="dDL9IKKH-91g"
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

<!-- #region colab_type="text" id="7Z12R-1P4G8b" -->
### <font color= 'blue'>Encoder <font>
<!-- #endregion -->

```python colab={} colab_type="code" id="PdyE-MD891It"
input_img = layers.Input(shape=(28, 28, 1))  
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
```

<!-- #region colab_type="text" id="LPMXpBpQ4TqO" -->
### <font color='blue'> Decoder 
<!-- #endregion -->

```python colab={} colab_type="code" id="Vjcb4sSF7H2U"
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

```

<!-- #region colab_type="text" id="6ffMtbZJ4bZ8" -->
### <font color='blue'> Model 
<!-- #endregion -->

```python colab={} colab_type="code" id="jUHglwAB_BN5"
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss='binary_crossentropy')
```

<!-- #region colab_type="text" id="WvCvpRXd1od-" -->
### <font color='blue'> Train the model </font>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="Uh918nnl7iuf" outputId="a0c12e8e-e11f-41fb-e1d0-d788eab4e2c5"
autoencoder.fit(x_train_noicy, x_train,
                epochs=50,
                batch_size=128)
```

<!-- #region colab_type="text" id="rsxZvX8x13TI" -->
### <font color='blue'> Denoise the test set </font>
<!-- #endregion -->

```python colab={} colab_type="code" id="-gIpQSdU-mP4"
predicted=autoencoder.predict(x_test_noicy)
```

<!-- #region colab_type="text" id="Ac6P553T4t49" -->
### <font color='blue'> Image visualization </font>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 499} colab_type="code" id="uFIMhZoa-enO" outputId="90ca3c13-6079-41e6-bbe8-12c85b129991"
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([]) # to remove x-axis  the [] empty list indicates this
    plt.yticks([]) # to remove y-axis
    plt.grid(False) # to remove grid
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray') #display the image 
    plt.title('Original Image')
plt.tight_layout() # to have a proper space in the subplots
plt.show()

plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([]) # to remove x-axis  the [] empty list indicates this
    plt.yticks([]) # to remove y-axis
    plt.grid(False) # to remove grid
    plt.imshow(x_test_noicy[i].reshape(28, 28), cmap='gray') #display the image
    plt.title('Noisy Image') 
plt.tight_layout() # to have a proper space in the subplots
plt.show()

# to visualize reconstructed images(output of autoencoder)
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([]) # to remove x-axis  the [] empty list indicates this
    plt.yticks([]) # to remove y-axis
    plt.grid(False) # to remove grid
    plt.imshow(predicted[i].reshape(28, 28), cmap='gray') #display the image 
    plt.title('Denoised Image')
plt.tight_layout() # to have a proper space in the subplots
plt.show()
```

<!-- #region colab_type="text" id="LBFbaUAU4_eb" -->
### <font color='blue'> Denoised Input </font>
<!-- #endregion -->

```python colab={} colab_type="code" id="Mk2Zkz9EGAzS"
train_denoised = autoencoder.predict(x_train_noicy)
test_denoised =autoencoder.predict(x_test_noicy)
```

<!-- #region colab_type="text" id="DgoST0PL5P-s" -->
### <font color='blue' > Digits Classifier </font>


<!-- #endregion -->

```python colab={} colab_type="code" id="pmQR-BOhHIG1"
model = keras.Sequential()
```

```python colab={} colab_type="code" id="gtEypgCoHa0Y"
model.add(keras.Input(shape=input_shape))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))
```

<!-- #region colab_type="text" id="jkmtYD8H5npF" -->
### <font color='blue'> Model summary </font>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 391} colab_type="code" id="jtOwGU6OHfVP" outputId="50352b22-a39b-4f53-b790-512417bda78a"
model.summary()
```

```python colab={} colab_type="code" id="25eerg0KHyY2"
batch_size = 128
epochs = 15
```

```python colab={} colab_type="code" id="W4uZa3aJH4Hk"
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

<!-- #region colab_type="text" id="oOaWwr4O3lWs" -->
### <font color='blue'> Train the image classifier </font>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 544} colab_type="code" id="zcZ1KIiCHmU6" outputId="1ea36e6b-023c-4753-e62b-634388b47524"
model.fit(train_renoised, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="h2iyFd0HMtBl" outputId="f8243905-0b48-496d-b556-8a6a54f9b212"
score = model.evaluate(test_renoised, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```

<!-- #region colab_type="text" id="aqBF4MfuH1IY" -->
[Reference](https://blog.keras.io/building-autoencoders-in-keras.html)
<!-- #endregion -->

```python colab={} colab_type="code" id="XhsRK_cSH7UZ"

```
