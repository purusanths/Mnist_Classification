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

### <font color="blue"> Import Libraries </font>

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
```

```python
num_classes = 10
input_shape = (28, 28, 1)
```

### <font color='blue'> Download the training data and test data</font>

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

### <font color="blue"> Normalizing the data </font>

```python
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
```

### <font color='blue'> Synthesis noisy image <font>

```python
noise_factor=0.5
x_train_noicy=x_train+noise_factor*np.random.normal(loc=0,scale=1.0,size=x_train.shape)
x_test_noicy=x_test+noise_factor*np.random.normal(loc=0,scale=1.0,size=x_test.shape)
x_train_noicy=np.clip(x_train_noicy,0.,1.)
x_test_noicy=np.clip(x_test_noicy,0.,1.)
```

### <font color='blue'> Print the shape of the datasets </font>

```python
print("shape of x_train:",x_train_noicy.shape)
print("shape of y_train:",x_test_noicy.shape)
print("shape of x_test:",x_train_noicy.shape)
print("shape of y_test:",x_test_noicy.shape)
```

### <font color="blue"> Reshape the data </font>

```python
x_train_noicy = np.expand_dims(x_train_noicy, -1) # Make sure images have shape (28, 28, 1)
x_test_noicy = np.expand_dims(x_test_noicy, -1)
```

### <font color='blue'> Print the shape of the split </font>

```python
print("shape of x_train:",x_train_noicy.shape)
print("shape of y_train:",y_train.shape)
print("shape of x_test:",x_test_noicy.shape)
print("shape of y_test:",y_test.shape)
```

### <font color='blue'> Create a binary class matrix </font>

```python
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

### <font color='blue'> Model architecture </font>

```python
model = keras.Sequential()
```

```python
model.add(keras.Input(shape=input_shape))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))
```

### <font color='blue'> Print the model summary </font>

```python
model.summary()
```

### <font color='blue'> Model hyperparameters </font>

```python
batch_size = 128
epochs = 15
```

### <font color='blue'> Compile the model </font>

```python
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

### <font color='blue'> Train the model </font>

```python
model.fit(x_train_noicy, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
```

### <font color='blue'> Evaluate the model performance of test set</font>

```python
score = model.evaluate(x_test_noicy, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```

### <font color='blue'>Visualization of predicted result</font>

```python
labels=model.predict(x_test_noicy[:5])
```

```python
predicted_label=[np.argmax(y, axis=None, out=None) for y in labels]
```

```python
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([]) # to remove x-axis  the [] empty list indicates this
    plt.yticks([]) # to remove y-axis
    plt.grid(False) # to remove grid
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray') #display the image 
    plt.title("Pred label:"+ str(predicted_label[i]))
plt.tight_layout() # to have a proper space in the subplots
plt.show()
```

### <font color='blue'> Reference1 :https://keras.io/examples/vision/mnist_convnet/ </font>
### <font color='blue'> Reference2 :https://iq.opengenus.org/image-denoising-autoencoder-keras/ </font>

```python

```
