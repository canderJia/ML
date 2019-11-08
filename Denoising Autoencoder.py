
import numpy as np
from keras.layers import Input  
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Sequential
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from keras.datasets import mnist

#load data
img_rows, img_cols = 28, 28
input_shape = [28,28,1]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.double(np.reshape(x_train, (len(x_train), 28, 28, 1)))/255
x_test = np.double(np.reshape(x_test, (len(x_test), 28, 28, 1)))/255

# add noisy
noise_factor = 0.3
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#show the original image and damaged image
n = 10
plt.figure(figsize=(20, 4))
b=[2,0,1,8,2,1,2,1,0,0,2,0]
n=len(b)
for j in range(len(b)):
# display original
	for i in range(10000):
		if b[j]==y_test[i]:
    			ax = plt.subplot(2, n, j + 1)
    			plt.imshow(x_test[i].reshape(28, 28))
    			plt.gray()
    			ax.get_xaxis().set_visible(False)
    			ax.get_yaxis().set_visible(False)


    			ax = plt.subplot(2, n, j + 1 + n)
    			plt.imshow(x_test_noisy[i].reshape(28, 28))
    			plt.gray()
    			ax.get_xaxis().set_visible(False)
    			ax.get_yaxis().set_visible(False)
			break
plt.show()


# construct an Denosing AutoEncoder which is different from demo1

###############################################################
auto_encoder = Sequential()
auto_encoder.add(Conv2D(15, (3, 3), activation='relu', padding='same',input_shape=input_shape))  # (?, 28, 28, 32)
auto_encoder.add(MaxPooling2D((2, 2), padding='same'))  # (?, 14, 14, 32)
auto_encoder.add(Conv2D(10, (3, 3), activation='relu', padding='same',input_shape=input_shape))  # (?, 28, 28, 32)
auto_encoder.add(MaxPooling2D((2, 2), padding='same'))  # (?, 14, 14, 32)
auto_encoder.add(UpSampling2D((2, 2)))  # (?, 28, 28, 32)
auto_encoder.add(Conv2D(10, (3, 3), activation='relu', padding='same',input_shape=input_shape))  # (?, 28, 28, 32)
auto_encoder.add(UpSampling2D((2, 2)))  # (?, 28, 28, 32)
auto_encoder.add(Conv2D(15, (3, 3), activation='relu', padding='same')) # (?, 28, 28, 32)
auto_encoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same')) # (?, 28, 28, 1)
###############################################################

#model compile: optimizer, loss function
auto_encoder.compile(optimizer='sgd', loss='mean_squared_error')

#model summary
auto_encoder.summary()
auto_encoder.load_weights('denoise_epoch_3_5.h5')

#train the model
auto_encoder.fit(x_train_noisy, x_train,  
                 epochs=2,
                 batch_size=128,
                 shuffle=True,
                 validation_data=(x_test_noisy, x_test)) 

#obtain the model predictions in testing dataset
decoded_imgs_noise = auto_encoder.predict(x_test_noisy)  
decoded_imgs = auto_encoder.predict(x_test)  
#save the model
auto_encoder.save('denoise_epoch_3_7.h5')
n = 10
plt.figure(figsize=(20, 4))
b=[2,0,1,8,2,1,2,1,0,0,2,0]
n=len(b)
for j in range(len(b)):
# display original
	for i in range(10000):
		if b[j]==y_test[i]:
    			ax = plt.subplot(3, n, j + 1)
    			plt.imshow(x_test_noisy[i].reshape(28, 28))
    			plt.gray()
    			ax.get_xaxis().set_visible(False)
    			ax.get_yaxis().set_visible(False)

    # display reconstruction
    			ax = plt.subplot(3, n, j + 1 + n)
    			plt.imshow(decoded_imgs_noise[i].reshape(28, 28))
    			plt.gray()
    			ax.get_xaxis().set_visible(False)
    			ax.get_yaxis().set_visible(False)
			
  			ax = plt.subplot(3, n, j + 1 + n + n)
    			plt.imshow(decoded_imgs[i].reshape(28, 28))
    			plt.gray()
    			ax.get_xaxis().set_visible(False)
    			ax.get_yaxis().set_visible(False)
			break
plt.show()
