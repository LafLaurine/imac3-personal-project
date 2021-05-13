from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

def define_model(SIZE):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 1)))
	model.add(MaxPooling2D((2, 2), padding='same'))
	model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2), padding='same'))
	model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))


	model.add(MaxPooling2D((2, 2), padding='same'))

	model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
	# summarize model
	print(model.summary())
	return model