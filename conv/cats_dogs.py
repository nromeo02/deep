import matplotlib.pyplot as plt

from keras import Sequential, preprocessing, models, layers, optimizers, losses, mixed_precision

DATASET_PATH = "/home/user/Desktop/UNI/DEEP/lab4/dataset_fixed/PetImages"

# Hardware specific optimization.
HW_OPTIMIZATION = True

# Hyperparameters.
STARTING_NEURONS = 16
NUM_LAYERS = 4
LEARNING_RATE = 0.001
DROPOUT = 0.3
KERNEL_SIZE = (3, 3)
EPOCHS = 40

# Dataset parameters.
SEED = 1337
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 128

if HW_OPTIMIZATION:
    mixed_precision.set_global_policy('mixed_float16')

# Load dataset.
train_ds = preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

num_classes = len(train_ds.class_names)

# Build the model.
model = models.Sequential()
model.add(Sequential([
    layers.RandomFlip("horizontal", input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
]))
model.add(layers.Rescaling(1. / 255.))

for layer in range(NUM_LAYERS):
    model.add(layers.Conv2D(STARTING_NEURONS * pow(2, layer), KERNEL_SIZE, activation = 'relu'))
    model.add(layers.MaxPool2D((2, 2))) # Halve the data.

model.add(layers.Flatten())
model.add(layers.Dense(STARTING_NEURONS * pow(2, NUM_LAYERS - 1), activation = 'relu'))
model.add(layers.Dropout(DROPOUT))

if HW_OPTIMIZATION:
    model.add(layers.Dense(num_classes, activation = 'softmax', dtype='float32'))
else:
    model.add(layers.Dense(num_classes, activation = 'softmax'))

model.compile(optimizer = optimizers.Adam(LEARNING_RATE),
        loss = losses.SparseCategoricalCrossentropy(),
        metrics = ['accuracy'])

model.summary()

# Train the model.
h = model.fit(train_ds, validation_data = val_ds, epochs = EPOCHS)

# Plot the loss and the accuracy.
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(h.history['loss'])
ax[0].plot(h.history['val_loss'])
ax[0].set_title('Model loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Val'], loc='upper right')

ax[1].plot(h.history['accuracy'])
ax[1].plot(h.history['val_accuracy'])
ax[1].set_title('Model accuracy')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Val'], loc='lower right')

plt.show()