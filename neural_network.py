import tensorflow as tf
import tensorflow_datasets as tfds

# Preprocessing

BUFFER = 70_000 # for reshuffling
BATCH = 128
EPOCHS = 20

MNIST_dataset, MNIST_info = tfds.load(name='MNIST', with_info=True, as_supervised=True)

MNIST_train, MNIST_test = MNIST_dataset['train'], MNIST_dataset['test']

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.

    return image, label

train_and_validation_data = MNIST_train.map(scale)
test_data = MNIST_test.map(scale)

num_validation_samples = 0.1 * MNIST_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = MNIST_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

train_and_validation_data = train_and_validation_data.shuffle(BUFFER)

train_data = train_and_validation_data.skip(num_validation_samples)
validation_data = train_and_validation_data.take(num_validation_samples)

train_data = train_data.batch(BATCH)
validation_data = validation_data.batch(num_validation_samples) 
test_data = test_data.batch(num_test_samples)

# Training the Model

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(50, 5, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 
    tf.keras.layers.Conv2D(50, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10) 
])

model.summary(line_length = 75)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',    
    min_delta = 0,
    patience = 2,
    verbose = 0, 
    restore_best_weights = True
)

model.fit(
    train_data, 
    epochs = EPOCHS, 
    callbacks = [early_stopping], 
    validation_data = validation_data,
    verbose = 2
)

# Testing

test_loss, test_accuracy = model.evaluate(test_data)

print('LOSS: {0:.3f}. ACCURACY: {1:.1f}%'.format(test_loss, test_accuracy*100.))
