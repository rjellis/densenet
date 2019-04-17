import tensorflow as tf
from model2 import DenseNet

def convert_types(image, label):
  image = tf.cast(image, tf.float32)
  # image = tf.expand_dims(image, -1)
  image /= 255
  return image, label


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    mnist_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    mnist_train = mnist_train.map(convert_types).shuffle(1000).batch(32)
    mnist_test = mnist_test.map(convert_types).batch(32)
    
    model = DenseNet(block_config=(2, 2, 2))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr=1e-1)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    # @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            predictions = model(image)
            predictions = tf.keras.activations.softmax(predictions)
            loss = loss_object(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(label, predictions)

    # @tf.function 
    def test_step(image, label):
        predictions = model(image)
        t_loss = loss_object(label, predictions)
        test_loss(t_loss)
        test_accuracy(label, predictions)

    EPOCHS = 10 
    for epoch in range(EPOCHS):
        for image, label in mnist_train:
            train_step(image, label)
        for test_image, test_label in mnist_test:
            test_step(test_image, test_label)
        
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(
            epoch + 1, 
            train_loss.result(), 
            train_accuracy.result()*100, 
            test_loss.result(), 
            test_accuracy.result()*100))
