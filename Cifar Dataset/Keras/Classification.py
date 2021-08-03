import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from Model_CNN import MyModel

seed_no = 2021
rng = np.random.default_rng(seed_no)

#download fifar dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.33, random_state=seed_no)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
output_classes = len(class_names)

#Pre-processing the dataset
train_images, val_images = train_images/255.0, val_images/255.0

train_labels_encode = tf.one_hot(train_labels.reshape(train_labels.shape[0]), output_classes)
val_labels_encode = tf.one_hot(val_labels.reshape(val_labels.shape[0]), output_classes)


#Convertring dataset into tensor dataset
batch_size = 254
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels_encode))
dataset = dataset.shuffle( 1024 ).batch( batch_size )

dataset_val = tf.data.Dataset.from_tensor_slices((val_images, val_labels_encode))
dataset_val = dataset_val.shuffle( 1024 ).batch( batch_size )

######################### Modeling ######################
# Create an instance of the model
model = MyModel(output_classes, droput_rate = 0.4)


#Model hyperparameters
num_epochs = 20
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

train_summary_writer = tf.summary.create_file_writer('Summaries/train')

@tf.function
def train_step(images, labels):
    labels = tf.argmax(labels, 1)
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = model(images, training=True)
      loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
    train_loss(loss)
    train_accuracy(labels, predictions)
  
  
@tf.function
def test_step(images, labels):
    labels = tf.argmax(labels, 1)
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    

#Training
for epoch in range(num_epochs):
  with train_summary_writer.as_default():
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    
    for batch in dataset:
        images = batch[0]
        labels = batch[1]
        train_step(images, labels)
    
    for test_batch in dataset_val:
        images = test_batch[0]
        labels = test_batch[1]
        test_step(images, labels)
        
    print("Epoch : {:d} of {:d}  Loss : {:.4f}  Accuracy : {:.2%} Test Loss : {:.4f} Test Accuracy : {:.2%}"
          .format(epoch+1,num_epochs,train_loss.result(),train_accuracy.result(),test_loss.result(),test_accuracy.result())) 

#Save the model
tf.saved_model.save(model, './saved_model')

######################## Testing #####################
test_images = test_images/255.0
test_labels_encode = tf.one_hot(test_labels.reshape(test_labels.shape[0]), output_classes)

#load the saved model
loaded_model = tf.saved_model.load('./saved_model')

logit = loaded_model(tf.convert_to_tensor(test_images, dtype=tf.float32), training = False)
logit = tf.nn.softmax(logit)

for i in range(10):
    print("This image most likely belongs to {} with a {:.2f} percent confidence and actually it belongs to {}."
          .format(class_names[np.argmax(logit[i])], 100 * np.max(logit[i]), class_names[test_labels[i][0]]))
