import os
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, multiply, Dropout
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
import timeit

# sets path to dataset
data_path = "C:\\Users\\Brandon\\Desktop\\archive"

# creating data generators for training and testing
# uses ImageDataGenerator from Keras for image augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# uses flow_from_directory method from keras on training folder
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_path, "train"),  # adds train to data_path to specify training folder
                                       # can also use directory="<data-path>
    target_size=(128, 128),  # size of input images
    batch_size=50,  # number of images to be used per batch
    class_mode='binary'  # setting the number of classes to predict "binary" for two "categorical"
                         # can also set to "input" for Autoencoder system for input and output probably the same
                         # shuffle=True, to shuffle order of image being used, otherwise "False"
                         # seed=<int>, random seed for applying random image augmentation and shuffling order of image
)

# uses flow_from_directory method from keras on valid folder for validation testing
val_generator = val_datagen.flow_from_directory(
    os.path.join(data_path, "valid"),
    target_size=(128, 128),
    batch_size=50,
    class_mode='binary',
    shuffle=True
)

# test model
test_generator = val_datagen.flow_from_directory(
    os.path.join(data_path, "test"),
    target_size=(128, 128),
    batch_size=50,
    class_mode='binary',
    shuffle=False
)

#  display sample images from training set with FILENAMES and TRAINING LABELS
class_indices = train_generator.class_indices
# float values in inches
plt.figure(figsize=(5, 5))
for i in range(6):
    img, label = train_generator.next()
    # gets filename of current image
    img_filename = train_generator.filenames[train_generator.batch_index - 1]
    # gets the class name of current image
    img_class = list(class_indices.keys())[list(class_indices.values()).index(label[0].argmax())]
    # takes 3 args that describe layout of figure
    # first = num rows
    # second = num columns
    # third = index of current plot
    plt.subplot(2, 3, i + 1)
    plt.imshow(img[0])
    # generates and formats title to print for each image
    # os.path.basename(img_filename) prints the file name os.path.basename() pulls the base filename from specified path
    # img_class is class name as specified by subdirectory
    plt.title(f"Image: {os.path.basename(img_filename)}\nLabel: {img_class}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# CNN model with Attention Mechanism
input_img = Input(shape=(128, 128, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_img)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.1)(x)

# Attention Layer
attention = Conv2D(1, (1, 1), activation='sigmoid')(x)
x = multiply([x, attention])

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.1)(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

# Attention Layer
attention = Conv2D(1, (1, 1), activation='sigmoid')(x)
x = multiply([x, attention])

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(2, activation='softmax')(x)
model = Model(inputs=input_img, outputs=output)

# compiles model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# prints summary of model
model.summary()


# displays model architecture and generates picture in png format
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)


# model callbacks
# allows customization of training behavior
# allows response to certain events during training
# examples include saving best model, reducing learning rate, and early stopping
checkpoint = ModelCheckpoint("./model/real_fake_classification.keras",
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                              patience=15,
                              min_lr=0)

early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=15,
                               verbose=1,
                               restore_best_weights=True,
                               mode='max')

start_time = timeit.default_timer()

# Begin Training Model
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    # Use checkpoint to save the best model during training.
    callbacks=[checkpoint, reduce_lr, early_stopping]
)
# calculates total execution time
elapsed = timeit.default_timer() - start_time
print("Total time: ", elapsed, "seconds")

# Generate Plots for training and validation loss using history that was returned
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Generate plots for training and validation accuracy using history that was returned
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Display the plots that are generated
plt.show()

# Make predictions using the test set
predictions = model.predict(test_generator)

# Show example of some of the test images along with their true and predicted classification
plt.figure(figsize=(15, 12))
for i in range(6):
    img, true_label = test_generator.next()
    # Sets filename of image
    img_filename = test_generator.filenames[test_generator.batch_index - 1]
    # Sets the predicted class name
    true_class = list(test_generator.class_indices.keys())[list(test_generator.class_indices.values()).index(predictions[i].argmax())]
    predicted_class = list(test_generator.class_indices.keys())[list(test_generator.class_indices.values()).index(predictions[i].argmax())]
    plt.subplot(2, 3, i + 1)
    plt.imshow(img[0])
    plt.title(f"Image: {os.path.basename(img_filename)}\n"
              f"True label: {true_class} \n Predicted Label: {predicted_class}")
    plt.axis('off')
    # Set layout to show images and display images with predictions and actual labels
plt.tight_layout()
plt.show()

# Use the model on the test set to classify the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:0.4f}")
print(f"Test Accuracy: {test_accuracy:0.4f}")
