#Image Data Generator: Make sure your test_data_keras generator is correctly set up and that it can yield images for display.
#Class Labels: Adjust the class labels in the titles according to how your model interprets the labels (e.g., 0 for benign, 1 for malignant).
#Batch Size: Ensure that num_images is appropriate for the batch size of your data generator.

#model.summary() can be used over multiple epochs
print(f"Test loss: {score[0]}%")
print(f"Test accuracy: {score[1]*100}%")

# Plotting accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Function to display images with predictions
def display_images_with_predictions(datagen, model, num_images=5):
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        # Get a batch of test data
        x, y = next(datagen)
        
        # Make predictions
        predictions = model.predict(x)
        
        # Convert predictions to class labels
        predicted_classes = np.argmax(predictions, axis=1)
        
        for j in range(len(x)):
            plt.subplot(num_images, 2, 2 * i + 1)
            plt.imshow(x[j])  # Display image
            plt.title(f'Predicted: {"Malignant" if predicted_classes[j] == 1 else "Benign"}')
            plt.axis('off')

            plt.subplot(num_images, 2, 2 * i + 2)
            plt.imshow(x[j])  # Display image again (or use any other example)
            plt.title(f'Actual: {"Malignant" if np.argmax(y[j]) == 1 else "Benign"}')
            plt.axis('off')
            
    plt.tight_layout()
    plt.show()

# Call the function
display_images_with_predictions(test_data_keras, model)
