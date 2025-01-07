from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt

# Load the model without compiling
model = load_model('model_5.h5', compile=False)
print("Model loaded successfully!")

# Recompile the model with a valid loss function and reduction argument
model.compile(
    optimizer='adam',  # Use the optimizer of your choice
    loss=BinaryCrossentropy(reduction='sum_over_batch_size'),  # Or reduction='none', 'mean', etc.
    metrics=['accuracy']
)
print("Model compiled successfully!")

from PIL import Image
import numpy as np
def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Convert to RGB (3 channels)
    img = img.convert('RGB')  # Ensure 3 color channels

    # Resize to the input size expected by the model
    img = img.resize((256, 256))

    # Convert to a NumPy array
    img_array = np.array(img)

    # Normalize pixel values to range [0, 1]
    img_array = img_array / 255.0

    # Add batch dimension (batch size = 1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(image_path, model):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Predict using the model
    predictions = model.predict(processed_image)

    # Assuming the model is for binary classification (fracture or no fracture)
    fracture_probability = predictions[0][0]  # Extract the probability for the "fracture" class
    if fracture_probability > 0.5:
        return f"Fracture detected with probability: {fracture_probability:.2f}"
    else:
        return f"No fracture detected. Probability: {1 - fracture_probability:.2f}"


def visualize_prediction(image_path, model):
    result = predict_image(image_path, model)
    img = Image.open(image_path)
    
    plt.imshow(img, cmap='gray')  # Use grayscale if applicable
    plt.title(result)
    plt.axis('off')
    plt.show()

# Test the function
image_path = 'image.png'
result = predict_image(image_path, model)
print(result)

visualize_prediction(image_path, model)