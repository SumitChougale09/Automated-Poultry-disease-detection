# classifier.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from model.cbam_model import CBAM, ChannelAttention, SpatialAttention  # Import CBAM components
import cv2
import numpy as np

# Define your classes
class_names = ['Coccidiosis', 'Healthy', 'Newcastle disease', 'Salmonellosis']


def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocess the input image:
    1. Read image
    2. Convert to RGB
    3. Resize to target size
    4. Apply histogram equalization
    5. Apply Gaussian blur
    6. Normalize pixel values
    """
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, target_size)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Split LAB channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels back
        limg = cv2.merge((cl, a, b))
        
        # Convert back to RGB
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Apply Gaussian blur
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")

def load_model():
    # Load the model with CBAM custom layers
    model_path = "model/hybrid_resnet_cbam_model.h5"
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={"CBAM": CBAM, "ChannelAttention": ChannelAttention, "SpatialAttention": SpatialAttention}
    )
    return model

def predict_image(model, image_path):
    # Preprocess the image before prediction
    preprocessed_image = preprocess_image(image_path)
    
    # Predict the class probabilities
    prediction = model.predict(preprocessed_image)
    class_index = tf.argmax(prediction[0]).numpy()
    predicted_class_name = class_names[class_index]
    
    return predicted_class_name
