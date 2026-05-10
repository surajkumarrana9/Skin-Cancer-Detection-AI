import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1. Model load karo
model = tf.keras.models.load_model('models/skin_cancer_model.keras')

# 2. Bimariyon ke naam (Categories)
class_names = [
    'Actinic_keratoses', 'Basal_cell_carcinoma', 'Benign_keratosis_like_lesions',
    'Dermatofibroma', 'Melanoma', 'Melanocytic_nevi', 'Vascular_lesions'
]

def predict_skin_issue(img_path):
    # Image ko prepare karna
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization

    # Prediction
    predictions = model.predict(img_array)
    
    # Sabse zyada probability wala result nikalna
    score = predictions[0]
    result_index = np.argmax(score)
    
    print("\n--- AI Diagnostic Report ---")
    print(f"Top Prediction: {class_names[result_index]}")
    print(f"Confidence: {100 * score[result_index]:.2f}%")
    print("----------------------------\n")

# Yahan apni test image ka path daalo
# test_image = 'data/organized_data/Melanoma/ISIC_0024306.jpg' # Example path
# predict_skin_issue(test_image)

test_image = 'test.jpg'
test_image = r'E:\Skin_Cancer_Detection\data\all_images\ISIC_0024306.jpg'
predict_skin_issue(test_image)
