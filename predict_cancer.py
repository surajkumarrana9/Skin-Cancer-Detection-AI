import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "skin_cancer_model.keras"
TEST_IMAGE = ROOT_DIR / "test.jpg"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# 1. Model load karo
model = tf.keras.models.load_model(MODEL_PATH)

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

# Use the repository sample image by default. Pass another path by importing
# predict_skin_issue from a separate script when testing a different image.
predict_skin_issue(TEST_IMAGE)
