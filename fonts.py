import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

# Font Matching and Detection
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_features(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 100:
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            hull = cv2.convexHull(approx)
            feature = np.squeeze(hull)
            features.append(feature)
    return features

def train_font_model(train_data_path):
    font_features = []
    font_labels = []
    for font_dir in os.listdir(train_data_path):
        font_path = os.path.join(train_data_path, font_dir)
        for image_file in os.listdir(font_path):
            image_path = os.path.join(font_path, image_file)
            image = preprocess_image(image_path)
            features = extract_features(image)
            font_features.extend(features)
            font_labels.extend([font_dir] * len(features))
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(font_features, font_labels)
    return neigh

def detect_fonts(image_path, font_model):
    image = preprocess_image(image_path)
    features = extract_features(image)
    distances, indices = font_model.kneighbors(features)
    detected_fonts = [font_model._y[indices[i][0]] for i in range(len(features))]
    return detected_fonts

if __name__== "__main__":
    # Font Matching and Detection
    train_data_path = "path/to/train/data"
    font_model = train_font_model(train_data_path)
    image_path = "path/to/test/image.jpg"
    detected_fonts = detect_fonts(image_path, font_model)
    print(detected_fonts)

