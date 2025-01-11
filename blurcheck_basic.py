import numpy as np
import cv2  #type: ignore
import matplotlib.pyplot as plt #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from skimage.filters import sobel, laplace #type: ignore
from scipy.fftpack import fft2, fftshift #type: ignore

# Utility function to load and preprocess images
def load_images(image_paths, color_mode="grayscale"):
    images = []
    for path in image_paths:
        if color_mode == "grayscale":
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        elif color_mode == "rgb":
            img = cv2.imread(path, cv2.IMREAD_COLOR)  # Load as RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        else:
            raise ValueError("Unsupported color_mode. Choose 'grayscale' or 'rgb'.")

        img = cv2.resize(img, (224, 224))  # Resize to uniform size
        images.append(img)

    images = np.array(images)
    if color_mode == "grayscale":
        images = np.expand_dims(images, axis=-1)  # Add channel dimension for grayscale

    return images

# Basic Solution: Variance of Laplacian
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Basic Solution: Edge Strength via Sobel Filter
def edge_strength(image):
    sobel_edges = sobel(image)
    return np.sum(sobel_edges)

# Intermediate Solution: Frequency Domain Analysis
def high_frequency_energy(image):
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    low_freq_energy = np.sum(magnitude_spectrum[:112, :112])
    high_freq_energy = np.sum(magnitude_spectrum[112:, 112:])
    return high_freq_energy / (low_freq_energy + 1e-5)

# Intermediate Solution: Tenengrad Metric
def tenengrad_metric(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.sum(gradient_magnitude)

# Deep Learning: Simple CNN for Blur Detection
def build_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Autoencoder for Blur Detection
def build_autoencoder(input_shape):
    autoencoder = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')
    ])
    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder

# Placeholder for dataset
image_paths = ["/Users/gauravkasat/Desktop/Screenshot 2025-01-11 at 10.39.46 AM.png", "/Users/gauravkasat/Desktop/Screenshot 2025-01-11 at 10.39.37 AM.png"]  # Replace with actual paths
color_mode = "grayscale"  # Options: "grayscale" or "rgb"
images = load_images(image_paths, color_mode=color_mode)

# Laplacian Output
laplacian_variances = [variance_of_laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if color_mode == "rgb" else img) for img in images]
print("Laplacian Variances:", laplacian_variances)

# Edge Strength Output
edge_strengths = [edge_strength(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if color_mode == "rgb" else img) for img in images]
print("Edge Strengths:", edge_strengths)

# Frequency Domain Analysis
freq_energies = [high_frequency_energy(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if color_mode == "rgb" else img) for img in images]
print("Frequency Energies:", freq_energies)

# Tenengrad Metric Output
tenengrad_scores = [tenengrad_metric(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if color_mode == "rgb" else img) for img in images]
print("Tenengrad Scores:", tenengrad_scores)

# Data preparation
images = images / 255.0  # Normalize to [0, 1]
labels = np.array([0, 1])  # Example labels: 0 for sharp, 1 for blurry

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
input_shape = X_train.shape[1:]

# CNN TRAINING
cnn_model = build_cnn(input_shape)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16, callbacks=[early_stopping])

# Autoencoder training
autoencoder_model = build_autoencoder(input_shape)
autoencoder_model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=20, batch_size=16, callbacks=[early_stopping])

# Detect Blurry Images Using CNN
def detect_blur_cnn(model, images):
    predictions = model.predict(images)
    return ["Blurry" if pred > 0.5 else "Sharp" for pred in predictions]

# Detect Blurry Images Using Autoencoder
def detect_blur_autoencoder(model, images):
    reconstructed = model.predict(images)
    reconstruction_errors = np.mean((images - reconstructed) ** 2, axis=(1, 2, 3))
    threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)  # Example threshold
    return ["Blurry" if error > threshold else "Sharp" for error in reconstruction_errors]


cnn_predictions = detect_blur_cnn(cnn_model, X_val)
print("CNN Predictions:", cnn_predictions)

autoencoder_predictions = detect_blur_autoencoder(autoencoder_model, X_val)
print("Autoencoder Predictions:", autoencoder_predictions)