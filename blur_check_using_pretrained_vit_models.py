from transformers import AutoModelForImageClassification, ViTImageProcessor #type: ignore
from PIL import Image #type: ignore
import torch #type: ignore

# Loading the pretrained model
model_name = "mansee/vit-base-patch16-224-blur_vs_clean"
device = "mps" if torch.backends.mps.is_available() else "cpu"  # Use MPS if available
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
image_processor = ViTImageProcessor.from_pretrained(model_name)

# Loading the image
image_path = "/Users/gauravkasat/Desktop/Screenshot 2025-01-11 at 10.39.46 AM.png"  # Ensure this path is correct
image = Image.open(image_path).convert("RGB")  # Convert image to RGB

# Preprocesings the image
inputs = image_processor(images=image, return_tensors="pt").to(device)

# Extracting features and Generating predictions m
outputs = model(**inputs)
logits = outputs.logits

# Decoding the predicted class
predicted_class_id = logits.argmax(-1).item()
predicted_class_name = model.config.id2label[predicted_class_id]

# Print the result
print(f"The image is predicted to be: {predicted_class_name}")