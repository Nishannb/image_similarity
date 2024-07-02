from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from memory_profiler import profile


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load a lighter image processing model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
model.to(device)
model.eval()

# Image preprocessing function
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to encode an image
def encode_image(img):
    img = Image.open(img).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = model(img_tensor)
    return img_features

# Function to calculate similarity score
@profile
def generateScore(image1, image2):
    try:
        img1 = encode_image(image1)
        img2 = encode_image(image2)
        
        # Calculate cosine similarity
        similarity_score = torch.nn.functional.cosine_similarity(img1, img2).item()
        return round(similarity_score * 100, 2)
    except Exception as e:
        print(f"Error in generateScore: {e}")
        return None

@app.route('/check_similarity', methods=['POST'])
def similarity():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Missing image file(s)'}), 400
        
        file1 = request.files['image1']
        file2 = request.files['image2']

        score = generateScore(file1, file2)
        
        if score is None:
            return jsonify({'error': 'Error processing images'}), 500
        
        return jsonify({'similarity_score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/', methods=['GET'])
def hello():
    return 'Hello'

if __name__ == '__main__':
    app.run(debug=True)
