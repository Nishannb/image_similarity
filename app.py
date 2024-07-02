from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
import io
import numpy as np


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(image1, image2):
    img1 = cv2.imdecode(np.frombuffer(image1, np.uint8), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imdecode(np.frombuffer(image2, np.uint8), cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(img1)
    img2 = imageEncoder(img2)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0]) * 100, 2)
    return score

@app.route('/check_similarity', methods=['POST'])
def similarity():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Missing image file(s)'}), 400
        
        file1 = request.files['image1'].read()
        file2 = request.files['image2'].read()

        score = generateScore(file1, file2)
        return jsonify({'similarity_score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/', methods=['GET'])
def hello():
    return 'Hello'


if __name__ == '__main__':
    app.run(debug=True)