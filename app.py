import os
import time
import asyncio
from asyncio import WindowsSelectorEventLoopPolicy

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import BartModel, BartTokenizer
import torchvision.models as models
import torchaudio
from PIL import Image
import g4f

# ------------------------------
# Configuration and Flask Setup
# ------------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this for production

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Set the event loop policy for Windows (if needed)
asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

# ------------------------------
# 1. Define the Audio Model (DeepSpeech2)
# ------------------------------
class DeepSpeech2AudioModel(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, conv_out_channels=32, 
                 rnn_hidden_size=256, num_rnn_layers=3, bidirectional=True, output_dim=128):
        """
        A DeepSpeech2-inspired model:
         - Computes a mel-spectrogram,
         - Processes it through two convolutional layers,
         - Feeds the output to a multi-layer bidirectional GRU,
         - Pools over time and projects to a fixed-dimension embedding.
        """
        super(DeepSpeech2AudioModel, self).__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU()
        )
        self.rnn_input_size = conv_out_channels * (n_mels // 4)
        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=rnn_hidden_size,
                          num_layers=num_rnn_layers,
                          batch_first=True,
                          bidirectional=bidirectional)
        rnn_output_dim = rnn_hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_output_dim, output_dim)
        
    def forward(self, waveform):
        mel = self.melspec(waveform)         # [batch, n_mels, time]
        mel = mel.unsqueeze(1)               # [batch, 1, n_mels, time]
        conv_out = self.conv(mel)            # [batch, conv_out_channels, new_n_mels, new_time]
        batch_size, channels, freq, time_steps = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2)  # [batch, new_time, channels, freq]
        conv_out = conv_out.contiguous().view(batch_size, time_steps, -1)  # [batch, new_time, channels*freq]
        rnn_out, _ = self.rnn(conv_out)        # [batch, new_time, rnn_output_dim]
        pooled = rnn_out.mean(dim=1)           # [batch, rnn_output_dim]
        out = self.fc(pooled)                  # [batch, output_dim]
        return out

# ------------------------------
# 2. Define the Multimodal Classifier
# ------------------------------
class MultiModalClassifier(nn.Module):
    def __init__(self, text_model, image_model, audio_model,
                 text_feat_dim, image_feat_dim, audio_feat_dim,
                 hidden_dim, num_classes):
        """
        Combines text, image, and audio encoders. Each branch is projected
        into a common hidden space and the features are averaged (if more than one
        modality is provided) before classification.
        """
        super(MultiModalClassifier, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.audio_model = audio_model
        
        self.text_fc = nn.Linear(text_feat_dim, hidden_dim)
        self.image_fc = nn.Linear(image_feat_dim, hidden_dim)
        self.audio_fc = nn.Linear(audio_feat_dim, hidden_dim)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, text_input=None, image_input=None, audio_input=None):
        features = None
        modality_count = 0
        
        if text_input is not None:
            text_input_filtered = {k: v for k, v in text_input.items() if k != "labels"}
            text_outputs = self.text_model(**text_input_filtered)
            pooled_text = text_outputs.last_hidden_state.mean(dim=1)
            text_features = self.text_fc(pooled_text)
            features = text_features if features is None else features + text_features
            modality_count += 1
            
        if image_input is not None:
            image_features = self.image_model(image_input)
            image_features = self.image_fc(image_features)
            features = image_features if features is None else features + image_features
            modality_count += 1
            
        if audio_input is not None:
            audio_features = self.audio_model(audio_input)
            audio_features = self.audio_fc(audio_features)
            features = audio_features if features is None else features + audio_features
            modality_count += 1
            
        if modality_count > 1:
            features = features / modality_count
            
        logits = self.classifier(features)
        return logits

# ------------------------------
# 3. Define Inference Functions
# ------------------------------
def inference_text(model, tokenizer, text, device, max_length=128):
    model.eval()
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=None, audio_input=None)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

def inference_image(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(text_input=None, image_input=image, audio_input=None)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

def inference_audio(model, audio_path, device, sample_rate=16000):
    model.eval()
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(text_input=None, image_input=None, audio_input=waveform)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

def inference_all(model, tokenizer, text, image_path, audio_path, transform, device, sample_rate=16000, max_length=128):
    model.eval()
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=image, audio_input=waveform)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

# ------------------------------
# 4. Set Up the Model and Load Weights
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)

# Text Encoder (BART)
text_encoder = BartModel.from_pretrained(model_name)
text_encoder.to(device)
text_feat_dim = text_encoder.config.d_model

# Image Encoder (ResNet18)
image_encoder = models.resnet18(pretrained=True)
num_img_features = image_encoder.fc.in_features
image_encoder.fc = nn.Identity()
image_encoder.to(device)
image_feat_dim = num_img_features

# Audio Encoder (DeepSpeech2-Inspired)
audio_encoder = DeepSpeech2AudioModel(
    sample_rate=16000, 
    n_mels=128, 
    conv_out_channels=32, 
    rnn_hidden_size=256, 
    num_rnn_layers=3, 
    bidirectional=True, 
    output_dim=128
)
audio_encoder.to(device)
audio_feat_dim = 128

# Label list and mapping (update as needed)
label_list = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy", "Leaf Blast", 
    "Leaf Blight", "Leaf Scald", "Leaf Smut", "Narrow Brown Spot"
]
id2label = {idx: label for idx, label in enumerate(label_list)}
num_classes = len(label_list)

# Instantiate the Multimodal Classifier
model = MultiModalClassifier(
    text_model=text_encoder,
    image_model=image_encoder,
    audio_model=audio_encoder,
    text_feat_dim=text_feat_dim,
    image_feat_dim=image_feat_dim,
    audio_feat_dim=audio_feat_dim,
    hidden_dim=512,
    num_classes=num_classes
)

MODEL_SAVE_PATH = "multimodal_model.pth"
if not os.path.exists(MODEL_SAVE_PATH):
    raise FileNotFoundError(f"Saved model not found at {MODEL_SAVE_PATH}")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()

# ------------------------------
# 5. Define Transforms for Image Inference
# ------------------------------
image_inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------------------
# 6. Flask Routes and File Upload Handling
# ------------------------------

# Landing page
@app.route("/")
def landing():
    return render_template("landing.html")

# Index page with form
@app.route("/index")
def index():
    return render_template("index.html")

# Route to serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Route to download the suggestion report
@app.route("/download_report")
def download_report():
    report_filename = "suggestion_report.txt"
    return send_from_directory(app.config["UPLOAD_FOLDER"], report_filename, as_attachment=True)

# Process form submission and display results
@app.route("/result", methods=["POST"])
def result():
    # Get text input
    text_input = request.form.get("text_input")
    if not text_input:
        flash("Text input is required.")
        return redirect(url_for("index"))
    
    # Get uploaded files
    image_file = request.files.get("image_file")
    audio_file = request.files.get("audio_file")
    
    if not image_file or not audio_file:
        flash("Both image and audio files are required.")
        return redirect(url_for("index"))
    
    # Save the uploaded files
    image_filename = secure_filename(image_file.filename)
    audio_filename = secure_filename(audio_file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
    image_file.save(image_path)
    audio_file.save(audio_path)
    
    # Run inference on all modalities
    predicted_disease = inference_all(model, tokenizer, text_input, image_path, audio_path, image_inference_transform, device)
    
    # Generate suggestion report via GPT-4
    def generate_response(user_input):
        try:
            response = g4f.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": user_input}],
                temperature=0.6,
                top_p=0.9
            )
            return response.strip() if response else "No response generated."
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_suggestion_report(disease_label):
        prompt = (
            f"Provide a detailed, step-by-step suggestion report for the plant disease '{disease_label}' detected. "
            "The report should have the following sections with clear headings:\n"
            "1. What it is: Explain what the disease is in simple terms.\n"
            "2. Why it occurs: Describe the causes and contributing factors for the disease.\n"
            "3. How to overcome: Provide a step-by-step guide on how to manage and overcome the disease.\n"
            "4. Fertilizer Recommendations: Suggest the type of fertilizer and application methods suitable for this condition.\n"
            "Please provide the report in both English and Tamil, with each section written in both languages."
        )
        return generate_response(prompt)
    
    suggestion_report = generate_suggestion_report(predicted_disease)
    
    # Save the suggestion report for download
    report_filename = "suggestion_report.txt"
    report_path = os.path.join(app.config["UPLOAD_FOLDER"], report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(suggestion_report)
    
    # Render the result page with predicted disease, suggestion report, and audio file URL
    return render_template("result.html",
                           predicted_disease=predicted_disease,
                           suggestion_report=suggestion_report,
                           audio_filename=audio_filename)

# ------------------------------
# 7. Chatbot Route (Plant-Focused)
# ------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided."}), 400
    
    # Enforce that the chatbot only talks about plants and plant diseases.
    prompt = (
        f"You are a helpful chatbot specializing in plants and plant diseases. "
        f"Please ensure your responses are strictly about plants, gardening, and related topics. "
        f"The user said: {user_message}"
    )
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.9
        )
        response_text = response.strip() if response else "No response generated."
    except Exception as e:
        response_text = f"Error generating response: {e}"
    
    return jsonify({"response": response_text})

# ------------------------------
# Run the Flask App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
