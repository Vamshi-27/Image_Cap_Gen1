import streamlit as st
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image

# Custom CSS for styling
st.markdown("""
    <style>
    .uploaded-image {
        border: 2px solid #ddd;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .caption-box {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        font-size: 1.1em;
        color: #333;
    }
    .title {
        font-size: 2em;
        font-weight: bold;
        color: #4A90E2;
    }
    .stButton>button {
            font-size: 1em;
            font-weight: bold;
            color: white;
            background-color: #4CAF50;
            border-radius: 5px;
            padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# App title with custom styling
st.markdown("<div class='title'>Image Caption Generator</div>", unsafe_allow_html=True)

# Sidebar for uploading image
st.sidebar.title("Upload Image")
uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Preprocess the uploaded image
def load_and_preprocess_image_streamlit(image_file, target_size=(224, 224)):
    image = Image.open(image_file)
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image

# Feature extraction with VGG16
def extract_features_from_image_streamlit(model, image):
    feature = model.predict(image, verbose=0)
    return feature

# Convert integer to word
def word_for_id(integer, tokenizer_1):
    for word, index in tokenizer_1.word_index.items():
        if index == integer:
            return word
    return None

# Caption generation
def generate_caption(model, tokenizer_1, features, max_length):
    input_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer_1.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([features, sequence])
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer_1)
        
        if word is None:
            break
        input_text += ' ' + word
        
        if word == 'endseq':
            break
            
    caption = input_text.replace('startseq', '').replace('endseq', '').strip()
    return caption

# Load the VGG16 model and captioning model
base_model = VGG16(include_top=True)
model_vgg = Model(inputs=base_model.inputs, outputs=base_model.get_layer('fc2').output)
model_caption = load_model('Models\\model_epoch_7_keras.keras')

with open('tokenizer_1.pkl', 'rb') as f:
    tokenizer_1 = pickle.load(f)

max_length = 35

# Process and display the uploaded image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = image.resize((300, 200))
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="JPEG", width=300)
    
    # Preprocess and extract features from the image
    processed_image = load_and_preprocess_image_streamlit(uploaded_image)
    features = extract_features_from_image_streamlit(model_vgg, processed_image)
    
    # Generate caption
    caption = generate_caption(model_caption, tokenizer_1, features, max_length)
    
    # Display caption with custom style
    st.markdown(f"<div class='caption-box'><strong>Generated Caption:</strong> {caption}</div>", unsafe_allow_html=True)
