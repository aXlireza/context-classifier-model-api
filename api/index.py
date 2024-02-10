from flask import Flask, request, jsonify
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow_text as text

CLASS_NAMES = ['negative', 'neutral', 'other', 'positive', 'toxic']

# Initialize Flask app
app = Flask(__name__)

# Name of the directory that holds the model files, relative to this Python file
MODEL_DIR_NAME = "../model"

# Determine the current absolute path of this Python file
current_file_path = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model directory
model_path = os.path.join(current_file_path, MODEL_DIR_NAME)

# Load the model
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]
print(list(model.signatures.keys()))  # Show available signatures

# Assuming your model expects a certain max length for input text
MAX_SEQUENCE_LENGTH = 100

# Initialize tokenizer here (ensure it's the same tokenizer used during training)
# You might need to load it from file if you saved your tokenizer
tokenizer = Tokenizer(num_words=10000)  # Adjust `num_words` as per your training

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text_input = data['text']
    threshold = float(data.get('threshold', 0.5))  # Default threshold is 0.5 if not provided

    # Prepare the input and make a prediction
    predictions = infer(text=tf.constant([text_input]))['classifier']
    sigmoid_output = tf.sigmoid(predictions).numpy()[0]  # Apply sigmoid and convert to numpy

    # Create a response dictionary with scores and labels
    response = {
        'scores': sigmoid_output.tolist(),
        'labels': CLASS_NAMES,
    }

    # Determine the highest scoring label that meets the confidence threshold
    max_score_index = np.argmax(sigmoid_output)
    max_score = sigmoid_output[max_score_index]
    selected_label = None if max_score < threshold else CLASS_NAMES[max_score_index]

    # Update the response with the selected label and its score
    response.update({
        'selected_label': selected_label,
        'selected_score': float(max_score) if selected_label else None,
    })

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
