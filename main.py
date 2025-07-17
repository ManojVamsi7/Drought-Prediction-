# main.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from keras.models import Sequential, load_model # Import load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
from keras import backend as K # Import Keras backend for session management

# Suppress TensorFlow FutureWarnings and Keras Backend warnings for cleaner output
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except AttributeError:
    pass

app = Flask(__name__, static_folder='static')

# Global variables to store the model, TensorFlow graph, and session
model = None
graph = None
tf_session = None # Add this line to store the TensorFlow session

def load_model():
    """Load the trained DCNN model"""
    global model, graph, tf_session # Declare all as global

    # --- Step 1: Create a new TensorFlow session and set it as default for Keras ---
    tf.compat.v1.reset_default_graph() # Clear any existing default graph
    tf_session = tf.compat.v1.Session() # Create a new session
    K.set_session(tf_session) # Set this session as the default Keras session
    
    graph = tf.compat.v1.get_default_graph() # Capture the default graph of this new session

    # Define the same model architecture as in your script
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=4, activation='softmax')) # 4 output classes

    # Load weights if available
    weights_path = "model/model_weights.hdf5"
    if os.path.exists(weights_path):
        with tf_session.as_default(): # Ensure operations (like loading weights) happen in this session
            with graph.as_default(): # And in this graph
                try:
                    # Keras.models.load_model can load architecture and weights
                    # but if you're loading weights into a pre-defined Sequential model,
                    # model.load_weights() is correct.
                    model.load_weights(weights_path)
                    
                    # It's important to compile the model *after* loading weights
                    # to ensure all layers are properly built with the loaded weights
                    # and attached to the current graph/session.
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    print("Model loaded successfully")

                except Exception as e:
                    print(f"Error loading model weights from {weights_path}: {e}")
                    model = None
                    tf_session.close() # Close session if error
                    tf_session = None
                    graph = None
    else:
        print(f"Warning: model weights file not found at {weights_path}. Predictions will not be accurate.")
        model = None
        tf_session.close() # Close session if no weights
        tf_session = None
        graph = None


@app.route('/')
def index():
    """Serve the main page"""
    return render_template("index.html")

@app.route('/team')
def team():
    """Serve the team page"""
    return render_template("team.html")

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to handle image prediction"""
    if 'image' not in request.files:
        print("No 'image' file part in the request.")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        print("Uploaded file has no filename.")
        return jsonify({'error': 'No selected image file'}), 400

    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    temp_path = os.path.join(upload_folder, file.filename)

    try:
        file.save(temp_path)
        print(f"File saved to: {temp_path}")

        image = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"cv2.imread returned None for file: {temp_path}")
            return jsonify({'error': 'Invalid image file or unable to read'}), 400

        if len(image.shape) == 2: # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # Convert to BGR first
        
        if image.shape[2] == 3: # If it's BGR, convert to BGRA
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] == 4: # If it's already BGRA, use as is
            pass
        else:
            print(f"Unexpected number of channels: {image.shape[2]}")
            return jsonify({'error': 'Unsupported image format (expected 3 or 4 channels)'}), 400
            
        img_resized = cv2.resize(image, (28, 28))
        
        input_image_for_model = np.expand_dims(img_resized, axis=0)
        input_image_for_model = input_image_for_model.astype('float32')
        input_image_for_model = input_image_for_model / 255.0

        # --- CRUCIAL CHANGES HERE FOR SESSION/GRAPH ---
        if model is not None and graph is not None and tf_session is not None:
            with graph.as_default(): # Activate the stored graph
                with tf_session.as_default(): # Activate the stored session
                    preds = model.predict(input_image_for_model)
                    predict_class = np.argmax(preds)
                    
                    class_labels = {
                        0: "Drought Detected (Class 0)",
                        1: "Drought Detected (Class 1)",
                        2: "No Drought Detected (Class 2)",
                        3: "No Drought Detected (Class 3)"
                    }
                    predicted_label = class_labels.get(predict_class, "Unknown Class")
                    is_drought = predict_class in [0, 1]
                    
                    return jsonify({
                        'prediction': 'Drought Detected' if is_drought else 'No Drought Detected',
                        'class_index': int(predict_class),
                        'class_label': predicted_label,
                        'confidence': float(preds[0][predict_class])
                    })
        else:
            print("Model, TensorFlow graph, or session is not loaded, indicating a critical loading failure.")
            return jsonify({'error': 'Model, graph, or session not loaded on server. Check server logs.'}), 500
    
    except Exception as e:
        print(f"An error occurred during prediction processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error during prediction: {str(e)}'}), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temporary file: {temp_path}")

if __name__ == '__main__':
    # Load the model before starting the app
    load_model()
    
    os.makedirs('model', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)