from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load saved models
model = pickle.load(open('./saved_models/genre_model.pkl', 'rb'))
scaler = pickle.load(open('./saved_models/scaler.pkl', 'rb'))
label_encoder = pickle.load(open('./saved_models/label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure file is uploaded
    if 'file' not in request.files:
        return "No file uploaded!", 400

    uploaded_file = request.files['file']
    # Process CSV features
    features = pd.read_csv(uploaded_file).values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    genre_index = model.predict(features_scaled)[0]
    genre_name = label_encoder.inverse_transform([genre_index])[0]

    return f"The predicted genre is: {genre_name}"

if __name__ == '__main__':
    app.run(debug=True)
