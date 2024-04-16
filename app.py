from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and data
model = pickle.load(open('automl.pkl', 'rb'))

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request sent by index.html
    data = request.json
    print(data)
    # Convert data to numpy array for prediction
    input_data = np.array(data['input_data']).reshape(1, -1)
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
