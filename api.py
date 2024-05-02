from flask import Flask, request, jsonify
import numpy as np
from star_generator import StarGenerator

#create the app
app = Flask(__name__)

#load the model
model = StarGenerator()
loaded_model, loaded_vectorizer= model.load()

#defining flask route
@app.route('/predict', methods=['GET'])
def predict_stars():
    # Get data from request
    data = request.get_json(force=True)

    # Convert data to numpy array
    input_data = np.array(data['input'])

    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Return the prediction
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)