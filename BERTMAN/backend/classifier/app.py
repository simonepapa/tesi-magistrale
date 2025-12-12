from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import do_label

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/classifier/label-articles', methods=['POST'])
def label_articles():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    jsonFile = data.get("jsonFile")
    quartiere = data.get("quartiere")

    # Validation
    if not jsonFile or not isinstance(jsonFile, list):
        return jsonify({"error": "Invalid input: 'jsonFile' must be a non-empty list"}), 400
    
    if not quartiere or not isinstance(quartiere, str):
        return jsonify({"error": "Invalid input: 'quartiere' must be a non-empty string"}), 400

    required_fields = ["link", "title", "date", "content"]
    for index, article in enumerate(jsonFile):
        if not isinstance(article, dict):
             return jsonify({"error": f"Invalid input: Item at index {index} is not an object"}), 400
        
        missing_fields = [field for field in required_fields if field not in article]
        if missing_fields:
            return jsonify({"error": f"Invalid input: Item at index {index} is missing fields: {', '.join(missing_fields)}"}), 400

    try:
        labeled_articles = do_label(jsonFile, quartiere)
        return jsonify(labeled_articles)
    except Exception as e:
        return jsonify({"error": f"Labeling failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)