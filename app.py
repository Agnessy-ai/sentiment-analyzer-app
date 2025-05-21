from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the sentiment analysis pipeline
# This will download the model on first run, so it might take a moment.
# We load it once when the app starts to avoid reloading on every request.
try:
    logging.info("Loading sentiment analysis pipeline...")
    # You can specify a model: pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # If no model is specified, it uses a default one.
    sentiment_pipeline = pipeline("sentiment-analysis")
    logging.info("Sentiment analysis pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Error loading sentiment analysis pipeline: {e}")
    sentiment_pipeline = None # Set to None if loading fails

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze():
    """Analyzes the sentiment of the provided text."""
    if not sentiment_pipeline:
        return jsonify({'error': 'Sentiment analysis model not available.'}), 500

    data = request.get_json()
    text_input = data.get('text')

    if not text_input:
        return jsonify({'error': 'No text provided'}), 400

    logging.info(f"Received text for analysis: {text_input[:50]}...") # Log first 50 chars

    try:
        result = sentiment_pipeline(text_input)
        if result:
            # The pipeline returns a list of dictionaries
            # For a single string input, it's a list with one element
            label = result[0]['label']
            score = result[0]['score']
            logging.info(f"Analysis result: Label={label}, Score={score}")
            return jsonify({'sentiment': label.upper(), 'score': score})
        else:
            logging.error("Sentiment pipeline returned an empty result.")
            return jsonify({'error': 'Could not analyze sentiment.'}), 500
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return jsonify({'error': f'Error during analysis: {str(e)}'}), 500

if __name__ == '__main__':
    # Make sure to run in debug mode only for development
    # For production, use a proper WSGI server like Gunicorn or Waitress
    app.run(debug=True, host='0.0.0.0', port=5000)