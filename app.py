# app.py
from flask import Flask, render_template, request, redirect, url_for, abort
import db
from model import load_model_and_tokenizer
from keras.utils import pad_sequences

# Load the model and tokenizer from the model.py file
model, tokenizer = load_model_and_tokenizer()

app = Flask(__name__)

# Define max length (set this as per the model's training config)
max_length = 100

# Prediction function for XSS detection
def predict_xss(payload):
    # Tokenize and pad the input payload
    test_sequences = tokenizer.texts_to_sequences([payload])
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    # Make prediction using the loaded model
    predictions = model.predict(test_padded)
    predicted_class = (predictions > 0.5).astype(int)

    return predicted_class[0][0]  # 1 for XSS, 0 for Not XSS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        comment = request.form['comment']

        # Check if the comment contains XSS
        xss_prediction = predict_xss(comment)
        if xss_prediction == 1:
            # Redirect to the "denied" page when XSS is detected in the comment
            abort(401, description="XSS detected in the comment.")
        # If no XSS detected, add the comment to the database
        db.add_comment(comment)

    # Handle search functionality
    search_query = request.args.get('q')
    
    # Check if the search query contains XSS
    if search_query:
        xss_prediction = predict_xss(search_query)
        if xss_prediction == 1:
            # Redirect to the "denied" page if the search query contains XSS
            abort(401, description="XSS detected in the search query.")
    
    # Retrieve the comments based on the search query
    comments = db.get_comments(search_query)

    return render_template('index.html', comments=comments, search_query=search_query)

# Error handler for 401 Unauthorized
@app.errorhandler(401)
def handle_unauthorized(error):
    # Ensure we pass the error description to the template
    return render_template('error.html', message=error.description), 401

if __name__ == '__main__':
    # Change the port here (for example, to 8000)
    app.run(debug=True, port=5051)
