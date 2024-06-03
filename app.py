from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('nb_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        statement = request.form['news']  # Updated to match the HTML form's 'name' attribute
        processed_statement = preprocess_text(statement)
        vectorized_statement = vectorizer.transform([processed_statement])
        prediction = model.predict(vectorized_statement)
        return render_template('index.html', prediction=prediction[0], statement=statement)

def preprocess_text(text):
    text = text.lower()
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.isalnum()]
    processed_text = ' '.join(filtered_tokens)
    return processed_text

if __name__ == '__main__':
    app.run(port=5001)
