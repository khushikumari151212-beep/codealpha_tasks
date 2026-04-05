from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data['text']
    src = data['source']
    dest = data['target']

    translated = GoogleTranslator(source=src, target=dest).translate(text)

    return jsonify({'translated_text': translated})

if __name__ == '__main__':
    app.run(debug=True)