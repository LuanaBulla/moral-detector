from flask import Flask, render_template, jsonify, request
from detector import MoralDetector  

app = Flask(__name__)

detector = MoralDetector()  # Crea un'istanza del MoralDetector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    txt = request.form['txt']
    result = detector.execution(txt)
    return render_template('index.html', txt=txt, result=result)

if __name__ == '__main__':
    app.run(debug=True)
