from flask import Flask, request, send_file, render_template
from PIL import Image
import numpy as np
import io

from process.compress import compress_image
from process.restore import restore_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']
    operation = request.form['operation']

    image = Image.open(file).convert('RGB')
    np_image = np.array(image)

    if operation == 'compress':
        result = compress_image(np_image)
    elif operation == 'restore':
        result = restore_image(np_image)
    else:
        return "Invalid operation", 400

    return send_file(io.BytesIO(result.tobytes()), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
