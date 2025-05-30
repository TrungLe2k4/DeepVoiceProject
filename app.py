# app.py
from flask import Flask, request, render_template, redirect, url_for
import os
from Predictions.predict_three_models import predict_file, save_spectrogram_image  # dùng hàm bạn đã có

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('audio_file')
        if not file or file.filename == '':
            return "No file selected", 400
        if not file.filename.lower().endswith('.wav'):
            return "Only WAV files are allowed", 400

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(save_path)

        # Tạo ảnh spectrogram
        spectro_folder = os.path.join('static', 'spectrograms')
        os.makedirs(spectro_folder, exist_ok=True)
        spectro_path = os.path.join(spectro_folder, file.filename + '.png')
        save_spectrogram_image(save_path, spectro_path)

        # Dự đoán
        results = predict_file(save_path)

        # Truyền đường dẫn ảnh spectrogram cho template
        spectro_url = url_for('static', filename=f'spectrograms/{file.filename}.png')

        return render_template('result.html', filename=file.filename, results=results, spectro_url=spectro_url)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
