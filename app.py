from flask import Flask, request, redirect,render_template,jsonify,abort,url_for,send_file,flash
import numpy as np
from sklearn.externals import joblib
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import os
import pandas as pd

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    file = FileField(validators=[FileRequired()])

app = Flask(__name__)
model = joblib.load('model.pkl')

app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))

@app.route('/model/<params>')
def model_predict(params):
    try:
        nums = [float(x.strip()) for x in params.split(',')]
        print(nums)
        
        pred = model.predict(np.array(nums).reshape(1,-1))[0]
        return str(pred)
    except Exception as e:
        print(str(e))
        return redirect(url_for('bad_request'))


@app.route('/img')
def show_image():
    link = '/static/iris.jpg'
    return f"<img src='{link}' alt='iris'>"


@app.route('/iris_post',methods=['POST'])
def iris_post():
    content = request.get_json()
    params = content['flower'].split(',')
    try:
        nums = [float(x.strip()) for x in params]
        pred = model.predict(np.array(nums).reshape(1,-1))[0]
        return jsonify({'class':str(pred)})

    except Exception as e:
        print(str(e))
        return redirect(url_for('bad_request'))


@app.route('/badrequest400')
def bad_request():
    return abort(400)


@app.route('/submit', methods=('GET', 'POST'))
def submit():
    form = MyForm()
    if form.validate_on_submit():
        f = form.file.data
        df = pd.read_csv(f,sep=',')
        pred = model.predict(df)
        res = pd.DataFrame(pred,columns=['label'])
        filename = form.name.data+'.csv'
        file_path = './files/'+filename
        res.to_csv(file_path,index=False)
        return send_file(file_path,mimetype='text/csv',attachment_filename=filename,as_attachment=True)

    return render_template('submit.html', form=form)


UPLOAD_FOLDER = './files/'
ALLOWED_EXTENSIONS = set(['csv'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            return str(df.shape)
    return redirect('/submit')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000')