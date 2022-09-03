from flask import Flask, render_template, url_for, request
import numpy as np
import pickle

# load the model from disk
filename = 'TheLogistic'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		me = request.form['message']
		message = [float(x) for x in me.split()]
		vect = np.array(message).reshape(1, -1)
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)
	   
		
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)