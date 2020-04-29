from LocallyWeighted import LocallyWeightedRegression
from EnsembleRegression import EnsembledRegression
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True


import flask
from flask import request, jsonify
app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"
@app.route('/predict',methods=['GET'])
def predictions():
    startDate = request.args.get('startdate',type=str)
    model = EnsembledRegression(startDate)
    endDate = request.args.get('enddate',type=str)
    results = model.predict(startDate, endDate)
    results.index = results.index.astype(str)
    jsonres = results.to_dict(orient='index')
    y=jsonify(jsonres)
    del model
    y.status_code=200

    return y

if __name__ == '__main__':
    app.run()