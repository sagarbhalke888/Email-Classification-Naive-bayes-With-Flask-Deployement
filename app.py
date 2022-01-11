from flask import Flask,render_template,request,jsonify
import pickle as pkl

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World"

@app.route('/test')
def test():
    return render_template("index.html")


@app.route('/predict',methods=["POST"])
def predict():

    features = [x for x in request.form.values()]
    classifier = pkl.load(open('model.pkl','rb'))
    cv = pkl.load(open('count_vectorizer.pkl','rb'))
    ans = classifier.predict(cv.transform(features))
    print(ans)
    if ans[0] == 0:
        return render_template("index.html",answer = str("Not a Spam"))
    else:
        return render_template("index.html",answer = str(" Spam"))

if __name__ == '__main__':
    app.run(debug=True)