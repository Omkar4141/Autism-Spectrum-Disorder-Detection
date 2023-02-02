from  flask import Flask,request,jsonify
import numpy as np
import pickle
model=pickle.load(open('LRmodel_Toodler.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return "Hello World"
@app.route('/predict',methods=['GET'])
def predict():
    a1=request.form.get('a1')
    a2=request.form.get('a2')
    a3=request.form.get('a3')
    a4=request.form.get('a4')
    a5=request.form.get('a5')
    a6=request.form.get('a6')
    a7=request.form.get('a7')
    a8=request.form.get('a8')
    a9=request.form.get('a9')
    a10=request.form.get('a10')
    input_q=np.array([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]]).astype(int)
  
    result=model.predict(input_q)[0]
    return jsonify({'ASD':str(result)})
    # result={'a1':a1,'a2':a2,'a3':a3,'a4':a4,'a5':a5,'a6':a6,'a7':a7,'a8':a8,'a9':a9,'a10':a10}
    #return jsonify(result)#it will get in json format
if __name__ == '__main__':
    app.run(debug=True)


