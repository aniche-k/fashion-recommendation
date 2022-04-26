from enum import unique
from flask import Flask, flash,request, send_from_directory,url_for,render_template
from flask_sqlalchemy import SQLAlchemy
import os
import werkzeug
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from numpy import asarray
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import tensorflow as tf
from flask_login import UserMixin
from flask_login import LoginManager,login_user,login_required,logout_user,current_user
from flask.json import jsonify
import wardrobe,recommender
fashion_model=tf.keras.models.load_model('models/fashion-010.model')
color_model=tf.keras.models.load_model('models/fashion-colors-020.model')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(12)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
temp=None
next= -1
res=[[]]
colors_array=['black', 'blue', 'brown', 'green', 'grey', 'khaki', 'marron', 'orange', 'pink', 'red', 'white', 'yellow']
type_array=['longsleeve','outwear','pants','shirt','shoes','t-shirt']

def matching(arr,target):
    for i in range(len(arr)):
        if arr[i] == 1:
            return target[i]



class User(UserMixin,db.Model):
    id=db.Column(db.Integer,primary_key=True)
    uname=db.Column(db.String(100),unique=True)
    password=db.Column(db.String(100),unique=True)
    def __repr__(self) -> str:
        return '<Name %r>' %self.id
    def to_json(self):        
        return {"name": self.uname,
                "ID": self.id}

    def is_authenticated(self):
        return True

    def is_active(self):   
        return True           

    def is_anonymous(self):
        return False          

    def get_id(self):         
        return str(self.id)
    
    def get_uname(self):
        return str(self.uname)


login_manager=LoginManager()
login_manager.init_app(app)
login_manager.login_view='login_result'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def hello_world():
    return render_template("homepage.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/loginresult',methods=["POST"])
def login_result():
    name=request.form['uname']
    pwd=request.form['pwd']
    user=User.query.filter_by(uname=name).first()
    if not user or not check_password_hash(user.password,pwd):
        return "<h1>Enter correct Details </h1>"
    login_user(user,remember=True)
    return render_template("loggedin.html",pred=user.to_json())

@app.route('/logout')
@login_required
def logout():
    logout_user()
    #return jsonify(**{'result': 200,'data': {'message': 'logout success'}})
    return render_template("logout.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/signupresult',methods=["POST"])
def signup_result():
    name=request.form['uname']
    pwd=request.form['pwd']
    user=User.query.filter_by(uname=name).first()
    if user:
        return "Already exists"
    new_user = User(uname=name, password=generate_password_hash(pwd, method='sha256'))
    wardrobe.create_wardrobe(name)
    db.session.add(new_user)
    db.session.commit()
    return "Added User. Please Login"

@app.route('/logout')
def mogout():
    logout_user()
    return "<h1>Logged Out Successfully<h1>"

@app.route("/fashion")
@login_required
def fashion():
    return render_template("fashion.html")

@app.route("/fashionpredict",methods=["POST"])
def fashion_predict():
    filestr = request.files['img'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite("static/test.jpg",img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
    resized=cv2.resize(gray,(100,100))
    r=resized.reshape(1,100,100,1)
    predict=fashion_model.predict(r).tolist()
    predict2=color_model.predict(r).tolist()
    predict=matching(predict[0],type_array)
    predict2=matching(predict2[0],colors_array)
    return render_template("fashion.html",pred=predict,pred2=predict2,image1="test.jpg")

@app.route('/upload',methods=["POST"])
def outfit_upload():
    type=request.form["type"]
    color=request.form["color"]
    #occasion=request.form["occasion"]
    #wardrobe.upload(occasion,type,color,"users/"+current_user.get_uname(),cv2.imread("static/test.jpg"))
    wardrobe.upload(type,color,"users/"+current_user.get_uname(),cv2.imread("static/test.jpg"))
    return render_template("fashion.html",pred3=type,pred4=color)

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory("users/", filename, as_attachment=True)

@app.route('/select')
def select_clothes():
    global next
    next = 0
    recommender.clear()
    root=r"users/"
    l=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            l.append(((os.path.join(path, name)).replace("\\","/")).replace("users/",""))
    print(l)
    return render_template("display_clothes.html",img=l)

@app.route('/recommend',methods=["POST"])
def recommend_combination():
    try:
        pred=(request.form["cloth"]).split("/")
        type=pred[1]
        color=pred[2]
        path=request.form["cloth"]
    except werkzeug.exceptions.BadRequestKeyError:
        type="None"
        color="None"
        path="None"
    occasion=request.form["occasion"]
    #print(pred)
    global next
    next=0
    global res
    res=[[]]
    res=recommender.execute(type,color,path,occasion)
    print(res)
    print(next)
    #print(pred)
    return render_template("display_result.html",img=res[next])

@app.route('/recommendnext')
def recommend_next():
    global next
    next=next+1
    global res
    try:
        mes=res[next]
        return render_template("display_result.html",img=mes)
    except IndexError:
        flash("Reached end of list")
        return render_template("display_result.html",img=res[next-1],hiden="hidden")


if __name__ == '__main__':
    app.run(debug=True)