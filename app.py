from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/birds')
def birds_page():
    return render_template('birds.html')

if __name__ == '__main__':
    app.run(debug=True)