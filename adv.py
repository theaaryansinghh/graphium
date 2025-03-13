import random
import time
from flask import Flask, render_template, request

app = Flask(__name__)

# Define options
options = [f"Option {i+1}" for i in range(20)]
positions = {option: [random.randint(10, 90), random.randint(10, 90)] for option in options}

@app.route('/')
def home():
    search_query = request.args.get('search', '')
    return render_template('index.html', options=options, positions=positions, search_query=search_query)

@app.route('/option/<name>')
def option_page(name):
    return render_template('option.html', name=name)

# Move options slowly
def update_positions():
    for option in options:
        positions[option][0] += random.uniform(-2, 2)
        positions[option][1] += random.uniform(-2, 2)
        positions[option][0] = max(10, min(90, positions[option][0]))
        positions[option][1] = max(10, min(90, positions[option][1]))
    time.sleep(0.5)

if __name__ == '__main__':
    app.run(debug=True)
