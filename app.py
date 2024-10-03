from flask import Flask, jsonify, request, render_template
from flask_caching import Cache
from students import students  # Import students from data.py
from data import data  # Import data from data.py

app = Flask(__name__)

# Configure cache settings (replace with your desired configuration)
app.config['CACHE_TYPE'] = 'simple'  # Change this to 'redis' or 'memcached' for production
cache = Cache(app)

# Define a route for the root endpoint
@app.route('/')
def index():
    return "Welcome to the data API!"

# Define a route for returning the sample data
@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(data)

# Define a route for returning all students
@app.route('/students', methods=['GET'])
def get_all_students():
    return jsonify(students)

# Define a route for returning a specific student by email
# Define a route for returning a specific student by email
@app.route('/students/<string:email>', methods=['GET'])
def get_student_by_email(email):
    # Search for the student with the given email
    for student in students:
        if student['email'] == email:
            return jsonify(student)
    
    # If the student is not found, return an error
    return jsonify({"error": "Student not found"}), 404


# Define a route for searching students
@app.route('/students/search', methods=['GET'])
def search_students():
    # Get the search query from the request parameters
    query = request.args.get('q')

    # If no query is provided, return an error
    if not query:
        return jsonify({"error": "No search query provided"}), 400

    # Search for students whose names contain the query
    search_results = [student for student in students if query.lower() in student['name'].lower()]

    # Return the search results
    return jsonify(search_results)

# Define a route for serving the search form
@app.route('/search', methods=['GET'])
def search_form():
    return render_template('search.html')

# Define a route for receiving data via POST
@app.route('/students', methods=['POST'])
def receive_student_data():
    # Get the JSON data from the request
    request_data = request.json

    # Process the received data as needed
    # For demonstration purposes, let's just return the received data
    return jsonify(request_data)

if __name__ == '__main__':
    app.run(debug=True)
