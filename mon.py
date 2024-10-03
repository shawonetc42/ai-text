import redis
import json
from datetime import datetime
from flask import Flask, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
import os  # Add this import

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Connect to MongoDB
client = MongoClient("mongodb+srv://shawondata:shawondata@cluster0.sigdzxx.mongodb.net/shawon?retryWrites=true&w=majority")
db = client.shawon

# Connect to Redis
redis_host = 'redis-13202.c305.ap-south-1-1.ec2.redns.redis-cloud.com'
redis_port = 13202
redis_password = 'NcsAxaKUmj3p17qXzakYc3m7rSPwnro1'

redis_client = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

def datetime_handler(x):
    if isinstance(x, datetime):
        return x.isoformat()
    raise TypeError("Unknown type")

# Function to fetch answers and cache them
def fetch_and_cache_answers():
    try:
        questions = list(db.questions.find())

        result = []

        for question in questions:
            answers = list(db.answers.find({'questionId': question['_id']}))

            question_text = question.get('questiontext', '')

            formatted_answers = [{
                '_id': str(answer['_id']),
                'answerText': answer.get('answerText', ''),
                'answeredBy': answer.get('answeredBy', ''),
                'questionId': str(answer.get('questionId', '')),
                'timestamp': answer.get('timestamp', '').isoformat() if answer.get('timestamp', '') else '',
                'answerUserPhoto': answer.get('answerUserPhoto', ''),
                'questiontext': question_text
            } for answer in answers]

            result.extend(formatted_answers)

        redis_client.setex('answersall_cache', 10, json.dumps(result, default=datetime_handler))

        return result

    except Exception as e:
        app.logger.error(f"Failed to fetch answers: {str(e)}")
        return None

@app.route('/answersall', methods=['GET'])
def get_answers_datas():
    try:
        cached_response = redis_client.get('answersall_cache')
        if cached_response:
            app.logger.debug("Cache hit")
            return jsonify(json.loads(cached_response)), 200
        else:
            app.logger.debug("Cache miss")

        result = fetch_and_cache_answers()
        if result is None:
            return jsonify({'error': 'Failed to fetch answers'}), 500

        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Failed to fetch answers: {str(e)}")
        return jsonify({'error': 'Failed to fetch answers', 'details': str(e)}), 500

@app.route('/answers', methods=['GET'])
def get_questions_with_answers():
    try:
        questions = list(db.questions.find())
        
        for question in questions:
            question['_id'] = str(question['_id'])
            question['answers'] = list(db.answers.find({'questionId': ObjectId(question['_id'])}))
            for answer in question['answers']:
                answer['_id'] = str(answer['_id'])
                answer['questionId'] = str(answer['questionId'])
                if 'timestamp' in answer and isinstance(answer['timestamp'], datetime):
                    answer['timestamp'] = answer['timestamp'].isoformat()
        
        return jsonify(questions), 200
    except Exception as e:
        return jsonify({'error': 'Failed to fetch questions', 'details': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
