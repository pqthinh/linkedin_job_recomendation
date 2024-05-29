from flask import Flask, request, jsonify
from model.recomend_jobs import get_recommended_jobs

app = Flask(__name__)

@app.route('/api/search', methods=['GET'])
def search():
    skills = request.args.get('skills')
    recommended_jobs = get_recommended_jobs(skills)
    return jsonify(recommended_jobs.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(port = 5000, debug=True)