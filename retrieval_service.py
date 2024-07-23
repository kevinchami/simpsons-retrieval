import logging
import datetime
from flask import Flask, request, send_from_directory
from flask_restful import Api, Resource
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from pinecone import Pinecone

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
api = Api(app)

# Pinecone and model setup
api_key = os.getenv('PINECONE_API_KEY')
index = Pinecone(api_key=api_key).Index('simpsons-index')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

logging.basicConfig(level=logging.INFO)

@app.route('/')
def serve_index():
    return send_from_directory('', 'index.html')

@app.route('/styles.css')
def serve_css():
    return send_from_directory('', 'styles.css')

class Retrieve(Resource):
    def get(self):
        text = request.args.get('text')
        num = int(request.args.get('num', 5))
        if text:
            emb = model.encode(text).tolist()
            response = index.query(vector=emb, top_k=num, include_metadata=True)
            if response and 'matches' in response and response['matches']:
                html = '<table><tr><th>Character</th><th>Quote</th><th>Score</th></tr>'
                for match in response['matches']:
                    character = match['metadata'].get('character', 'Unknown')
                    quote = match['metadata'].get('quote', 'No quote found')
                    score = match['score']
                    html += f'<tr><td>{character}</td><td>{quote}</td><td>{score:.2f}</td></tr>'
                html += '</table>'
                return html, 200
            return "No matches found", 200
        return "Bad Request", 400

api.add_resource(Retrieve, '/retrieve')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
