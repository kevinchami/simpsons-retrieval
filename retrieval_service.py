import logging
import datetime
from flask import Flask, request, send_from_directory
from flask_restful import Api, Resource
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
api = Api(app)

# Retrieve the API key from the environment variables
api_key = os.getenv('PINECONE_API_KEY')
index = Pinecone(api_key=api_key).Index('quotes-index')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

logging.basicConfig(level=logging.INFO)

# Serve static files
@app.route('/')
def serve_index():
    return send_from_directory('', 'index.html')

@app.route('/styles.css')
def serve_css():
    return send_from_directory('', 'styles.css')

class Retrieve(Resource):
    def get(self):
        if not model:
            return {"message": "ERROR: NO MODEL"}, 500
        if not index:
            return {"message": "ERROR: NO INDEX"}, 500
        try:
            text = request.args.get('text')
            namespace = request.args.get('namespace', '')  # Use default namespace if not specified
            num = int(request.args.get('num', 5))  # Default to 5 results if not specified
            if text:
                logging.info("Received text: %s", text)
                emb = model.encode(text).tolist()
                logging.info("Encoded text to embeddings: %s", emb)
                response = index.query(vector=emb, top_k=num, include_metadata=True, namespace=namespace)
                logging.info("Response from index: %s", response)
                if not response or 'matches' not in response or len(response['matches']) == 0:
                    return {"message": "No matches found"}, 200
                
                html = '<!DOCTYPE html><html><head>'
                html += '<style> table, th, td { border: 1px solid black; } </style>'
                html += '</head><body><table>'
                html += '<tr><th>Code</th>'
                html += '<th>Results retrieved for text: ' + text + '</th>'
                html += '<th>Author</th>'
                html += '<th>Category</th>'
                html += '<th>Score</th></tr>'
                for match in response['matches']:
                    html += '<tr><td>' + match['id'] + '</td>'
                    metadata = match['metadata']
                    if 'quote' in metadata:
                        html += '<td>' + metadata['quote'] + '</td>'
                    else:
                        html += '<td>unknown</td>'
                    if 'author' in metadata:
                        html += '<td>' + metadata['author'] + '</td>'
                    else:
                        html += '<td>unknown</td>'
                    if 'category' in metadata:
                        html += '<td>' + metadata['category'] + '</td>'
                    else:
                        html += '<td>unknown</td>'
                    html += '<td>' + str(match['score']) + '</td></tr>'
                html += '</table></body></html>'
                return html, 200
            else:
                return {"message": "Bad Request"}, 400

        except Exception as e:
            logging.warning("{} : ERROR: {}".format(datetime.datetime.now(), str(e)))
            return {"message": "ERROR: " + str(e)}, 500

api.add_resource(Retrieve, '/retrieve')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
