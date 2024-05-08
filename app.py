from flask import Flask, request
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pickle
import numpy as np
import pandas as pd 

app = Flask(__name__)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Load data and model
with open("pickle_file", 'rb') as f:
    loaded_data = pickle.load(f)

chunks_embeddings = []

def initialize_chunks_embeddings():
    global chunks_embeddings
    df = pd.read_csv("jobs.csv")
    df["result"] = df["company_id"] + df["job_name"] + df["taglist"] + df["location"] + df["three_reasons"] + df["description"]
    for text in df["result"]:
        embedding = model.encode(text)
        chunks_embeddings.append((text, embedding))

initialize_chunks_embeddings()

# API Key for GenerativeAI
api_key = "YOUR_API_KEY"
genai.configure(api_key=api_key)
genmodel = genai.GenerativeModel('gemini-pro')

# Answer function
def answer(text):
    query_embedding = model.encode(text)
    query_embedding = np.array([query_embedding]).astype('float32')
    k = 10
    D, I = loaded_data.search(query_embedding, k)
    retrieved_list = [chunks_embeddings[i][0] for i in I[0]]
    texts = []
    for text in retrieved_list:
        texts.append(text)
    prompt = "Given the following text: '{texts}', please generate a json file of dictionary with keys 'company', 'job_title', 'description', and 'location'. The generated content should include the name of the company, job title, job description, and location associated with the provided text."
    response = genmodel.generate_content(prompt.format(texts=texts))
    response_text = response.text
    start_index = response_text.find("{")
    end_index = response_text.rfind("}") + 1
    json_content = response_text[start_index:end_index].strip().replace('```', '')
    return json_content

# Flask route
@app.route('/getDetails', methods=['POST'])
def getJobDetails():
    data = request.get_json()
    text = data.get('text', '')
    response = answer(text)
    return response

@app.route('/hello', methods=['GET'])
return "HELLO"

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8080, debug=True)
    except Exception as e:
        print("An error occurred:", e)
