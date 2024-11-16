import os
from flask import Flask, render_template, request, jsonify
from model.classifier import load_model, predict_image, class_names
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama 

# Initialize Flask app
app = Flask(__name__)

# Load model
model_path = 'artifacts/my_model.keras'
model = load_model(model_path)

# Class names for prediction
class_names = ['Healthy', 'Coccidiosis', 'Newcastle disease', 'Salmonellosis']

# Load documents for RAG (assuming disease-related data is stored in text files)
disease_docs = {
    "Coccidiosis": "data/coccidiosis.txt",
    "Newcastle disease": "data/newcastle_disease.txt",
    "Salmonellosis": "data/salmonellosis.txt"
}

# Set up Llama 3.2 for embeddings and FAISS for vector store
embedding = OllamaEmbeddings(model="llama3.2")
vector_db = {}

for disease, doc_path in disease_docs.items():
    with open(doc_path, 'r') as file:
        text = file.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        db = FAISS.from_texts(chunks, embedding)
        vector_db[disease] = db

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Predict the class
    predicted_class = predict_image(model, file_path)
    
    # Check if the prediction is a disease
    if predicted_class in vector_db:
        response = {"prediction": predicted_class, "info": "You can now ask questions about this disease."}
    else:
        response = {"prediction": predicted_class, "info": "No further information is available for this result."}
    
    return jsonify(response)
ollama_model = Ollama(model='llama3.2')
@app.route('/qa', methods=['POST'])
def qa():
    data = request.get_json()
    question = data.get('question')
    disease = data.get('disease')
    
    if not disease or disease not in vector_db:
        return jsonify({"error": "Invalid or missing disease information."})
    
    db = vector_db[disease]
    docs = db.similarity_search(question, k=3)
    
    # Use Ollama for generation
    context = "\n".join([doc.page_content for doc in docs])
    template = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the context: {context}\nAnswer the question: {question}"
    )
    prompt = template.format(context=context, question=question)
    
    # Generate response using the Ollama model
    llm_response = ollama_model.generate([prompt])  # Wrap prompt in a list

# Access the generated response text from the LLMResult object
    answer = llm_response.generations[0][0].text

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
