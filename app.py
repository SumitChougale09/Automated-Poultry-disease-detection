from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2  # Assuming OpenCV is used for image processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from model.classifier import load_model,predict_image
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document
import os

app = Flask(__name__)

# Load your disease prediction model here (pseudo-code)
model=load_model('artifacts/my_model.keras')
class_names = ['Healthy', 'Coccidiosis', 'Newcastle disease', 'Salmonellosis']
# Convert content to Document format for disease information
content = """
1. Overview
Coccidiosis is a parasitic disease of the intestinal tract caused by protozoa of the genus Eimeria. It is one of the most common and economically significant diseases affecting poultry worldwide, particularly young birds.

2. Causative Agents
	•	Multiple species of Eimeria can infect poultry, including:
	•	Eimeria tenella (causes severe cecal coccidiosis)
	•	Eimeria acervulina (affects the upper intestine)
	•	Eimeria maxima (middle section of the intestine)
	•	Eimeria necatrix (highly pathogenic, damages mid-intestine)
	•	Eimeria brunetti (lower small intestine and large intestine)
	•	Each species targets specific parts of the intestines, leading to varying degrees of severity.

3. Transmission
	•	Oral-Fecal Route: Birds become infected by ingesting oocysts (infective form of Eimeria) present in contaminated feed, water, or litter.
	•	Environmental Conditions: High humidity, warmth, and crowded living conditions favor the survival and spread of oocysts.

4. Lifecycle
	•	Oocysts are ingested by the bird and undergo sporulation in the digestive system.
	•	The parasite goes through several developmental stages, including trophozoites, schizonts, and merozoites, which damage the intestinal lining.
	•	After completing the lifecycle, new oocysts are excreted in the feces, contaminating the environment.

5. Clinical Signs
	•	Mild Infections: May show no obvious symptoms but can lead to poor feed conversion and slower growth.
	•	Severe Infections:
	•	Diarrhea, sometimes with blood (especially with E. tenella).
	•	Reduced feed and water intake.
	•	Lethargy, ruffled feathers.
	•	Weight loss and poor growth.
	•	Subclinical Infections: Reduced productivity without overt signs; may affect the flock’s performance.

6. Diagnosis
	•	Clinical Observation: Symptoms such as bloody droppings and reduced feed intake.
	•	Post-Mortem Examination: Lesions in specific parts of the intestines can help identify the Eimeria species.
	•	Microscopic Examination: Presence of oocysts in fecal samples.
	•	PCR Testing: Can be used for precise identification of Eimeria species.

7. Treatment
	•	Anticoccidial Medications: Such as amprolium, sulfa drugs, or ionophores (e.g., monensin, salinomycin).
	•	Supportive Therapy: Ensuring adequate hydration and nutrition during treatment.

8. Prevention and Control
	•	Vaccination: Live oocyst vaccines can provide immunity to specific Eimeria species.
	•	Biosecurity Measures:
	•	Regular cleaning and disinfection of poultry houses.
	•	Avoid overcrowding and maintain dry litter to reduce the spread of oocysts.
	•	Management Practices:
	•	Implementing rotational use of anticoccidial drugs to prevent resistance.
	•	Use of probiotics and prebiotics to promote gut health.
	•	Monitoring: Routine fecal sampling for early detection of coccidial infections.

9. Economic Impact
	•	Coccidiosis can lead to significant economic losses due to decreased growth rates, increased feed conversion ratios, higher mortality, and treatment costs.
	•	Subclinical infections can impair overall productivity and increase vulnerability to other diseases.

10. Research and Future Trends
	•	Genetic Resistance: Ongoing research focuses on breeding poultry with genetic resistance to Eimeria.
	•	Alternative Control Measures: Exploring plant-based treatments, essential oils, and natural extracts as supplementary or alternative therapies.
	•	Immunology Studies: Understanding the immune response in poultry to develop more effective vaccines and immunostimulants.
"""

# Create a Document object with metadata for disease information
documents = [Document(
    page_content=content,
    metadata={"source": "Disease Information Guide"}
)]

# Split the text into chunks for better retrieval performance
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
split_documents = text_splitter.split_documents(documents)

# Initialize Embeddings and Create Vector Store for RAG
embedding = OllamaEmbeddings(model="llama3.2")
vectordb = FAISS.from_documents(
    documents=split_documents,
    embedding=embedding
)

# Initialize the Language Model for generating responses
llm = Ollama(model="llama3.2")

# Setup Retriever to fetch relevant documents based on user queries
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Create Prompt Template for QA interactions
prompt = PromptTemplate.from_template("""
You are a medical information assistant. Please answer the question :

Context: {context}

Question: {question}

Provide a clear and accurate answer based on the information in the context.
Answer:""")

# Create the RAG Pipeline for processing queries about diseases
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to process queries about diseases using the RAG system
def process_query(query: str) -> str:
    try:
        # Get relevant documents for source tracking based on query
        docs = retriever.get_relevant_documents(query)
        
        # Get the response from RAG pipeline
        response = rag_chain.invoke({"question": query})
        
        # Format response with sources used in answering the query
        full_response = f"Answer: {response}\n\nSource sections used:\n"
        for i, doc in enumerate(docs, 1):
            full_response += f"{i}. {doc.page_content[:100]}...\n"
        
        return full_response
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Endpoint for image prediction (pseudo-code)
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
    
    return jsonify({"prediction": predicted_class})
# Endpoint for querying the RAG bot
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    
    # Process query using RAG system
    response = process_query(question)
    
    return jsonify({'response': response})

# Endpoint to serve HTML interface


if __name__ == "__main__":
    app.run(debug=True)