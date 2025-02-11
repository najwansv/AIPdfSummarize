from flask import Flask, request, jsonify, render_template, session
from PyPDF2 import PdfReader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.template_folder = 'Dashboard'

# Constants for text processing
MAX_CHUNK_SIZE = 1024  # Maximum chunk size for processing
MAX_NEW_TOKENS = 150   # Maximum new tokens to generate

def initialize_model():
    global global_tokenizer, global_model, global_nlp
    try:
        model_name = "gpt2"
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        print("Loading model from cache or downloading...")
        global_tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            truncation=True,
            padding=True
        )
        
        global_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        global_nlp = pipeline(
            "text-generation",
            model=global_model,
            tokenizer=global_tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Store chunks in session to handle large files
        chunks = [text[i:i + MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)]
        session['pdf_chunks'] = chunks[:5]  # Store first 5 chunks to limit session size
        
        # Generate initial summary
        input_text = chunks[0][:MAX_CHUNK_SIZE]
        summary = global_nlp(
            input_text, 
            max_length=len(input_text) + MAX_NEW_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=1,
            pad_token_id=global_tokenizer.eos_token_id
        )[0]['generated_text']
        
        return jsonify({
            "success": True,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    if 'pdf_chunks' not in session:
        return jsonify({"error": "No PDF loaded"})
    
    data = request.get_json()
    question = data.get('message', '')
    
    try:
        context = session['pdf_chunks'][0]  # Use first chunk as context
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        response = global_nlp(
            prompt,
            max_length=len(prompt) + MAX_NEW_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=1,
            pad_token_id=global_tokenizer.eos_token_id
        )[0]['generated_text']
        
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)})

def summarize_text(text):
    # Truncate text if too long
    max_length = 1024
    input_text = f"Summarize the following text:\n{text[:max_length]}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=2
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def generate_response(question, context):
    prompt = f"Context: {context[:1024]}\nQuestion: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Initialize model when starting the app
if __name__ == '__main__':
    if initialize_model():
        app.run(debug=True)
    else:
        print("Failed to initialize model. Exiting.")