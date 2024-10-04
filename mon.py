from flask import Flask, request, jsonify
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import os

# Initialize Flask app
app = Flask(__name__)

# Load the GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

@app.route('/generate_code', methods=['POST'])
def generate_code():
    # Get the input prompt from the request JSON
    data = request.get_json()
    input_text = data.get('input_text')

    if not input_text:
        return jsonify({"error": "Input text is required."}), 400

    # Tokenize the input
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Check the number of tokens
    if len(input_ids[0]) > 2048:  # Maximum token limit for GPT-Neo
        return jsonify({"error": "Input prompt is too long."}), 400

    # Generate output from the model
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.7)

    # Decode the generated tokens into text
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"generated_code": generated_code})

# Run the Flask app
if __name__ == '__main__':
    # Get the PORT environment variable (set by Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
