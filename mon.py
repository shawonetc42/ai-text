from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize Flask app
app = Flask(__name__)

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"  # GPT-Neo model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Flask API route to generate code
@app.route('/generate_code', methods=['POST'])
def generate_code():
    # Get the input prompt from the request JSON
    data = request.get_json()
    input_text = data.get('input_text')

    # Check if input text is provided
    if not input_text:
        return jsonify({"error": "No input text provided."}), 400

    # Limit input length to prevent errors
    input_text = input_text[:1000]  # Limit to first 1000 characters

    # Generate input IDs for the model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Check token length before generating output
    if len(input_ids[0]) > 1024:  # Keep within a safe limit
        return jsonify({"error": "Input text is too long. Please shorten the input."}), 400

    # Generate output from the model
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated tokens into text
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

    # Manually override the output if the input is 'what is your name'
    if input_text.lower() == "what is your name":
        generated_code = "My name is Shawon, and I am here to assist you!"

    # Return the generated code as a JSON response
    return jsonify({"generated_code": generated_code})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
