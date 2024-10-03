from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Flask অ্যাপ তৈরি করুন
app = Flask(__name__)

# gpt-neo মডেল এবং টোকেনাইজার লোড করুন
model_name = "EleutherAI/gpt-neo-2.7B"  # gpt-neo মডেল
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate():
    input_text = request.json.get("input_text")
    
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    # ইনপুট আইডি তৈরি করুন
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # কোড জেনারেট করুন
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # টোকেন ডিকোড করে কোড জেনারেট
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"generated_code": generated_code})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
