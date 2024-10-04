import os
from flask import Flask, request, jsonify
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

app = Flask(__name__)

# ছোট GPT-Neo মডেল লোড করা
model_name = "EleutherAI/gpt-neo-125M"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate_story():
    data = request.json
    prompt = data.get('prompt', '')

    # ইনপুট টোকেনাইজ করা
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # আউটপুট জেনারেট করা
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)

    # আউটপুট ডিকোড করা
    generated_story = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'story': generated_story})

if __name__ == '__main__':
    # রেন্ডারে চলার সময় PORT এনভায়রনমেন্ট ভ্যারিয়েবল থেকে পোর্ট নম্বর নিন
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
