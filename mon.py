from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

app = Flask(__name__)

# DistilGPT-2 মডেল লোড করা
model_name = "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
