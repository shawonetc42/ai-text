from flask import Flask, request, jsonify
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import os  # os মডিউল ইম্পোর্ট করা হচ্ছে

app = Flask(__name__)

# ছোট GPT-Neo মডেল লোড করা
model_name = "EleutherAI/gpt-neo-125M"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate_story():
    # ইনপুট ডেটা থেকে প্রম্পট নেয়া
    data = request.json
    prompt = data.get('prompt', '').strip()  # ইনপুট থেকে অতিরিক্ত স্পেস সরানো

    # ইনপুট ভ্যালিডেশন
    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    try:
        # ইনপুট টোকেনাইজ করা
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # আউটপুট জেনারেট করা
        output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            attention_mask=attention_mask
        )

        # আউটপুট ডিকোড করা
        generated_story = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({'story': generated_story})

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # ত্রুটির ক্ষেত্রে 500 স্ট্যাটাস কোড ফেরত দিন


if __name__ == '__main__':
    # PORT পরিবেশ পরিবর্তনশীল ব্যবহার করে Flask অ্যাপ্লিকেশন চালানো
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
