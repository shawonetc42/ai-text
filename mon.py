from flask import Flask, request, jsonify
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import os

# Flask অ্যাপ ইনিশিয়ালাইজ করা
app = Flask(__name__)

# লগিং মেসেজ
print("Initializing model loading...")

# GPT-Neo মডেল এবং টোকেনাইজার লোড করা (লগ সহ)
model_name = "EleutherAI/gpt-neo-125M"
try:
    print("Loading the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully!")

    print("Loading the model...")
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error during model loading: {e}")

# রুট তৈরি করা
@app.route('/generate', methods=['POST'])
def generate_story():
    try:
        # লগging incoming request
        print("Received a request for story generation...")

        # ইনপুট প্রম্পট প্রাপ্তি
        data = request.json
        prompt = data.get('prompt', '')

        # ইনপুট যাচাই
        if not prompt:
            print("No prompt provided!")
            return jsonify({"error": "No prompt provided"}), 400

        print(f"Prompt received: {prompt}")

        # ইনপুট টোকেনাইজ করা (লগ সহ)
        print("Tokenizing input...")
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # আউটপুট জেনারেট করা (লগ সহ)
        print("Generating output from model...")
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7)
        
        print("Output generated successfully!")

        # আউটপুট ডিকোড করা
        generated_story = tokenizer.decode(output[0], skip_special_tokens=True)

        print("Story generated: ", generated_story)

        # JSON রেসপন্স পাঠানো
        return jsonify({'story': generated_story})
    
    except Exception as e:
        print(f"Error during story generation: {e}")
        return jsonify({"error": "An error occurred while generating the story"}), 500

# অ্যাপ রান করা
if __name__ == '__main__':
    # লগging যে সার্ভারটি শুরু হচ্ছে
    print("Starting the Flask app...")

    # Flask অ্যাপ রান করো
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
