from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# ছোট GPT-Neo মডেল লোড করা
model_name = "EleutherAI/gpt-neo-125M"  # ছোট মডেল
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# প্রম্পট সংজ্ঞায়িত করা
prompt = "Once upon a time, there was a girl named Alice who lived in a small town."

# ইনপুট টোকেনাইজ করা
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# আউটপুট জেনারেট করা
output = model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)

# আউটপুট ডিকোড করা
generated_story = tokenizer.decode(output[0], skip_special_tokens=True)

# জেনারেট করা স্টোরি প্রিন্ট করা
print(generated_story)
