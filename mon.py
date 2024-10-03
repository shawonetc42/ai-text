from transformers import AutoTokenizer, AutoModelForCausalLM

# gpt-neo মডেল এবং টোকেনাইজার লোড করুন
model_name = "EleutherAI/gpt-neo-2.7B"  # gpt-neo মডেল
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ইনপুট টেক্সট
input_text = "Write a Python function that takes a string input and returns the string in reverse order."

# ইনপুট আইডি তৈরি করুন
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# কোড জেনারেট করুন
output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# টোকেন ডিকোড করে কোড জেনারেট
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Code:")
print(generated_code)
