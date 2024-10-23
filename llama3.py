import transformers
import torch

model_id = "/data/llava/LLaVA-Meta-Llama-3-8B-Instruct-FT"

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Define the conversation
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Combine messages into a single input string
input_text = ""
for msg in messages:
    input_text += f"{msg['role']}: {msg['content']}\n"

# Tokenize the input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

# Generate the response
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# Decode the output
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
