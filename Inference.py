import torch
from Model import Transformer
from tokenizers import Tokenizer

prompt = "Once upon a time there was a happy bear" # <============= Input prompt here

model_path = "saved_models/checkpoint_epoch_0.pt"
tokenizer_path = "tokenizer.json"
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer.from_file(tokenizer_path)
    
model = Transformer(
vocab_size=tokenizer.get_vocab_size(),
num_groups=2 
).to(device)
    
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Encode input
encoding = tokenizer.encode(prompt)
input_ids = encoding.ids
input_tensor = torch.tensor([input_ids], device=device)
generated_ids = []
eos_id = tokenizer.token_to_id("<|endoftext|>")
max_length = 200  
temperature = 0.75 # Lower values make the model more deterministic, higher values make it more random

with torch.no_grad():
    for _ in range(max_length):
        outputs = model(input_tensor)
        logits = outputs[0, -1, :]  
        scaled_logits = logits / temperature
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token == eos_id:
            break
        generated_ids.append(next_token)
        input_tensor = torch.cat([
            input_tensor,
            torch.tensor([[next_token]], device=device)
        ], dim=1)

# Decode and print the generated text
full_response = tokenizer.decode(generated_ids)
clean_response = full_response.replace('Ġ', ' ').replace('##', '') \
                                    .replace('<|endoftext|>', '').strip()
        
print(f"\n{prompt} {clean_response}")

