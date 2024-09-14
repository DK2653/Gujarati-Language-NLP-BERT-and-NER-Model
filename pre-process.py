import torch
from transformers import BertTokenizer

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("gujarati-wordpiece-tokenizer.json")

# Function to read combined text file
def read_combined_text(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()  # Read and strip any extra whitespace
            texts.append(text)
    return texts

# Path to your combined text file
combined_text_file = "processed_combined_text.txt"

# Read combined text file
texts = read_combined_text(combined_text_file)

# Tokenize and process each text
for text in texts:
    # Tokenize input text
    tokenized_text = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Example processing (masking example)
    mask_indices = [4, 7]  # Example: mask 4th and 7th tokens
    masked_input_ids = input_ids.copy()
    for idx in mask_indices:
        masked_input_ids[idx] = tokenizer.mask_token_id

    # Convert masked_input_ids to tensor
    masked_input_tensor = torch.tensor([masked_input_ids])

    # Labels for MLM (only masked tokens)
    labels = torch.tensor([input_ids[idx] if idx in mask_indices else -100 for idx in range(len(input_ids))])

    # Example output
    print("Original Text:", text)
    print("Tokenized Text:", tokenized_text)
    print("Input IDs:", input_ids)
    print("Masked Input IDs:", masked_input_ids)
    print("Labels for MLM:", labels)
    print()
