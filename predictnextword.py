import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load the evaluation model and tokenizer
model_path = 'TRY3/gujarati_nsp_dataset.txt'  # Path to the saved evaluation model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)


def predict_next_word(input_text, max_length=50):
    # Add a mask token to the end of the input text
    masked_text = input_text + ' [MASK]'

    # Tokenize input text and convert to tensor
    inputs = tokenizer(masked_text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits

    # Get the index of the [MASK] token
    masked_index = torch.where(input_ids[0] == tokenizer.mask_token_id)[0]
    predicted_id = torch.argmax(predictions[0, masked_index], dim=-1)

    # Decode predicted token ID
    predicted_token = tokenizer.decode(predicted_id[0])

    # Append predicted token to input text
    completed_sentence = input_text + ' ' + predicted_token

    return completed_sentence


# Example input
input_text = "ાજ્યના દક્ષિણ"
completed_sentence = predict_next_word(input_text)
print("Completed sentence:", completed_sentence)
