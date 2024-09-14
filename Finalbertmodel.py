import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertModel


# Define the read_combined_text function to read all text files in a directory
def read_combined_text(directory):
    texts = []
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.strip()  # Read and strip any extra whitespace
                    texts.append(text)
    return texts


# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("gujarati-wordpiece-tokenizer.json")

# Define the BERT configuration
config = BertConfig(
    vocab_size=119547,  # Size of your vocabulary for the specific tokenizer used
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)


# Define the BERT model with MLM and NSP
class BertForPreTraining(nn.Module):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__()
        self.bert = BertModel(config)
        self.cls = nn.Linear(config.hidden_size, config.vocab_size)  # MLM head
        self.nsp = nn.Linear(config.hidden_size, 2)  # NSP head

    def forward(self, input_ids, attention_mask, token_type_ids, labels_mlm=None, labels_nsp=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output

        # MLM
        prediction_scores = self.cls(sequence_output)

        # NSP
        seq_relationship_score = self.nsp(pooled_output)

        total_loss = None
        if labels_mlm is not None and labels_nsp is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), labels_mlm.view(-1))
            nsp_loss = loss_fct(seq_relationship_score.view(-1, 2), labels_nsp.view(-1))
            total_loss = mlm_loss + nsp_loss

        return total_loss, prediction_scores, seq_relationship_score


# Create the dataset class for BERT
class TextDatasetForBERT(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, mlm_probability=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.texts) - 1  # NSP requires pairs

    def __getitem__(self, idx):
        # Get sentence pairs for NSP
        sentence_a = self.texts[idx]
        sentence_b = self.texts[idx + 1] if random.random() > 0.5 else self.texts[
            random.randint(0, len(self.texts) - 1)]
        is_next = 1 if sentence_b == self.texts[idx + 1] else 0

        encoding = self.tokenizer.encode_plus(
            sentence_a,
            sentence_b,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        token_type_ids = encoding['token_type_ids'].flatten()

        # Create MLM labels
        labels_mlm = input_ids.clone()
        probability_matrix = torch.full(labels_mlm.shape, self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels_mlm[~masked_indices] = -100  # Only compute loss on masked tokens

        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels_mlm': labels_mlm,
            'labels_nsp': torch.tensor(is_next, dtype=torch.long)
        }


# Path to your dataset directory containing multiple text files
dataset_directory = "datasett"

# Read combined text files from the directory
texts = read_combined_text(dataset_directory)

# Check if texts are read correctly
if not texts:
    print("Error: No texts found in the specified directory.")
else:
    print(f"Successfully read {len(texts)} texts.")

# Create dataset and dataloader only if texts are not empty
if len(texts) > 1:  # Need at least two texts for NSP
    dataset = TextDatasetForBERT(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Instantiate the model
    model = BertForPreTraining(config)

    # Training parameters
    epochs = 3
    learning_rate = 2e-5

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            labels_mlm = batch['labels_mlm']
            labels_nsp = batch['labels_nsp']

            loss, prediction_scores, seq_relationship_score = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels_mlm=labels_mlm,
                labels_nsp=labels_nsp
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    print("Training completed!")

    # Save the model
    model_save_path = "bert_with_mlm_nsp_finetuned.bin"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
else:
    print("Error: Not enough texts to create dataset for NSP.")
