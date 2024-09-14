from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.processors import TemplateProcessing
import os

# Initialize a tokenizer
tokenizer = Tokenizer(WordPiece())

# Normalization and pre-tokenization
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()

# Setup trainer
trainer = WordPieceTrainer(vocab_size=30522, min_frequency=2,
                           special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

# Train tokenizer on your dataset
files = [os.path.join("datasett", f) for f in os.listdir("datasett") if f.endswith(".txt")]

tokenizer.train(files, trainer)

# Save the tokenizer
tokenizer.save("gujarati-wordpiece-tokenizerontxtfile.json")

print("Tokenizer trained and saved as 'gujarati-wordpiece-tokenizer.json'")
