import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tiktoken

class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        """
        Args:
            text (str): Complete Shakespeare text.
            seq_length (int): Length of each sequence (T).
        """
        self.seq_length = seq_length
        self.text = text
        enc = tiktoken.get_encoding('gpt2')
        self.encoded_text = enc.encode(text)  # Convert text to integers

    def __len__(self):
        """Returns the number of sequences available."""
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        """Returns a single sample of input and target sequences."""
        input_seq = self.encoded_text[idx:idx + self.seq_length]
        target_seq = self.encoded_text[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

    def num_tokens(self):
        """Returns the number of tokens in the dataset."""
        return len(self.encoded_text)

def get_shakespeare_dataloader(batch_size, seq_length, text, shuffle=True):
    """
    Creates a DataLoader for the Shakespeare dataset.

    Args:
        batch_size (int): Number of sequences per batch (B).
        seq_length (int): Length of each sequence (T).
        text (str): Complete Shakespeare text.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader instance for the dataset.
    """
    dataset = ShakespeareDataset(text, seq_length)
    num_tokens = dataset.num_tokens()
    print(f"Number of tokens in the dataset: {num_tokens}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, num_tokens