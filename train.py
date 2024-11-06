import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, AdamW
import argparse
import os
from model.gpt_2 import GPT2
from utils.gpt_config import ModelConfig
import torch.nn.functional as F


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encodings = self.tokenizer(
            text, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )

        return encodings.input_ids.squeeze(), encodings.attention_mask.squeeze()


def get_device(requested_device):
    """Safely determine the available device."""
    if requested_device == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    elif requested_device == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def train(model, tokenizer, train_dataset, batch_size, lr, epochs, device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # Using PyTorch's AdamW
    
    # Safely move model to device
    device = torch.device(device)
    model = model.to(device)
    
    print(f"Training on device: {device}")
    print("\nModel Parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")
        total_params += param.numel()

    print(f"\nTotal number of parameters: {total_params:,}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, (input_ids, attention_mask) in enumerate(train_loader):
            try:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                optimizer.zero_grad()
                logits, loss = model(input_ids=input_ids, target=input_ids)
                
                if loss is None:
                    continue
                    
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{step}], Loss: {loss.item():.4f}")

            except RuntimeError as e:
                print(f"Error during training step: {e}")
                continue

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        try:
            checkpoint_path = os.path.join(args.model_output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 Training Script")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training text data file")
    parser.add_argument("--model_output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for training")
    parser.add_argument("--device", type=str, default='cuda', help="Device to train on ('cuda', 'cpu', or 'mps')")

    args = parser.parse_args()

    # Determine the available device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load model and tokenizer
    try:
        config = ModelConfig()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2(config)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load training data
        with open(args.train_data, 'r', encoding='utf-8') as f:
            train_texts = f.readlines()

        train_dataset = TextDataset(train_texts, tokenizer, max_length=args.max_length)

        # Create output directory
        if not os.path.exists(args.model_output_dir):
            os.makedirs(args.model_output_dir)

        # Update device in args
        args.device = device
        
        # Train the model
        train(model, tokenizer, train_dataset, args.batch_size, args.learning_rate, args.epochs, device)

    except Exception as e:
        print(f"An error occurred during initialization: {e}")
        raise


if __name__ == "__main__":
    main()