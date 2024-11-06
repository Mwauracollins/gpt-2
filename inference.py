import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from model.gpt_2 import GPT2
from utils.gpt_config import ModelConfig

import argparse


def generate_text(model, tokenizer, prompt, max_length, device):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
            )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Inference Script")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to start text generation")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the generated text")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of generated text")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on ('cpu' or 'cuda')")

    args = parser.parse_args()

    config = ModelConfig()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2(config)
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.to(args.device)

    #generate the text from prompt
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_length, args.device)

    #save the generated text to the output file
    with open(args.output_file, 'w') as f:
        f.write(generated_text)

    print(f"Generated text saved to {args.output_file}")


if __name__ == "__main__":
    main()