# train.py
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import EsmTokenizer, EsmModel, get_linear_schedule_with_warmup

# --- 1. Custom Dataset Definition ---
class BFCDataset(Dataset):
    """
    Dataset for B-Factor to RMSF mapping.
    This minimalist version uses only the sequence and the normalized B-factor as input.
    """
    def __init__(self, jsonl_file):
        print(f"Loading data from {jsonl_file}...")
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        sequence = item['Sequence']
        
        # The only extra feature is the normalized B-factor
        norm_b_factor = torch.tensor(item['Norm_B_factor'], dtype=torch.float).unsqueeze(1)
        
        # Label
        label = torch.tensor(item['Norm_RMSF'], dtype=torch.float)

        return {
            'sequence': sequence,
            'other_features': norm_b_factor,
            'label': label
        }

# --- 2. Collate Function for Padding ---
def create_collate_fn(tokenizer, max_len):
    """Creates a collate function for padding batches to a fixed max_len."""
    def collate_fn(batch):
        sequences = [item['sequence'][:max_len] for item in batch]
        other_features_list = [item['other_features'][:max_len] for item in batch]
        labels_list = [item['label'][:max_len] for item in batch]

        tokenized = tokenizer(
            sequences,
            padding='longest',
            truncation=True,
            max_length=max_len,
            return_tensors='pt',
            add_special_tokens=False
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        other_features_padded = pad_sequence(other_features_list, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-1.0)

        current_len = input_ids.size(1)
        if current_len < max_len:
            pad_len = max_len - current_len
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=tokenizer.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
            other_features_padded = torch.nn.functional.pad(other_features_padded, (0, 0, 0, pad_len), value=0.0)
            labels_padded = torch.nn.functional.pad(labels_padded, (0, pad_len), value=-1.0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'other_features': other_features_padded,
            'labels': labels_padded
        }
    return collate_fn

# --- 3. Model Architecture ---
class BFCModel(nn.Module):
    """B-Factor Corrector (BFC) Model using ESM-2."""
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D", num_extra_features=1, dropout_rate=0.1):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)
        esm_hidden_size = self.esm.config.hidden_size
        
        self.regression_head = nn.Sequential(
            nn.Linear(esm_hidden_size + num_extra_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, other_features):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        esm_embeddings = outputs.last_hidden_state
        combined_features = torch.cat([esm_embeddings, other_features], dim=-1)
        predictions = self.regression_head(combined_features)
        return predictions.squeeze(-1)

# --- 4. Custom Metrics ---
def masked_mae_loss(predictions, labels):
    mask = labels != -1.0
    masked_preds = torch.masked_select(predictions, mask)
    masked_labels = torch.masked_select(labels, mask)
    return nn.functional.l1_loss(masked_preds, masked_labels)

def masked_pearson_corr(predictions, labels):
    mask = labels != -1.0
    masked_preds = torch.masked_select(predictions, mask)
    masked_labels = torch.masked_select(labels, mask)
    
    if len(masked_preds) < 2: return torch.tensor(0.0)
        
    vx = masked_preds - torch.mean(masked_preds)
    vy = masked_labels - torch.mean(masked_labels)
    
    denom = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    if denom < 1e-6: return torch.tensor(0.0)

    return torch.sum(vx * vy) / denom

# --- 5. Main Training Function ---
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = EsmTokenizer.from_pretrained(args.model_name)
    
    train_dataset = BFCDataset(args.train_file)
    val_dataset = BFCDataset(args.val_file)
    
    custom_collate_fn = create_collate_fn(tokenizer, args.max_len)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True
    )
    
    model = BFCModel(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )
    
    best_val_pcc = -1.0
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            other_features = batch['other_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask, other_features)
            loss = masked_mae_loss(predictions, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        all_val_pcc = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]")
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                other_features = batch['other_features'].to(device)
                labels = batch['labels'].to(device)

                predictions = model(input_ids, attention_mask, other_features)
                loss = masked_mae_loss(predictions, labels)
                pcc = masked_pearson_corr(predictions, labels)

                total_val_loss += loss.item()
                all_val_pcc.append(pcc.item())
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_pcc = np.mean(all_val_pcc)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val PCC: {avg_val_pcc:.4f}")
        
        if avg_val_pcc > best_val_pcc:
            best_val_pcc = avg_val_pcc
            save_path = os.path.join(args.output_dir, "best_bfc_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🎉 New best model saved to {save_path} with PCC: {best_val_pcc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train BFC Model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data (.jsonl)")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation data (.jsonl)")
    parser.add_argument("--output_dir", type=str, default="bfc_model", help="Directory to save the best model")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="ESM model name from Hugging Face")
    
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length for padding and truncation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for AdamW optimizer")
    
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")

    args = parser.parse_args()
    main(args)