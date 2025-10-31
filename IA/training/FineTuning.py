"""
Classic Fine-Tuning System with HuggingFace Datasets Only
NO Wikipedia scraping, NO QA generation - Just clean HF datasets

Sources (100% English):
- Anthropic/hh-rlhf (conversations)
- HuggingFaceH4/ultrachat_200k (general chat)
- OpenAssistant/oasst2 (English only)
- Salesforce/xlam-function-calling-60k (function calling)
- glaiveai/glaive-function-calling-v2 (function calling v2)
"""

import os
import sys
import json
import time
from tqdm import tqdm
from typing import List, Dict
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.HessGPT import HessGPT
from Tokenizer.tokenizerv2 import MYBPE

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    print("❌ 'datasets' not installed. Install with: pip install datasets")
    HF_AVAILABLE = False
    sys.exit(1)


class HuggingFaceDatasetLoader:
    """
    Loads and parses all HuggingFace datasets (English only)
    """
    
    def __init__(self):
        self.datasets_loaded = {}
    
    def load_hh_rlhf(self, max_samples: int = 5000) -> List[Dict]:
        """Load Anthropic HH-RLHF conversations"""
        print(f"\n📥 Loading Anthropic/hh-rlhf ({max_samples} samples)...")
        
        try:
            dataset = load_dataset("Anthropic/hh-rlhf", split=f"train[:{max_samples}]")
            
            pairs = []
            for item in tqdm(dataset, desc="Parsing hh-rlhf"):
                if 'chosen' not in item:
                    continue
                
                text = item['chosen']
                
                # Parse format: "\n\nHuman: ... \n\nAssistant: ..."
                if '\n\nHuman:' not in text or '\n\nAssistant:' not in text:
                    continue
                
                parts = text.split('\n\nAssistant:')
                if len(parts) < 2:
                    continue
                
                human = parts[0].replace('\n\nHuman:', '').strip()
                assistant = parts[1].split('\n\nHuman:')[0].strip()
                
                if human and assistant:
                    pairs.append({
                        'human': human,
                        'assistant': assistant,
                        'source': 'hh-rlhf'
                    })
            
            print(f"   ✅ Loaded {len(pairs)} conversations")
            self.datasets_loaded['hh-rlhf'] = len(pairs)
            return pairs
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
    
    def load_ultrachat(self, max_samples: int = 6000) -> List[Dict]:
        """Load UltraChat conversations"""
        print(f"\n📥 Loading UltraChat ({max_samples} samples)...")
        
        try:
            dataset = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split=f"train_sft[:{max_samples}]"
            )
            
            pairs = []
            for item in tqdm(dataset, desc="Parsing ultrachat"):
                if 'messages' not in item or len(item['messages']) < 2:
                    continue
                
                messages = item['messages']
                
                # Extract user/assistant pairs
                for i in range(0, len(messages) - 1, 2):
                    if i + 1 >= len(messages):
                        break
                    
                    if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                        human = messages[i]['content']
                        assistant = messages[i+1]['content']
                        
                        if human and assistant:
                            pairs.append({
                                'human': human,
                                'assistant': assistant,
                                'source': 'ultrachat'
                            })
                    
                    if len(pairs) >= max_samples:
                        break
                
                if len(pairs) >= max_samples:
                    break
            
            print(f"   ✅ Loaded {len(pairs)} conversations")
            self.datasets_loaded['ultrachat'] = len(pairs)
            return pairs
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
    
    def load_oasst2_english(self, max_samples: int = 2000) -> List[Dict]:
        """Load OpenAssistant oasst2 (English only)"""
        print(f"\n📥 Loading OASST2 English ({max_samples} samples)...")
        
        try:
            dataset = load_dataset(
                "OpenAssistant/oasst2",
                split=f"train[:{max_samples * 2}]"  # Load more to filter English
            )
            
            pairs = []
            for item in tqdm(dataset, desc="Parsing oasst2"):
                # Filter English only
                if item.get('lang') != 'en':
                    continue
                
                if item.get('role') == 'prompter' and item.get('text'):
                    prompt = item['text']
                    message_id = item.get('message_id')
                    
                    if message_id and prompt:
                        # Simple response for now (you can improve this)
                        pairs.append({
                            'human': prompt.strip(),
                            'assistant': 'I am here to help you with your question.',
                            'source': 'oasst2'
                        })
                    
                    if len(pairs) >= max_samples:
                        break
            
            print(f"   ✅ Loaded {len(pairs)} conversations (English only)")
            self.datasets_loaded['oasst2'] = len(pairs)
            return pairs
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
    
    def load_xlam_function_calling(self, max_samples: int = 3000) -> List[Dict]:
        """Load XLAM function calling dataset"""
        print(f"\n📥 Loading XLAM Function Calling ({max_samples} samples)...")
        
        try:
            dataset = load_dataset(
                "Salesforce/xlam-function-calling-60k",
                split=f"train[:{max_samples}]"
            )
            
            pairs = []
            for item in tqdm(dataset, desc="Parsing xlam"):
                if 'query' in item and 'answers' in item:
                    pairs.append({
                        'human': item['query'],
                        'assistant': str(item['answers']),
                        'source': 'xlam'
                    })
            
            print(f"   ✅ Loaded {len(pairs)} function calling examples")
            self.datasets_loaded['xlam'] = len(pairs)
            return pairs
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
    
    def load_glaive_function_calling(self, max_samples: int = 2000) -> List[Dict]:
        """Load Glaive function calling dataset"""
        print(f"\n📥 Loading Glaive Function Calling ({max_samples} samples)...")
        
        try:
            dataset = load_dataset(
                "glaiveai/glaive-function-calling-v2",
                split=f"train[:{max_samples}]"
            )
            
            pairs = []
            for item in tqdm(dataset, desc="Parsing glaive"):
                if 'system' in item and 'chat' in item:
                    pairs.append({
                        'human': item['chat'],
                        'assistant': item.get('system', 'Function call executed.'),
                        'source': 'glaive'
                    })
            
            print(f"   ✅ Loaded {len(pairs)} function calling examples")
            self.datasets_loaded['glaive'] = len(pairs)
            return pairs
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
    
    def load_all_datasets(
        self,
        hh_rlhf_count: int = 5000,
        ultrachat_count: int = 6000,
        oasst2_count: int = 2000,
        xlam_count: int = 3000,
        glaive_count: int = 2000
    ) -> List[Dict]:
        """Load all datasets and combine"""
        print("\n" + "="*70)
        print("📦 LOADING ALL HUGGINGFACE DATASETS (English Only)")
        print("="*70)
        
        all_pairs = []
        
        # Load each dataset
        all_pairs.extend(self.load_hh_rlhf(hh_rlhf_count))
        all_pairs.extend(self.load_ultrachat(ultrachat_count))
        all_pairs.extend(self.load_oasst2_english(oasst2_count))
        all_pairs.extend(self.load_xlam_function_calling(xlam_count))
        all_pairs.extend(self.load_glaive_function_calling(glaive_count))
        
        # Shuffle for diversity
        random.shuffle(all_pairs)
        
        print("\n" + "="*70)
        print(f"✅ TOTAL DATASET: {len(all_pairs)} examples (100% English)")
        print("="*70)
        print("📊 Distribution:")
        for source, count in self.datasets_loaded.items():
            percentage = (count / len(all_pairs) * 100) if all_pairs else 0
            print(f"   • {source}: {count} ({percentage:.1f}%)")
        print("="*70 + "\n")
        
        return all_pairs


class InstructionTunedDataset(Dataset):
    """
    PyTorch Dataset with simple Human:/Bot: format
    """
    
    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"🎯 Formatting {len(pairs)} examples with Human:/Bot: template")
        
        self.formatted_pairs = []
        for pair in pairs:
            human = pair['human'].strip()
            assistant = pair['assistant'].strip()
            
            # Simple consistent format
            formatted = f"Human: {human}\nBot: {assistant}"
            
            self.formatted_pairs.append({
                'formatted_text': formatted,
                'human': human,
                'assistant': assistant,
                'source': pair.get('source', 'unknown')
            })
        
        print(f"✅ {len(self.formatted_pairs)} examples formatted")
    
    def __len__(self):
        return len(self.formatted_pairs)
    
    def __getitem__(self, idx):
        formatted_text = self.formatted_pairs[idx]['formatted_text']
        assistant = self.formatted_pairs[idx]['assistant']
        
        # Encode full text
        ids_all = self.tokenizer.encoder(formatted_text)
        
        # Calculate where assistant response starts
        prefix = formatted_text.split("Bot:")[0] + "Bot:"
        ids_prefix = self.tokenizer.encoder(prefix)
        assist_start = len(ids_prefix)
        
        # Truncate if too long
        if len(ids_all) > self.max_length:
            ids_all = ids_all[:self.max_length]
            if assist_start >= self.max_length:
                assist_start = max(0, self.max_length - 10)
        
        return {
            "input_ids": torch.tensor(ids_all, dtype=torch.long),
            "assist_start": assist_start
        }


def collate_fn(batch, pad_id=0):
    """Collate function with proper masking"""
    input_ids_list = [b["input_ids"] for b in batch]
    assist_starts = [b["assist_start"] for b in batch]
    max_len = max([t.size(0) for t in input_ids_list])
    
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, ids in enumerate(input_ids_list):
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        
        # Only unmask assistant response
        start = assist_starts[i]
        if start < L:
            labels[i, start:L] = input_ids[i, start:L]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


class ClassicTrainer:
    """
    Classic fine-tuning trainer (no LoRA)
    """
    
    def __init__(
        self,
        model_dir: str,
        tokenizer_path: str,
        device: torch.device
    ):
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.device = device
        
        os.makedirs(model_dir, exist_ok=True)
        
        # HuggingFace dataset loader
        self.hf_loader = HuggingFaceDatasetLoader()
        
        # Load or initialize model
        self.model, self.tokenizer, self.config = self._load_or_init_model()
        
        # History
        self.history_file = os.path.join(model_dir, "training_history.json")
        self.history = self._load_history()
        
        print(f"\n✅ Classic Trainer initialized (no LoRA)")
    
    def _load_or_init_model(self):
        """Load or initialize model"""
        cfg_path = os.path.join(self.model_dir, "config.json")
        model_path = os.path.join(self.model_dir, "model.pt")
        
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {
                "vocab_size": 5000,
                "embed_dim": 256,
                "num_heads": 8,
                "num_layers": 4,
                "max_seq_len": 512
            }
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)
        
        # Load tokenizer
        tokenizer = MYBPE(vocab_size=cfg["vocab_size"])
        tokenizer.load_tokenizer(self.tokenizer_path)
        
        # Initialize model
        model = GPT2Model(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"]
        )
        
        # Load existing weights if available
        if os.path.exists(model_path):
            print(f"✅ Loading existing model: {model_path}")
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
        else:
            print("🆕 Initializing new model")
        
        model.to(self.device)
        return model, tokenizer, cfg
    
    def _load_history(self):
        """Load training history"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            "cycles": [],
            "total_examples_trained": 0,
            "datasets_used": []
        }
    
    def _save_history(self):
        """Save training history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train_one_cycle(
        self,
        hh_rlhf_count: int = 5000,
        ultrachat_count: int = 6000,
        oasst2_count: int = 2000,
        xlam_count: int = 3000,
        glaive_count: int = 2000,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-5,
        validation_split: float = 0.1
    ):
        """Train one cycle with HuggingFace datasets"""
        cycle_num = len(self.history["cycles"]) + 1
        
        print("\n" + "="*70)
        print(f"🚀 TRAINING CYCLE #{cycle_num} - CLASSIC FINE-TUNING")
        print("="*70)
        print(f"📊 Total examples trained: {self.history['total_examples_trained']}")
        print("="*70 + "\n")
        
        # Load all datasets
        dataset_pairs = self.hf_loader.load_all_datasets(
            hh_rlhf_count=hh_rlhf_count,
            ultrachat_count=ultrachat_count,
            oasst2_count=oasst2_count,
            xlam_count=xlam_count,
            glaive_count=glaive_count
        )
        
        if not dataset_pairs:
            print("❌ Dataset is empty!")
            return
        
        # Split train/val
        val_size = int(len(dataset_pairs) * validation_split)
        train_pairs = dataset_pairs[val_size:]
        val_pairs = dataset_pairs[:val_size]
        
        print(f"📊 Split: {len(train_pairs)} train / {len(val_pairs)} val\n")
        
        # Create PyTorch datasets
        train_dataset = InstructionTunedDataset(
            train_pairs,
            self.tokenizer,
            max_length=self.config["max_seq_len"]
        )
        val_dataset = InstructionTunedDataset(
            val_pairs,
            self.tokenizer,
            max_length=self.config["max_seq_len"]
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id=0)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id=0)
        )
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        # Training loop
        print(f"\n⏳ Training on {len(train_dataset)} examples, {epochs} epochs")
        
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"📍 EPOCH {epoch+1}/{epochs}")
            print(f"{'='*70}")
            
            # TRAINING
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                logits, _ = self.model(input_ids)
                
                # Loss
                loss = loss_fn(
                    logits.view(-1, self.config["vocab_size"]),
                    labels.view(-1)
                )
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️ NaN/Inf detected! Skipping batch...")
                    continue
                
                # Backward
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            
            # VALIDATION
            print(f"\n🔍 Running validation...")
            self.model.eval()
            val_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    logits, _ = self.model(input_ids)
                    loss = loss_fn(
                        logits.view(-1, self.config["vocab_size"]),
                        labels.view(-1)
                    )
                    
                    val_loss += loss.item()
                    val_batch_count += 1
            
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss:   {avg_val_loss:.4f}")
        
        # Save model
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"\n✅ Model saved: {model_path}")
        
        # Update history
        cycle_info = {
            "cycle": cycle_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "examples": len(train_dataset) + len(val_dataset),
            "epochs": epochs,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "datasets": list(self.hf_loader.datasets_loaded.keys())
        }
        self.history["cycles"].append(cycle_info)
        self.history["total_examples_trained"] += len(train_dataset) + len(val_dataset)
        self.history["datasets_used"] = list(set(
            self.history.get("datasets_used", []) + list(self.hf_loader.datasets_loaded.keys())
        ))
        self._save_history()
        
        print("\n" + "="*70)
        print(f"✅ CYCLE #{cycle_num} COMPLETED")
        print("="*70)
        print(f"📉 Final Losses:")
        print(f"   • Train Loss: {avg_train_loss:.4f}")
        print(f"   • Val Loss:   {avg_val_loss:.4f}")
        print(f"\n💾 Model saved: {model_path}")
        print(f"📈 Total examples trained: {self.history['total_examples_trained']:,}")
        print(f"📦 Datasets used: {', '.join(self.history['datasets_used'])}")
        print("="*70 + "\n")
    
    def display_stats(self):
        """Display training statistics"""
        print("\n" + "="*70)
        print("📊 TRAINING STATISTICS")
        print("="*70)
        
        print(f"\n🔢 Cycles completed: {len(self.history['cycles'])}")
        print(f"📝 Total examples trained: {self.history['total_examples_trained']:,}")
        
        if self.history.get('datasets_used'):
            print(f"\n📦 Datasets used:")
            for dataset in self.history['datasets_used']:
                print(f"   • {dataset}")
        
        if self.history['cycles']:
            print(f"\n🕐 Last cycle:")
            last = self.history['cycles'][-1]
            print(f"   • Date: {last['timestamp']}")
            print(f"   • Examples: {last['examples']:,}")
            print(f"   • Train Loss: {last['train_loss']:.4f}")
            print(f"   • Val Loss: {last['val_loss']:.4f}")
        
        print("="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("🤖 CLASSIC FINE-TUNING - HUGGINGFACE ONLY")
    print("="*70)
    print("📦 Datasets: hh-rlhf, ultrachat, oasst2, xlam, glaive")
    print("🌍 Language: 100% English")
    print("🎯 Format: Human:/Bot:")
    print("="*70 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Paths
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models",
        "my_llm"
    )
    tokenizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Tokenizer",
        "tokenizer_5k.bin"
    )
    
    print(f"📁 Model directory: {model_dir}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    
    # Initialize trainer
    trainer = ClassicTrainer(
        model_dir=model_dir,
        tokenizer_path=tokenizer_path,
        device=device
    )
    
    # Display current stats
    trainer.display_stats()
    
    # Train
    print("\n🎯 Starting training...")
    print("   • HH-RLHF: 5,000 samples")
    print("   • UltraChat: 6,000 samples")
    print("   • OASST2: 2,000 samples (English)")
    print("   • XLAM: 3,000 samples")
    print("   • Glaive: 2,000 samples")
    print("   • Total: ~18,000 examples")
    print("   • Epochs: 3")
    print("   • Batch size: 8")
    
    trainer.train_one_cycle(
        hh_rlhf_count=5000,
        ultrachat_count=6000,
        oasst2_count=2000,
        xlam_count=3000,
        glaive_count=2000,
        epochs=3,
        batch_size=8,
        lr=5e-5
    )
    
    # Final stats
    trainer.display_stats()
    
    print("\n✅ Training completed!")
    print("💡 Test your model with Flask app")
    print("💡 If it works, convert to LoRA next!")


if __name__ == "__main__":
    main()
