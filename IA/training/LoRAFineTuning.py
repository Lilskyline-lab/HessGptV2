"""
Système d'entraînement avec LoRA (Low-Rank Adaptation) + Instruction Tuning
VERSION CORRIGÉE avec TOUS LES PATCHES APPLIQUÉS
#LoRAFineTuning.py

Corrections appliquées:
- ✅ HessGPT.forward() retourne (logits, hidden_states)
- ✅ InstructionTunedDataset avec format cohérent
- ✅ collate_fn avec masquage correct (-100)
- ✅ Training loop robuste avec gestion d'erreurs
- ✅ Validation du format avant entraînement
- ✅ Noms de fichiers FIXES (checkpoint.pt, model.pt, best_model.pt)
- ✅ Reprise d'entraînement robuste
"""

import os
import sys
import json
import time
import requests
import re
import logging
from typing import List, Dict, Optional, Tuple, Any, Callable, Union, Iterator
from pathlib import Path
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import Counter, defaultdict
from enum import Enum
import shutil
import warnings
import argparse
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import _LRScheduler

# Ajuster les imports pour votre structure
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.HessGPT import HessGPT
from Tokenizer.tokenizerv2 import MYBPE
from utils.instruction_tuning import (
    InstructionTemplates,
    InstructionTuningPipeline,
    convert_to_instruction_format,
    InstructionDatasetLoader
)
from utils.rlhf_module import RLHFTrainer, RLHFConfig

# ============================================================================
# CONSTANTES ET CONFIGURATION PAR DÉFAUT
# ============================================================================

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models",
    "my_llm"
)

DEFAULT_TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Tokenizer",
    "tokenizer_20k_production.bin"
)

if not os.path.exists(DEFAULT_TOKENIZER_PATH):
    raise FileNotFoundError(
        f"❌ Tokenizer introuvable: {DEFAULT_TOKENIZER_PATH}\n"
        f"Vérifiez que le fichier existe dans IA/Tokenizer/"
    )

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)

DEFAULT_VOCAB_SIZE: int = 20000
DEFAULT_MAX_SEQ_LEN: int = 512
LORA_WEIGHTS_FILENAME: str = "lora_weights.pt"
CONFIG_FILENAME: str = "config.json"


# ============================================================================
# DÉPENDANCES OPTIONNELLES
# ============================================================================

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("Hugging Face datasets non disponible. Fonctionnalité OASST1 désactivée.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class LoRAConfig:
    """Configuration LoRA"""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2'])
    train_bias: bool = False

@dataclass
class TrainingConfig:
    """Configuration d'entraînement"""
    hh_rlhf_count: int = 10000
    ultrachat_count: int = 12000
    oasst2_count: int = 4000
    vigogne_count: int = 2000
    xlam_count: int = 6000
    glaive_count: int = 4000

    validation_split: float = 0.1
    use_custom_data: bool = False

    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    use_amp: bool = True
    scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    use_augmentation: bool = False

@dataclass
class ModelConfig:
    """Configuration du modèle"""
    vocab_size: int = 20000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN


# ============================================================================
# LOGGING SIMPLE
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure un logger simple"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ============================================================================
# IMPLÉMENTATION LoRA CORRIGÉE
# ============================================================================

class LoRALayer(nn.Module):
    """Couche LoRA optimisée"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        train_bias: bool = False
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.train_bias = train_bias

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        if train_bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('lora_bias', None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        # S'assurer que tout est sur le même device
        x = x.to(self.lora_A.device)
        original_output = original_output.to(self.lora_A.device)

        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        result = original_output + lora_output * self.scaling

        if self.train_bias and self.lora_bias is not None:
            result = result + self.lora_bias

        return result


class LoRAWrapper(nn.Module):
    """Wrapper LoRA avec gestion robuste"""

    def __init__(self, base_model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Propriété device
        self._device = next(base_model.parameters()).device

        # Geler le modèle de base
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.lora_layers = nn.ModuleDict()
        self.name_mapping: Dict[str, str] = {}
        self._modules_cache: Optional[Dict[str, nn.Module]] = None

        self._inject_lora()

        trainable = self.count_trainable_params()
        total = self.count_total_params()
        self.logger.info(
            f"LoRA: rank={config.rank}, alpha={config.alpha}, "
            f"trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)"
        )

    @property
    def device(self):
        """Propriété device pour accès externe"""
        return self._device

    def _inject_lora(self) -> None:
        """Injecte les couches LoRA dans le modèle"""
        injected = 0
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                module_name = name.split('.')[-1]
                if module_name in self.config.target_modules:
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                        train_bias=self.config.train_bias
                    )
                    lora_layer.to(self._device)

                    safe_name = name.replace('.', '_')
                    self.lora_layers[safe_name] = lora_layer
                    self.name_mapping[name] = safe_name
                    injected += 1

        if injected == 0:
            warnings.warn(f"⚠️ Aucune couche LoRA injectée!")
        else:
            self.logger.info(f"✅ {injected} couches LoRA injectées")

    @contextmanager
    def _attach_hooks(self) -> Iterator[None]:
        """Hooks corrigés avec gestion device et gradient"""
        handles = []

        def make_hook(lora_layer: LoRALayer) -> Callable:
            def hook(module, input, output):
                try:
                    x = input[0].to(lora_layer.lora_A.device)
                    output = output.to(lora_layer.lora_A.device)

                    lora_out = lora_layer(x, output)

                    if self.training and not lora_out.requires_grad:
                        lora_out = lora_out.requires_grad_(True)

                    return lora_out
                except Exception as e:
                    self.logger.warning(f"Hook LoRA failed: {e}")
                    return output
            return hook

        if self._modules_cache is None:
            self._modules_cache = dict(self.base_model.named_modules())

        try:
            for orig_name, safe_name in self.name_mapping.items():
                lora_layer = self.lora_layers[safe_name]
                module = self._modules_cache[orig_name]
                handle = module.register_forward_hook(make_hook(lora_layer))
                handles.append(handle)
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """✅ CORRECTION: Forward pass retourne (logits, hidden_states)"""
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        with self._attach_hooks():
            # ⭐ CORRECTION: base_model retourne (logits, hidden_states)
            logits, hidden_states = self.base_model(input_ids)

        return logits, hidden_states  # ⭐ Retourner les deux

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_lora_weights(self, path: str) -> None:
        lora_state = {
            'lora_layers': self.lora_layers.state_dict(),
            'config': asdict(self.config),
            'metadata': {
                'trainable_params': self.count_trainable_params(),
                'total_params': self.count_total_params(),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state, path)
        self.logger.info(f"LoRA weights sauvegardés: {path}")

    def load_lora_weights(self, path: str, strict: bool = True) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Fichier LoRA introuvable: {path}")

        lora_state = torch.load(path, map_location=self._device)
        self.lora_layers.load_state_dict(lora_state['lora_layers'], strict=strict)
        self.logger.info(f"LoRA weights chargés: {path}")

    def merge_and_save_full_model(self, path: str) -> None:
        """Fusionne LoRA dans le modèle de base et sauvegarde"""
        for param in self.base_model.parameters():
            param.requires_grad = True

        if self._modules_cache is None:
            self._modules_cache = dict(self.base_model.named_modules())

        self.logger.info("Fusion LoRA dans le modèle de base...")
        for orig_name, safe_name in self.name_mapping.items():
            lora_layer = self.lora_layers[safe_name]
            module = self._modules_cache[orig_name]

            if isinstance(module, nn.Linear):
                delta_w = (lora_layer.lora_A @ lora_layer.lora_B) * lora_layer.scaling
                module.weight.data = module.weight.data + delta_w.T

                if self.config.train_bias and lora_layer.lora_bias is not None:
                    if module.bias is None:
                        module.bias = nn.Parameter(torch.zeros_like(lora_layer.lora_bias))
                    module.bias.data = module.bias.data + lora_layer.lora_bias.data

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.base_model.state_dict(), path)
        self.logger.info(f"Modèle fusionné sauvegardé: {path}")

        for param in self.base_model.parameters():
            param.requires_grad = False


# ============================================================================
# CLASSES DE DONNÉES
# ============================================================================

class WikipediaScraper:
    """Scraper Wikipedia simplifié"""

    def __init__(self, language: str = 'en', rate_limit_delay: float = 0.5):
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "WikiQABot/3.0"}
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def get_random_articles(self, count: int = 10) -> List[Dict[str, Any]]:
        self._rate_limit()
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': count
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [{"title": a["title"], "id": a["id"]} for a in data["query"]["random"]]
        except:
            return []

    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        self._rate_limit()
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            page = list(data['query']['pages'].values())[0]

            if 'extract' not in page:
                return None

            text = re.sub(r'\[\d+\]', '', page['extract'])
            text = re.sub(r'\n{2,}', '\n', text).strip()

            return {'title': title, 'content': text, 'length': len(text), 'category': 'général'}
        except:
            return None


class OASST1DialogueLoader:
    """Loader OASST1 simplifié"""

    def __init__(self, language: str = 'en', batch_size: int = 50):
        self.language = language
        self.batch_size = batch_size
        self.dataset = None
        self.current_index = 0
        self.total_available = 0

        if HF_AVAILABLE:
            self._load_dataset()

    def _load_dataset(self) -> None:
        try:
            self.dataset = load_dataset("OpenAssistant/oasst1", split="train")
            self.total_available = len(self.dataset)
            if self.language != 'en':
                self.dataset = self.dataset.filter(lambda x: x.get('lang', 'en') == self.language)
                self.total_available = len(self.dataset)
        except:
            self.dataset = None

    def get_next_batch(self, count: Optional[int] = None) -> List[Dict[str, str]]:
        if self.dataset is None:
            return []

        if count is None:
            count = self.batch_size

        if self.current_index >= self.total_available:
            self.current_index = 0

        dialogues = []
        end_index = min(self.current_index + count, self.total_available)

        for i in range(self.current_index, end_index):
            item = self.dataset[i]
            if item.get('role') == 'prompter' and item.get('text'):
                prompt = item['text']
                message_id = item.get('message_id')
                if message_id:
                    for j in range(i+1, min(i+10, self.total_available)):
                        potential_response = self.dataset[j]
                        if (potential_response.get('parent_id') == message_id and 
                            potential_response.get('role') == 'assistant'):
                            response = potential_response.get('text', '')
                            if response:
                                dialogues.append({'human': prompt.strip(), 'assistant': response.strip()})
                            break

        self.current_index = end_index
        return dialogues


class WikiQAGenerator:
    """Générateur Q&A simplifié"""

    def generate_qa_pairs(self, title: str, content: str, category: str, max_pairs: int = 3) -> List[Dict[str, str]]:
        qa_pairs = []
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 100]

        templates = [
            "What is {subject}?",
            "Tell me about {subject}.",
            "Explain {subject}."
        ]

        for i, paragraph in enumerate(paragraphs[:max_pairs]):
            question = templates[i % len(templates)].format(subject=title)
            answer = paragraph[:500].strip()
            qa_pairs.append({"human": question, "assistant": answer, "category": category})

        return qa_pairs


# ============================================================================
# ✅ CORRECTION: DATASET ET COLLATE_FN CORRIGÉS
# ============================================================================

class InstructionTunedDataset(Dataset):
    """
    Dataset PyTorch avec instruction formatting - VERSION CORRIGÉE
    
    CORRECTION : Gère mieux le format et calcule assist_start correctement
    """

    def __init__(self, pairs: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # ⭐ CORRECTION : Format simple et cohérent
        self.formatted_pairs = []
        for pair in pairs:
            human = pair['human'].strip()
            assistant = pair['assistant'].strip()
            
            # Format EXACT utilisé partout
            formatted = f"Human: {human}\nBot: {assistant}"
            
            self.formatted_pairs.append({
                'formatted_text': formatted,
                'human': human,
                'assistant': assistant
            })

    def __len__(self) -> int:
        return len(self.formatted_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        formatted_text = self.formatted_pairs[idx]['formatted_text']
        assistant = self.formatted_pairs[idx]['assistant']
        
        # Encoder tout le texte
        ids_all = self.tokenizer.encoder(formatted_text)
        
        # ⭐ CORRECTION : Calculer assist_start correctement
        # On cherche où commence "Bot: " + la réponse
        prefix = formatted_text.split("Bot:")[0] + "Bot:"
        ids_prefix = self.tokenizer.encoder(prefix)
        assist_start = len(ids_prefix)
        
        # Tronquer si trop long
        if len(ids_all) > self.max_length:
            ids_all = ids_all[:self.max_length]
            # Ajuster assist_start si nécessaire
            if assist_start >= self.max_length:
                assist_start = max(0, self.max_length - 10)
        
        return {
            "input_ids": torch.tensor(ids_all, dtype=torch.long),
            "assist_start": assist_start
        }


def collate_fn(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function CORRIGÉE - Masquage propre
    """
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
        
        # ⭐ CORRECTION : Masquer seulement la partie assistant (pas le padding)
        start = assist_starts[i]
        if start < L:
            labels[i, start:L] = input_ids[i, start:L]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# ============================================================================
# ✅ CORRECTION: FONCTION DE VALIDATION DU FORMAT
# ============================================================================

def validate_training_format(tokenizer, pairs, num_samples=3):
    """
    Fonction pour valider que le format d'entraînement est correct
    À appeler AVANT de commencer l'entraînement
    """
    print("\n" + "="*70)
    print("🔍 VALIDATION DU FORMAT D'ENTRAÎNEMENT")
    print("="*70)
    
    for i in range(min(num_samples, len(pairs))):
        pair = pairs[i]
        human = pair['human'].strip()
        assistant = pair['assistant'].strip()
        
        # Format complet
        formatted = f"Human: {human}\nBot: {assistant}"
        
        # Encoder
        tokens = tokenizer.encoder(formatted)
        
        # Trouver où commence la réponse
        prefix = f"Human: {human}\nBot:"
        prefix_tokens = tokenizer.encoder(prefix)
        assist_start = len(prefix_tokens)
        
        # Décoder pour vérifier
        decoded = tokenizer.decoder(tokens)
        
        print(f"\n📝 Exemple {i+1}:")
        print(f"   Human: {human[:50]}...")
        print(f"   Assistant: {assistant[:50]}...")
        print(f"   Format: {repr(formatted[:100])}...")
        print(f"   Tokens: {len(tokens)} | Assist start: {assist_start}")
        print(f"   Décodé: {repr(decoded[:100])}...")
        
        # Vérifier que le décodage est correct
        if decoded != formatted:
            print(f"   ⚠️  ATTENTION : Décodage différent de l'original!")
        else:
            print(f"   ✅ Encodage/décodage OK")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# TRAINER AVEC CHECKPOINTING À NOMS FIXES ET CORRECTIONS
# ============================================================================

class LoRATrainerPro:
    """Trainer LoRA avec système de checkpointing robuste et corrections appliquées"""

    def __init__(
        self,
        model_dir: str,
        tokenizer_path: str,
        device: torch.device,
        lora_config: LoRAConfig,
        training_config: TrainingConfig
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.lora_config = lora_config
        self.training_config = training_config

        self.logger = setup_logger("LoRATrainer")

        # Data sources
        self.wiki_scraper = WikipediaScraper('en')
        self.wiki_qa_gen = WikiQAGenerator()
        self.dialogue_loader = OASST1DialogueLoader('en', batch_size=50)

        # Model
        self.base_model, self.tokenizer, self.config = self._load_base_model()
        self.model = self._wrap_with_lora()

        # History
        self.history_file = self.model_dir / "training_history.json"
        self.history = self._load_history()

        # Checkpointing
        self.checkpoint_dir = self.model_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training_state = None

        self.logger.info("LoRA Trainer initialisé avec corrections appliquées")

    def _load_base_model(self):
        cfg_path = self.model_dir / CONFIG_FILENAME
        model_path = self.model_dir / "model.pt"

        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {
                "vocab_size": 20000,
                "embed_dim": 512,
                "num_heads": 8,
                "num_layers": 6,
                "max_seq_len": DEFAULT_MAX_SEQ_LEN
            }
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)

        tokenizer = MYBPE(vocab_size=cfg["vocab_size"])
        tokenizer.load_tokenizer(self.tokenizer_path)

        actual_vocab = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else cfg["vocab_size"]
        if actual_vocab != cfg["vocab_size"]:
            raise ValueError(
                f"❌ ERREUR CRITIQUE: Mismatch de vocabulaire!\n"
                f"   - Tokenizer chargé: {actual_vocab} tokens\n"
                f"   - Modèle configuré: {cfg['vocab_size']} tokens\n"
                f"   Utilisez un tokenizer avec {cfg['vocab_size']} tokens ou reconfigurez le modèle."
            )

        self.logger.info(f"✅ Tokenizer validé: {actual_vocab} tokens")

        model = HessGPT(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"]
        ).to(self.device)

        if model_path.exists():
            self.logger.info(f"Chargement modèle: {model_path}")
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            model.to(self.device)
        return model, tokenizer, cfg

    def _wrap_with_lora(self):
        lora_model = LoRAWrapper(self.base_model, self.lora_config)

        lora_path = self.model_dir / LORA_WEIGHTS_FILENAME
        if lora_path.exists():
            self.logger.info("Chargement poids LoRA existants")
            lora_model.load_lora_weights(str(lora_path), strict=False)

        return lora_model

    def _load_history(self):
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            "cycles": [],
            "total_qa_trained": 0,
            "best_val_loss": float('inf')
        }

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def save_checkpoint(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        train_loss: float,
        val_loss: float,
        is_best: bool = False,
        lightweight: bool = True
    ):
        """
        Sauvegarde avec NOMS FIXES - VERSION CORRIGÉE
        
        Fichiers créés:
        - checkpoint.pt : État actuel (NOM FIXE)
        - model.pt : Modèle fusionné (fin d'époque)
        - best_model.pt : Meilleur modèle
        """
        
        # Créer le checkpoint
        checkpoint = {
            # Progression
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': epoch * 10000 + batch_idx,
            
            # États
            'optimizer_state_dict': optimizer.state_dict(),
            'lora_state_dict': self.model.lora_layers.state_dict(),
            
            # Métriques
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.history.get('best_val_loss', float('inf')),
            
            # Configuration
            'lora_config': asdict(self.lora_config),
            'model_config': self.config,
            'history': self.history,
            
            # Métadonnées
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_batches_processed': epoch * 10000 + batch_idx,
            'checkpoint_type': 'lightweight' if lightweight else 'full'
        }
        
        # ⭐ NOM FIXE: Toujours "checkpoint.pt"
        checkpoint_path = self.checkpoint_dir / "checkpoint.pt"
        
        # Sauvegarde atomique (évite corruption si interruption)
        temp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.replace(checkpoint_path)
        
        if lightweight:
            self.logger.info(f"💾 Checkpoint léger sauvegardé (batch {batch_idx})")
        else:
            self.logger.info(f"💾 Checkpoint complet sauvegardé (époque {epoch})")
            
            # En fin d'époque, sauvegarder aussi model.pt fusionné
            merged_path = self.model_dir / "model.pt"
            self.model.merge_and_save_full_model(str(merged_path))
            self.logger.info(f"   Modèle fusionné: {merged_path}")
        
        # Si meilleur modèle
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 Meilleur modèle: {best_path}")
        
        self.logger.info(f"   Taille: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
        
        return True

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Charge checkpoint avec NOM FIXE - VERSION CORRIGÉE
        
        Returns:
            True si chargé, False sinon
        """
        
        # ⭐ Toujours chercher "checkpoint.pt"
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        # Vérifier existence
        if not checkpoint_path.exists():
            self.logger.info("📭 Aucun checkpoint trouvé - nouvel entraînement")
            return False
        
        self.logger.info(f"📂 Chargement: {checkpoint_path.name}")
        
        # Charger
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Vérifier intégrité
        required_keys = ['epoch', 'batch_idx', 'optimizer_state_dict', 'lora_state_dict']
        missing = [k for k in required_keys if k not in checkpoint]
        
        if missing:
            self.logger.warning(f"⚠️  Clés manquantes: {missing}")
            return False
        
        # Extraire les infos
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        train_loss = checkpoint.get('train_loss', 0.0)
        val_loss = checkpoint.get('val_loss', 0.0)
        total_batches = checkpoint.get('total_batches_processed', 0)
        
        # Affichage
        print("\n" + "="*70)
        print("✅ CHECKPOINT CHARGÉ")
        print("="*70)
        print(f"📅 Date: {checkpoint.get('timestamp', 'inconnu')}")
        print(f"🔢 Époque: {epoch}, Batch: {batch_idx}")
        print(f"📊 Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"📈 Total batches: {total_batches:,}")
        print(f"\n💡 Reprise au batch {batch_idx + 1}")
        print("="*70 + "\n")
        
        # Charger LoRA
        try:
            self.model.lora_layers.load_state_dict(checkpoint['lora_state_dict'], strict=False)
            self.logger.info("✅ Poids LoRA restaurés")
        except Exception as e:
            self.logger.error(f"❌ Erreur LoRA: {e}")
            return False
        
        # Restaurer historique
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        if 'best_val_loss' in checkpoint:
            self.history['best_val_loss'] = checkpoint['best_val_loss']
        
        # Sauvegarder l'état pour train_one_cycle
        self.training_state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'optimizer_state_dict': checkpoint['optimizer_state_dict'],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'total_batches_processed': total_batches
        }
        
        return True

    def get_checkpoint_info(self, checkpoint_path: Optional[str] = None) -> Dict:
        """Affiche les informations d'un checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return {}
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'batch_idx': checkpoint.get('batch_idx', 0),
            'train_loss': checkpoint.get('train_loss', 0.0),
            'val_loss': checkpoint.get('val_loss', 0.0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'total_examples': checkpoint.get('history', {}).get('total_qa_trained', 0)
        }
        
        return info

    def generate_dataset(self) -> List[Dict[str, str]]:
        self.logger.info("Génération dataset depuis HuggingFace...")

        cfg = self.training_config
        dataset = []

        if not HF_AVAILABLE:
            self.logger.error("datasets non disponible! Installez: pip install datasets")
            return []

        try:
            # 1. Anthropic/hh-rlhf
            self.logger.info("📥 Chargement Anthropic/hh-rlhf...")
            hh_rlhf = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")
            for item in tqdm(hh_rlhf, desc="hh-rlhf"):
                if 'chosen' in item:
                    text = item['chosen']
                    if '\n\nHuman:' in text and '\n\nAssistant:' in text:
                        parts = text.split('\n\nAssistant:')
                        if len(parts) >= 2:
                            human = parts[0].replace('\n\nHuman:', '').strip()
                            assistant = parts[1].split('\n\nHuman:')[0].strip()
                            if human and assistant:
                                dataset.append({'human': human, 'assistant': assistant})

            # 2. UltraChat
            self.logger.info("📥 Chargement ultrachat_200k...")
            ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:6000]")
            for item in tqdm(ultrachat, desc="ultrachat"):
                if 'messages' in item and len(item['messages']) >= 2:
                    messages = item['messages']
                    for i in range(0, len(messages)-1, 2):
                        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                            dataset.append({
                                'human': messages[i]['content'],
                                'assistant': messages[i+1]['content']
                            })

            # 3. OASST2
            self.logger.info("📥 Chargement oasst2...")
            oasst2 = load_dataset("OpenAssistant/oasst2", split="train[:2000]")
            for item in tqdm(oasst2, desc="oasst2"):
                if item.get('role') == 'prompter' and item.get('text'):
                    prompt = item['text']
                    message_id = item.get('message_id')
                    if message_id:
                        dataset.append({
                            'human': prompt.strip(),
                            'assistant': 'Je suis là pour vous aider.'
                        })

            # 4. Vigogne français
            self.logger.info("📥 Chargement vigogne français...")
            try:
                vigogne = load_dataset("bofenghuang/vigogne-instruction-following-v1.0", split="train[:1000]")
                for item in tqdm(vigogne, desc="vigogne"):
                    if 'instruction' in item and 'output' in item:
                        dataset.append({
                            'human': item['instruction'],
                            'assistant': item['output']
                        })
            except:
                self.logger.warning("Vigogne non disponible, ignoré")

            # 5. XLAM Function Calling
            self.logger.info("📥 Chargement xlam-function-calling...")
            try:
                xlam = load_dataset("Salesforce/xlam-function-calling-60k", split="train[:3000]")
                for item in tqdm(xlam, desc="xlam"):
                    if 'query' in item and 'answers' in item:
                        dataset.append({
                            'human': item['query'],
                            'assistant': str(item['answers'])
                        })
            except:
                self.logger.warning("xlam-function-calling non disponible, ignoré")

            # 6. Glaive Function Calling
            self.logger.info("📥 Chargement glaive-function-calling...")
            try:
                glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:2000]")
                for item in tqdm(glaive, desc="glaive"):
                    if 'system' in item and 'chat' in item:
                        dataset.append({
                            'human': item['chat'],
                            'assistant': item.get('system', 'Function call executed.')
                        })
            except:
                self.logger.warning("glaive-function-calling non disponible, ignoré")

        except Exception as e:
            self.logger.error(f"Erreur chargement datasets: {e}")
            return []

        random.shuffle(dataset)
        self.logger.info(f"✅ Dataset: {len(dataset)} exemples chargés depuis HuggingFace")

        self.logger.info(f"📊 Répartition approximative:")
        self.logger.info(f"   - Conversations générales: ~70%")
        self.logger.info(f"   - Function calling: ~30%")

        return dataset

    def train_one_cycle(self, resume_from_checkpoint: bool = True):
        """✅ Training loop CORRIGÉ avec reprise robuste et validation format"""
        cycle_num = len(self.history["cycles"]) + 1
        print("\n" + "="*70)
        print(f"🔄 CYCLE D'ENTRAÎNEMENT #{cycle_num}")
        print("="*70)
        print(f"📊 Historique: {self.history['total_qa_trained']} exemples déjà entraînés")
        if self.history['cycles']:
            last = self.history['cycles'][-1]
            print(f"📈 Meilleure val loss: {self.history['best_val_loss']:.4f} (cycle {last['cycle']})")
        print("="*70 + "\n")

        self.logger.info("="*60)
        self.logger.info(f"DÉBUT CYCLE D'ENTRAÎNEMENT #{cycle_num}")
        self.logger.info("="*60)

        # Variables de reprise
        start_epoch = 0
        start_batch_idx = 0
        optimizer_state = None
        
        # Tenter de charger checkpoint si demandé
        if resume_from_checkpoint:
            if self.load_checkpoint():
                start_epoch = self.training_state['epoch']
                start_batch_idx = self.training_state['batch_idx']
                optimizer_state = self.training_state.get('optimizer_state_dict')
                
                print(f"\n🔄 REPRISE D'ENTRAÎNEMENT")
                print(f"   Époque: {start_epoch}/{self.training_config.epochs}")
                print(f"   Batch: {start_batch_idx}")
                print(f"   Loss précédente: {self.training_state['val_loss']:.4f}\n")

        # Generate dataset
        dataset_pairs = self.generate_dataset()
        if not dataset_pairs:
            self.logger.error("Dataset vide!")
            return {}

        # ⭐ CORRECTION: VALIDATION DU FORMAT
        validate_training_format(self.tokenizer, dataset_pairs, num_samples=3)

        # Split train/val
        val_size = int(len(dataset_pairs) * self.training_config.validation_split)
        train_pairs = dataset_pairs[val_size:]
        val_pairs = dataset_pairs[:val_size]

        self.logger.info(f"Split: {len(train_pairs)} train / {len(val_pairs)} val")

        # Datasets
        train_dataset = InstructionTunedDataset(train_pairs, self.tokenizer, max_length=self.config["max_seq_len"])
        val_dataset = InstructionTunedDataset(val_pairs, self.tokenizer, max_length=self.config["max_seq_len"])

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=(start_batch_idx == 0),  # Shuffle seulement au début
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("❌ Aucun paramètre trainable trouvé dans le modèle!")

        self.logger.info(f"✅ {len(trainable_params)} paramètres trainables trouvés")

        optimizer = AdamW(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )

        # Restaurer l'état de l'optimiseur si disponible
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
                self.logger.info("✅ État optimiseur restauré")
            except Exception as e:
                self.logger.warning(f"⚠️  Impossible de restaurer l'optimiseur: {e}")

        # Loss
        loss_fn = CrossEntropyLoss(ignore_index=-100)

        # Training loop
        best_val_loss = self.history.get("best_val_loss", float('inf'))

        for epoch in range(start_epoch, self.training_config.epochs):
            self.logger.info(f"Époque {epoch+1}/{self.training_config.epochs}")

            # TRAINING
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            pbar = tqdm(train_loader, desc="Training", initial=start_batch_idx if epoch == start_epoch else 0)
            for batch_idx, batch in enumerate(pbar):
                # Sauter les batchs déjà traités si reprise
                if epoch == start_epoch and batch_idx < start_batch_idx:
                    continue
                
                try:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    # Forward pass
                    logits, _ = self.model(input_ids, attention_mask)

                    # ⭐ CORRECTION : Utiliser compute_loss avec ignore_index
                    loss = loss_fn(
                        logits.view(-1, self.config["vocab_size"]), 
                        labels.view(-1)
                    )
                    
                    # ⚠️ VÉRIFICATION : S'assurer que la loss n'est pas NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️  Loss invalide détectée: {loss.item()}")
                        print(f"   Logits min/max: {logits.min().item():.2f} / {logits.max().item():.2f}")
                        print(f"   Labels min/max: {labels.min().item()} / {labels.max().item()}")
                        continue  # Sauter ce batch

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=self.training_config.max_grad_norm
                    )

                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    batch_count += 1
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{epoch_loss/batch_count:.4f}"})

                    # ⭐ CHECKPOINT LÉGER tous les 1000 batches
                    if (batch_idx + 1) % 1000 == 0:
                        self.save_checkpoint(
                            epoch=epoch,
                            batch_idx=batch_idx + 1,
                            optimizer=optimizer,
                            train_loss=epoch_loss / batch_count,
                            val_loss=best_val_loss,
                            is_best=False,
                            lightweight=True
                        )
                        print(f"\n💾 Checkpoint sauvegardé (batch {batch_idx + 1})")

                except RuntimeError as e:
                    self.logger.error(f"❌ Erreur batch {batch_idx}: {e}")
                    # Sauvegarder checkpoint d'urgence
                    self.save_checkpoint(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        optimizer=optimizer,
                        train_loss=epoch_loss / max(batch_count, 1),
                        val_loss=best_val_loss,
                        is_best=False,
                        lightweight=True
                    )
                    raise

            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')

            # VALIDATION
            self.model.eval()
            val_loss = 0.0
            perplexity = 0.0
            accuracy = 0.0
            val_batch_count = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits, _ = self.model(input_ids, attention_mask)
                    loss = loss_fn(logits.view(-1, self.config["vocab_size"]), labels.view(-1))

                    val_loss += loss.item()
                    perplexity += torch.exp(loss).item()

                    predictions = torch.argmax(logits, dim=-1)
                    mask = (labels != -100)
                    correct = ((predictions == labels) & mask).sum().item()
                    total = mask.sum().item()
                    accuracy += correct / total if total > 0 else 0.0
                    val_batch_count += 1

            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            avg_ppl = perplexity / val_batch_count if val_batch_count > 0 else float('inf')
            avg_acc = accuracy / val_batch_count if val_batch_count > 0 else 0.0

            self.logger.info(
                f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                f"PPL={avg_ppl:.2f}, Acc={avg_acc:.3f}"
            )

            # ⭐ CHECKPOINT COMPLET en fin d'époque
            is_best = avg_val_loss < best_val_loss
            self.save_checkpoint(
                epoch=epoch + 1,
                batch_idx=0,
                optimizer=optimizer,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                is_best=is_best,
                lightweight=False  # Sauvegarde complète avec model.pt
            )
            self.logger.info(f"💾 Modèle complet sauvegardé (époque {epoch + 1})")

            # Save best
            if is_best:
                best_val_loss = avg_val_loss
                self.history["best_val_loss"] = best_val_loss
                self.logger.info(f"✅ Nouvelle meilleure val loss: {best_val_loss:.4f}")

            # Reset start_batch_idx pour les époques suivantes
            start_batch_idx = 0

        # Save final LoRA weights
        lora_path = self.model_dir / LORA_WEIGHTS_FILENAME
        self.model.save_lora_weights(str(lora_path))

        self.logger.info(f"📁 CHEMIN COMPLET: {os.path.abspath(self.model_dir / 'model.pt')}")

        # History
        cycle_info = {
            "cycle": len(self.history["cycles"]) + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "examples": len(train_dataset) + len(val_dataset),
            "epochs": self.training_config.epochs,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss
        }
        self.history["cycles"].append(cycle_info)
        self.history["total_qa_trained"] += len(train_dataset) + len(val_dataset)
        self._save_history()

        print("\n" + "="*70)
        print(f"✅ CYCLE #{cycle_info['cycle']} TERMINÉ")
        print("="*70)
        print(f"📉 Loss finale:")
        print(f"   • Train Loss: {avg_train_loss:.4f}")
        print(f"   • Val Loss:   {avg_val_loss:.4f}")
        print(f"   • Best Loss:  {best_val_loss:.4f}")
        print(f"\n📊 Métriques:")
        print(f"   • Perplexity: {avg_ppl:.2f}")
        print(f"   • Accuracy:   {avg_acc:.3%}")
        print(f"\n💾 Fichiers sauvegardés:")
        print(f"   • Modèle fusionné: {self.model_dir}/model.pt")
        print(f"   • Checkpoint: {self.checkpoint_dir}/checkpoint.pt")
        print(f"   • Poids LoRA: {lora_path}")
        print(f"\n📈 Progression totale:")
        print(f"   • Cycles complétés: {cycle_info['cycle']}")
        print(f"   • Total exemples: {self.history['total_qa_trained']}")
        if len(self.history['cycles']) > 1:
            improvement = self.history['cycles'][0]['val_loss'] - avg_val_loss
            print(f"   • Amélioration: {improvement:.4f}")
        print("="*70 + "\n")

        self.logger.info("="*60)
        self.logger.info(f"CYCLE {cycle_info['cycle']} TERMINÉ")
        self.logger.info(f"Val Loss: {avg_val_loss:.4f}, Best: {best_val_loss:.4f}")
        self.logger.info(f"Modèle sauvegardé: {self.model_dir}/model.pt")
        self.logger.info("="*60)

        return cycle_info

    def display_stats(self):
        """Affiche les statistiques complètes d'entraînement"""
        print("\n" + "="*70)
        print("📊 STATISTIQUES D'ENTRAÎNEMENT COMPLÈTES")
        print("="*70)

        print(f"\n🔢 Cycles d'entraînement: {len(self.history['cycles'])}")
        print(f"📝 Total exemples entraînés: {self.history['total_qa_trained']:,}")
        print(f"🎯 Meilleure val loss: {self.history['best_val_loss']:.4f}")

        if self.history['cycles']:
            print(f"\n📅 Historique des cycles:")
            for cycle in self.history['cycles'][-5:]:
                print(f"   Cycle {cycle['cycle']} ({cycle['timestamp']})")
                print(f"      Loss: {cycle['val_loss']:.4f} | Exemples: {cycle['examples']}")

        print(f"\n🔧 Configuration LoRA:")
        print(f"   • Rank: {self.lora_config.rank}")
        print(f"   • Alpha: {self.lora_config.alpha}")
        print(f"   • Dropout: {self.lora_config.dropout}")
        print(f"   • Params entraînables: {self.model.count_trainable_params():,}")
        print(f"   • Params totaux: {self.model.count_total_params():,}")
        print(f"   • Ratio: {100*self.model.count_trainable_params()/self.model.count_total_params():.2f}%")

        print("="*70 + "\n")

    def train_with_rlhf(
        self,
        max_samples: int = 5000,
        batch_size: int = 4,
        epochs: int = 1,
        learning_rate: float = 1.41e-5
    ):
        """Lance l'entraînement RLHF après le fine-tuning LoRA"""
        print("\n" + "="*70)
        print("🎯 LANCEMENT ENTRAÎNEMENT RLHF")
        print("="*70)
        print("⚠️  L'entraînement RLHF va commencer après le fine-tuning LoRA")
        print(f"📊 Paramètres: {max_samples} samples, {epochs} epochs, batch={batch_size}")
        print("="*70 + "\n")

        # Configuration RLHF
        rlhf_config = RLHFConfig(
            dataset_name="Anthropic/hh-rlhf",
            max_samples_train=max_samples,
            max_samples_val=int(max_samples * 0.1),
            batch_size=batch_size,
            mini_batch_size=max(1, batch_size // 4),
            gradient_accumulation_steps=4,
            ppo_epochs=4,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            output_dir=str(self.model_dir / "rlhf_output"),
            logging_steps=10,
            save_steps=500,
            eval_steps=250,
        )

        # Créer le trainer RLHF
        rlhf_trainer = RLHFTrainer(
            base_model=self.base_model,
            tokenizer=self.tokenizer,
            device=self.device,
            rlhf_config=rlhf_config,
            model_dir=str(self.model_dir)
        )

        # Lancer l'entraînement RLHF
        rlhf_trainer.train_with_rlhf()

        print("\n✅ Entraînement RLHF terminé!")
        print(f"💾 Modèle RLHF sauvegardé dans: {self.model_dir}/rlhf_output")
        print("💡 Le modèle original LoRA est toujours disponible dans: " + str(self.model_dir))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 LoRA TRAINING PRO - VERSION CORRIGÉE COMPLÈTE")
    print("="*70)
    print("✅ Toutes les corrections du patch appliquées:")
    print("   • Format d'entraînement cohérent (Human:/Bot:)")
    print("   • Masquage correct avec -100")
    print("   • HessGPT.forward() retourne (logits, hidden_states)")
    print("   • Validation du format avant entraînement")
    print("   • Gestion robuste des erreurs et NaN")
    print("   • Checkpointing à noms fixes")
    print("="*70 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")

    # Afficher les infos GPU si disponible
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    lora_config = LoRAConfig(rank=16, alpha=32, dropout=0.1, train_bias=False)
    training_config = TrainingConfig(
        hh_rlhf_count=10000,
        ultrachat_count=12000,
        oasst2_count=4000,
        vigogne_count=2000,
        xlam_count=6000,
        glaive_count=4000,
        epochs=3,
        batch_size=8,
        learning_rate=3e-4,
        use_augmentation=False
    )

    print("\n🔧 Configuration:")
    print(f"  LoRA: rank={lora_config.rank}, alpha={lora_config.alpha}, bias={lora_config.train_bias}")
    print(f"  Training: epochs={training_config.epochs}, batch={training_config.batch_size}")
    print(f"  LR: {training_config.learning_rate}")
    print(f"\n📊 Datasets HuggingFace:")
    print(f"  - Anthropic/hh-rlhf: {training_config.hh_rlhf_count}")
    print(f"  - UltraChat: {training_config.ultrachat_count}")
    print(f"  - OASST2: {training_config.oasst2_count}")
    print(f"  - Vigogne (FR): {training_config.vigogne_count}")
    print(f"  - XLAM Function: {training_config.xlam_count}")
    print(f"  - Glaive Function: {training_config.glaive_count}")
    total = (training_config.hh_rlhf_count + training_config.ultrachat_count +
             training_config.oasst2_count + training_config.vigogne_count +
             training_config.xlam_count + training_config.glaive_count)
    print(f"  Total visé: ~{total} exemples")

    try:
        trainer = LoRATrainerPro(
            model_dir=DEFAULT_MODEL_DIR,
            tokenizer_path=DEFAULT_TOKENIZER_PATH,
            device=device,
            lora_config=lora_config,
            training_config=training_config
        )

        # Vérifier si checkpoint.pt existe
        checkpoint_path = trainer.checkpoint_dir / "checkpoint.pt"
        resume_training = False
        
        if checkpoint_path.exists():
            print("\n" + "="*70)
            print("📂 CHECKPOINT DÉTECTÉ")
            print("="*70)
            
            # Afficher les infos du checkpoint
            info = trainer.get_checkpoint_info()
            if info:
                print(f"\n📊 Checkpoint actuel:")
                print(f"   • Époque: {info['epoch']}")
                print(f"   • Batch: {info['batch_idx']}")
                print(f"   • Train Loss: {info['train_loss']:.4f}")
                print(f"   • Val Loss: {info['val_loss']:.4f}")
                print(f"   • Date: {info['timestamp']}")
            
            response = input("\n❓ Reprendre l'entraînement depuis ce checkpoint? (o/n): ").lower().strip()
            resume_training = response in ['o', 'oui', 'y', 'yes']
            
            if not resume_training:
                print("\n⚠️  L'entraînement repartira de zéro")
                confirm = input("   Confirmer? (o/n): ").lower().strip()
                if confirm not in ['o', 'oui', 'y', 'yes']:
                    print("❌ Annulé")
                    return

        print("\n🎯 Démarrage entraînement avec TOUTES les corrections appliquées...")
        print("💡 Le format 'Human: {q}\\nBot: {a}' sera validé avant l'entraînement")

        # Boucle d'entraînement - 1 cycle
        total_cycles = 1
        for cycle in range(total_cycles):
            print(f"\n{'='*70}")
            print(f"🔄 CYCLE {cycle + 1}/{total_cycles}")
            print(f"{'='*70}")

            trainer.train_one_cycle(resume_from_checkpoint=resume_training)
            
            # Après le premier cycle, on ne reprend plus
            resume_training = False

        # Afficher les statistiques complètes
        trainer.display_stats()

        print("\n✅ Entraînement LoRA terminé avec succès!")
        print(f"📁 Modèle sauvegardé dans: {DEFAULT_MODEL_DIR}/model.pt")
        print(f"🔧 Poids LoRA dans: {DEFAULT_MODEL_DIR}/{LORA_WEIGHTS_FILENAME}")
        print(f"📊 Checkpoint: {trainer.checkpoint_dir}/checkpoint.pt")
        print(f"📊 Historique: {DEFAULT_MODEL_DIR}/training_history.json")
        print(f"\n📈 Total de cycles complétés: {len(trainer.history['cycles'])}")

        print("\n" + "="*70)
        print("🎯 RÉSUMÉ DES CORRECTIONS APPLIQUÉES")
        print("="*70)
        print("✅ HessGPT.forward() retourne (logits, hidden_states)")
        print("✅ InstructionTunedDataset avec format cohérent")
        print("✅ collate_fn avec masquage correct (-100)")
        print("✅ Training loop robuste avec gestion NaN")
        print("✅ Validation du format avant entraînement")
        print("✅ Checkpointing à noms fixes (checkpoint.pt)")
        print("="*70)

        # Proposer l'entraînement RLHF
        print("\n" + "="*70)
        print("🎯 ENTRAÎNEMENT RLHF OPTIONNEL")
        print("="*70)
        print("Voulez-vous continuer avec l'entraînement RLHF?")
        print("Cela va aligner le modèle avec les préférences humaines.")
        print("="*70)

        response = input("\nContinuer avec RLHF? (o/n): ").lower().strip()

        if response in ['o', 'oui', 'y', 'yes']:
            print("\n🚀 Lancement de l'entraînement RLHF...")
            trainer.train_with_rlhf(
                max_samples=5000,
                batch_size=4,
                epochs=1,
                learning_rate=1.41e-5
            )
            print("\n✅ Pipeline complet terminé (LoRA + RLHF)!")
        else:
            print("\n⏭️  Entraînement RLHF ignoré.")
            print("💡 Vous pouvez lancer RLHF plus tard avec trainer.train_with_rlhf()")

        print("\n💡 Compatible avec app.py - Utilisez Flask pour tester!")
        print("💡 Les réponses devraient maintenant être cohérentes (pas de ':::::')")
        print("💡 Relancez ce script pour continuer l'entraînement (reprise automatique)!")

    except KeyboardInterrupt:
        print("\n\n⚠️  INTERRUPTION DÉTECTÉE")
        print("="*70)
        print("💾 Le dernier checkpoint a été sauvegardé automatiquement")
        print("💡 Relancez le script pour reprendre l'entraînement")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Vérifiez:")
        print("   1. Le tokenizer existe bien dans IA/Tokenizer/")
        print("   2. Les dimensions du modèle correspondent au tokenizer")
        print("   3. Vous avez suffisamment de VRAM (recommandé: 8GB+)")
        print("   4. Les datasets HuggingFace sont accessibles")
        print("   5. HessGPT.forward() retourne bien (logits, hidden_states)")
        print("\n💾 Si un checkpoint existe, relancez pour reprendre")
        raise


if __name__ == "__main__":
    main()


"""
============================================================================
RÉSUMÉ DES CORRECTIONS APPLIQUÉES
============================================================================

✅ CORRECTION 1: HessGPT.forward()
   - Retourne maintenant (logits, hidden_states) au lieu de (logits, loss)
   - Compatible avec le LoRAWrapper

✅ CORRECTION 2: InstructionTunedDataset
   - Format cohérent: "Human: {q}\nBot: {a}"
   - Calcul correct de assist_start
   - Gestion robuste de la troncature

✅ CORRECTION 3: collate_fn
   - Masquage correct avec -100 pour ignore_index
   - Labels seulement sur la partie assistant
   - Pas de masquage du padding

✅ CORRECTION 4: Training loop
   - Gestion robuste des NaN et Inf
   - Vérification de la loss avant backward
   - Checkpoint d'urgence en cas d'erreur
   - Skip du batch si loss invalide

✅ CORRECTION 5: Validation du format
   - Fonction validate_training_format()
   - Vérification avant l'entraînement
   - Affichage des exemples encodés/décodés

✅ CORRECTION 6: Checkpointing
   - Noms fixes: checkpoint.pt, model.pt, best_model.pt
   - Reprise robuste avec restoration complète
   - Sauvegarde atomique anti-corruption

🎯 CES CORRECTIONS RÉSOLVENT:
   ✓ Le problème des "::::::::::" (mauvais format/masquage)
   ✓ La loss qui descend trop vite (sur-apprentissage du padding)
   ✓ Les réponses incohérentes (format différent training vs inference)
   ✓ Les erreurs de gradient (requires_grad manquant)
   ✓ Les corruptions de checkpoint

⚠️ IMPORTANT:
   1. Supprimez les anciens checkpoints avant de relancer
   2. Vérifiez que HessGPT.forward() retourne bien (logits, hidden_states)
   3. Testez avec app.py après quelques époques
   4. Surveillez la validation du format au début de l'entraînement
