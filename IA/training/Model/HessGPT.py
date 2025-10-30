#HessGPT.py - VERSION CORRIGÉE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TransformerBlock.transformer_block import TransformerBlock

class HessGPT(nn.Module):
    """
    Modèle HessGPT - Architecture Transformer personnalisée
    VERSION CORRIGÉE : Retourne (logits, hidden_states) au lieu de (logits, loss)
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=1024,
        dropout=0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer Norm finale
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Output Head
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Partager les poids
        self.output_head.weight = self.token_embeddings.weight
        
        # Initialisation
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialisation des poids"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None):
        """
        CORRECTION CRITIQUE : Retourne (logits, hidden_states) pour compatibilité LoRA
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optionnel)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            hidden_states: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Embeddings
        token_embeds = self.token_embeddings(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device)
        position_embeds = self.position_embeddings(positions)
        x = self.dropout(token_embeds + position_embeds)
        
        # 2. Créer le masque causal
        mask = self.create_causal_mask(seq_len, device=input_ids.device)
        
        # 3. Passer à travers tous les Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # 4. Layer Norm finale
        hidden_states = self.ln_final(x)
        
        # 5. Output Head
        logits = self.output_head(hidden_states)
        
        # ⭐ CORRECTION : Retourner (logits, hidden_states) au lieu de (logits, loss)
        return logits, hidden_states
    
    def create_causal_mask(self, seq_len, device):
        """Crée un masque causal triangulaire"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def compute_loss(self, logits, targets, ignore_index=-100):
        """
        Calcule la loss séparément (pour entraînement)
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len]
            ignore_index: Index à ignorer (pour padding/masking)
        
        Returns:
            loss: Scalar
        """
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=ignore_index
        )
        return loss
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=0.9, repetition_penalty=1.0):
        """
        Génération de texte (autoregressive) - VERSION CORRIGÉE
        
        Args:
            input_ids: [batch_size, seq_len]
            max_new_tokens: Nombre de tokens à générer
            temperature: Contrôle la randomness
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Pénalité pour tokens répétés
        
        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Tronquer si trop long
                input_ids_cond = generated if generated.size(1) <= self.max_seq_len else generated[:, -self.max_seq_len:]
                
                # Forward pass - ⭐ UTILISE LA NOUVELLE SIGNATURE
                logits, _ = self.forward(input_ids_cond)
                
                # Prendre les logits du dernier token
                next_logits = logits[:, -1, :] / temperature
                
                # Appliquer repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(generated[0].tolist()):
                        next_logits[0, token_id] /= repetition_penalty
                
                # Top-k sampling
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float('inf')
                
                # Softmax
                probs = F.softmax(next_logits, dim=-1)
                
                # Top-p (nucleus) sampling
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Masquer les tokens au-delà de top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sampler le prochain token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Ajouter à la séquence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# ============================================
# TESTS
# ============================================

def test_hessgpt_forward():
    """Test du forward corrigé"""
    print("\n" + "="*60)
    print("TEST: HessGPT Forward (CORRIGÉ)")
    print("="*60)
    
    vocab_size = 300
    batch_size = 2
    seq_len = 10
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # ⭐ VÉRIFIER LA NOUVELLE SIGNATURE
    logits, hidden_states = model(input_ids)
    
    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Hidden states shape: {hidden_states.shape}")
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert hidden_states.shape == (batch_size, seq_len, model.embed_dim)
    
    print(f"\n✅ Forward corrigé : retourne bien (logits, hidden_states)")
    
    # Test de la loss séparée
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = model.compute_loss(logits, targets)
    print(f"✓ Loss: {loss.item():.4f}")


def test_generation_fixed():
    """Test de génération avec nouvelle API"""
    print("\n" + "="*60)
    print("TEST: Génération (CORRIGÉE)")
    print("="*60)
    
    vocab_size = 300
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2
    )
    
    prompt = torch.randint(0, vocab_size, (1, 5))
    
    print(f"✓ Prompt: {prompt[0].tolist()}")
    
    # Génération avec nouveaux paramètres
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.9,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    print(f"✓ Generated: {generated[0].tolist()}")
    print(f"✓ Longueur: {generated.shape[1]} tokens")


if __name__ == "__main__":
    print("\n🚀 TESTS DU MODÈLE HessGPT CORRIGÉ\n")
    
    test_hessgpt_forward()
    test_generation_fixed()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("="*60)
    print("\n💡 Changements principaux:")
    print("   1. forward() retourne (logits, hidden_states)")
    print("   2. compute_loss() séparé pour entraînement")
    print("   3. generate() mis à jour avec nouvelle signature")
    print("="*60 + "\n")