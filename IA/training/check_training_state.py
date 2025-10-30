#!/usr/bin/env python3
"""
Script de vérification d'état d'entraînement - VERSION CORRIGÉE
Usage: python check_training_state.py
#check_training_state.py
Vérifie les fichiers avec NOMS FIXES:
- checkpoint.pt : État d'entraînement actuel
- model.pt : Modèle fusionné
- best_model.pt : Meilleur modèle
"""
import sys
import os
from pathlib import Path
import torch

# Chemins
CHECKPOINT_DIR = Path("./IA/saved_models/my_llm/checkpoints")
MODEL_DIR = Path("./IA/saved_models/my_llm")

def format_size(bytes_size):
    """Formate la taille en unités lisibles"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def check_training_state():
    """Vérifie et affiche l'état d'entraînement complet"""
    print("\n" + "="*70)
    print("🔍 VÉRIFICATION ÉTAT D'ENTRAÎNEMENT")
    print("="*70)
    
    # ========================================
    # 1. VÉRIFIER checkpoint.pt (priorité)
    # ========================================
    checkpoint_path = CHECKPOINT_DIR / "checkpoint.pt"
    
    if checkpoint_path.exists():
        print(f"\n✅ CHECKPOINT TROUVÉ: {checkpoint_path.name}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Informations détaillées
        checkpoint_type = checkpoint.get('checkpoint_type', 'unknown')
        timestamp = checkpoint.get('timestamp', 'inconnu')
        epoch = checkpoint.get('epoch', 0)
        batch_idx = checkpoint.get('batch_idx', 0)
        total_batches = checkpoint.get('total_batches_processed', 0)
        
        print(f"\n📊 Type: {checkpoint_type}")
        print(f"📅 Date: {timestamp}")
        print(f"📁 Taille: {format_size(checkpoint_path.stat().st_size)}")
        
        print(f"\n🔢 Progression:")
        print(f"   • Époque: {epoch}")
        print(f"   • Batch actuel: {batch_idx}")
        print(f"   • Total batches traités: {total_batches:,}")
        
        print(f"\n📈 Métriques:")
        print(f"   • Train Loss: {checkpoint.get('train_loss', 0):.4f}")
        print(f"   • Val Loss: {checkpoint.get('val_loss', 0):.4f}")
        print(f"   • Best Val Loss: {checkpoint.get('best_val_loss', float('inf')):.4f}")
        
        # Estimation de la progression
        if total_batches > 0:
            batch_size = 8  # Valeur par défaut
            estimated_examples = total_batches * batch_size
            next_save = 1000 - (batch_idx % 1000)
            
            print(f"\n📊 Estimation:")
            print(f"   • Exemples déjà vus: ~{estimated_examples:,}")
            print(f"   • Prochaine sauvegarde dans: {next_save} batches")
        
        # Vérifier si poids LoRA présents
        if 'lora_state_dict' in checkpoint:
            num_lora_params = sum(p.numel() for p in checkpoint['lora_state_dict'].values())
            print(f"\n🔧 LoRA:")
            print(f"   • Poids LoRA présents: {num_lora_params:,} paramètres")
            lora_cfg = checkpoint.get('lora_config', {})
            print(f"   • Rank: {lora_cfg.get('rank', 'N/A')}")
            print(f"   • Alpha: {lora_cfg.get('alpha', 'N/A')}")
        
        print(f"\n💡 Pour reprendre l'entraînement:")
        print(f"   python LoRAFineTuning.py")
        print(f"   → Reprise automatique à l'époque {epoch}, batch {batch_idx + 1}")
        
    else:
        print(f"\n❌ CHECKPOINT ABSENT: {checkpoint_path}")
        print(f"💡 L'entraînement démarrera depuis le début")
    
    # ========================================
    # 2. VÉRIFIER model.pt (modèle fusionné)
    # ========================================
    model_path = MODEL_DIR / "model.pt"
    
    print(f"\n{'='*70}")
    
    if model_path.exists():
        size = format_size(model_path.stat().st_size)
        print(f"✅ MODÈLE FUSIONNÉ: {model_path.name}")
        print(f"   Taille: {size}")
        
        # Essayer de charger pour vérifier l'intégrité
        try:
            model_state = torch.load(model_path, map_location='cpu', weights_only=True)
            num_params = sum(p.numel() for p in model_state.values())
            print(f"   Paramètres: {num_params:,}")
            print(f"   État: ✅ Valide")
        except Exception as e:
            print(f"   État: ⚠️  Erreur de chargement: {e}")
    else:
        print(f"❌ MODÈLE FUSIONNÉ ABSENT: {model_path}")
        print(f"💡 Sera créé à la fin de chaque époque")
    
    # ========================================
    # 3. VÉRIFIER best_model.pt
    # ========================================
    best_path = CHECKPOINT_DIR / "best_model.pt"
    
    print(f"\n{'='*70}")
    
    if best_path.exists():
        print(f"🏆 MEILLEUR MODÈLE: {best_path.name}")
        
        try:
            best = torch.load(best_path, map_location='cpu')
        except:
            best = torch.load(best_path, map_location='cpu', weights_only=False)
        
        print(f"   Val Loss: {best.get('val_loss', 0):.4f}")
        print(f"   Époque: {best.get('epoch', 0)}")
        print(f"   Date: {best.get('timestamp', 'inconnu')}")
        print(f"   Taille: {format_size(best_path.stat().st_size)}")
    else:
        print(f"❌ MEILLEUR MODÈLE ABSENT: {best_path}")
        print(f"💡 Sera créé quand val_loss s'améliore")
    
    # ========================================
    # 4. VÉRIFIER lora_weights.pt
    # ========================================
    lora_path = MODEL_DIR / "lora_weights.pt"
    
    print(f"\n{'='*70}")
    
    if lora_path.exists():
        print(f"✅ POIDS LoRA: {lora_path.name}")
        print(f"   Taille: {format_size(lora_path.stat().st_size)}")
        
        try:
            lora_state = torch.load(lora_path, map_location='cpu')
            if 'metadata' in lora_state:
                meta = lora_state['metadata']
                print(f"   Trainable: {meta.get('trainable_params', 0):,}")
                print(f"   Date: {meta.get('timestamp', 'inconnu')}")
        except:
            pass
    else:
        print(f"❌ POIDS LoRA ABSENTS: {lora_path}")
    
    # ========================================
    # 5. RÉSUMÉ FINAL
    # ========================================
    print(f"\n{'='*70}")
    print("📋 RÉSUMÉ")
    print("="*70)
    
    files_status = {
        "checkpoint.pt": checkpoint_path.exists(),
        "model.pt": model_path.exists(),
        "best_model.pt": best_path.exists(),
        "lora_weights.pt": lora_path.exists()
    }
    
    print(f"\n📁 Fichiers présents:")
    for filename, exists in files_status.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {filename}")
    
    all_present = all(files_status.values())
    
    if all_present:
        print(f"\n✅ SYSTÈME COMPLET - Prêt pour reprise ou déploiement")
    elif files_status["checkpoint.pt"]:
        print(f"\n⚠️  REPRISE POSSIBLE - checkpoint.pt présent")
        print(f"   Les fichiers manquants seront recréés automatiquement")
    else:
        print(f"\n❌ NOUVEL ENTRAÎNEMENT - Aucun checkpoint")
        print(f"   L'entraînement démarrera depuis zéro")
    
    # Vérifier l'espace disque disponible
    try:
        import shutil
        total, used, free = shutil.disk_usage(CHECKPOINT_DIR.parent)
        print(f"\n💾 Espace disque:")
        print(f"   Total: {format_size(total)}")
        print(f"   Utilisé: {format_size(used)}")
        print(f"   Libre: {format_size(free)}")
        
        if free < 1024**3:  # Moins de 1GB
            print(f"   ⚠️  ATTENTION: Espace faible!")
    except:
        pass
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    check_training_state()
