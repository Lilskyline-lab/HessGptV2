#!/usr/bin/env python3
"""
Script de gestion des checkpoints - VERSION NOMS FIXES
Usage:
    python manage_checkpoints.py --info         # Info sur checkpoint.pt
    python manage_checkpoints.py --best         # Info sur best_model.pt
    python manage_checkpoints.py --compare      # Compare checkpoint vs best
    python manage_checkpoints.py --validate     # Valide l'intégrité
"""
#manage_checkpoints.py
import argparse
import sys
import os
from pathlib import Path
import torch
import json

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Chemins fixes
CHECKPOINT_DIR = Path("./IA/saved_models/my_llm/checkpoints")
MODEL_DIR = Path("./IA/saved_models/my_llm")

def format_size(bytes_size):
    """Formate la taille"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def load_checkpoint_safe(path):
    """Charge un checkpoint en mode sécurisé"""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except:
        return torch.load(path, map_location='cpu')

def checkpoint_info():
    """Affiche les détails de checkpoint.pt"""
    checkpoint_path = CHECKPOINT_DIR / "checkpoint.pt"
    
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint introuvable: {checkpoint_path}")
        print("💡 Lancez l'entraînement pour créer un checkpoint")
        return
    
    data = load_checkpoint_safe(checkpoint_path)
    
    print("\n" + "="*70)
    print("📋 CHECKPOINT ACTUEL (checkpoint.pt)")
    print("="*70)
    
    print(f"\n📁 Fichier:")
    print(f"   Chemin: {checkpoint_path.absolute()}")
    print(f"   Taille: {format_size(checkpoint_path.stat().st_size)}")
    
    print(f"\n📅 Métadonnées:")
    print(f"   Timestamp: {data.get('timestamp', 'unknown')}")
    print(f"   Type: {data.get('checkpoint_type', 'unknown')}")
    
    print(f"\n🔢 Progression:")
    print(f"   Époque: {data.get('epoch', 0)}")
    print(f"   Batch: {data.get('batch_idx', 0)}")
    print(f"   Global Step: {data.get('global_step', 0)}")
    print(f"   Total batches: {data.get('total_batches_processed', 0):,}")
    
    print(f"\n📊 Métriques:")
    print(f"   Train Loss: {data.get('train_loss', 0):.4f}")
    print(f"   Val Loss: {data.get('val_loss', 0):.4f}")
    print(f"   Best Val Loss: {data.get('best_val_loss', float('inf')):.4f}")
    
    print(f"\n🔧 Configuration LoRA:")
    lora_cfg = data.get('lora_config', {})
    print(f"   Rank: {lora_cfg.get('rank', 'N/A')}")
    print(f"   Alpha: {lora_cfg.get('alpha', 'N/A')}")
    print(f"   Dropout: {lora_cfg.get('dropout', 'N/A')}")
    print(f"   Target modules: {lora_cfg.get('target_modules', [])}")
    
    print(f"\n📚 Configuration Modèle:")
    model_cfg = data.get('model_config', {})
    print(f"   Vocab Size: {model_cfg.get('vocab_size', 'N/A'):,}")
    print(f"   Embed Dim: {model_cfg.get('embed_dim', 'N/A')}")
    print(f"   Num Heads: {model_cfg.get('num_heads', 'N/A')}")
    print(f"   Num Layers: {model_cfg.get('num_layers', 'N/A')}")
    print(f"   Max Seq Len: {model_cfg.get('max_seq_len', 'N/A')}")
    
    print(f"\n💾 Historique:")
    history = data.get('history', {})
    print(f"   Total exemples entraînés: {history.get('total_qa_trained', 0):,}")
    print(f"   Cycles complétés: {len(history.get('cycles', []))}")
    
    # Vérifier intégrité LoRA
    if 'lora_state_dict' in data:
        num_params = sum(p.numel() for p in data['lora_state_dict'].values())
        print(f"\n✅ Poids LoRA présents: {num_params:,} paramètres")
    else:
        print(f"\n⚠️  Poids LoRA manquants!")
    
    # Vérifier optimizer
    if 'optimizer_state_dict' in data:
        print(f"✅ État optimiseur présent")
    else:
        print(f"⚠️  État optimiseur manquant!")
    
    print("="*70 + "\n")

def best_checkpoint_info():
    """Affiche les détails de best_model.pt"""
    best_path = CHECKPOINT_DIR / "best_model.pt"
    
    if not best_path.exists():
        print(f"\n❌ Meilleur modèle introuvable: {best_path}")
        print("💡 Sera créé quand val_loss s'améliore")
        return
    
    data = load_checkpoint_safe(best_path)
    
    print("\n" + "="*70)
    print("🏆 MEILLEUR MODÈLE (best_model.pt)")
    print("="*70)
    
    print(f"\n📁 Fichier:")
    print(f"   Taille: {format_size(best_path.stat().st_size)}")
    
    print(f"\n📅 Sauvegardé le: {data.get('timestamp', 'unknown')}")
    
    print(f"\n🔢 Époque: {data.get('epoch', 0)}")
    
    print(f"\n📊 Métriques:")
    print(f"   Train Loss: {data.get('train_loss', 0):.4f}")
    print(f"   Val Loss: {data.get('val_loss', 0):.4f} 🏆")
    print(f"   Best Val Loss: {data.get('best_val_loss', float('inf')):.4f}")
    
    print("="*70 + "\n")

def compare_checkpoints():
    """Compare checkpoint.pt et best_model.pt"""
    checkpoint_path = CHECKPOINT_DIR / "checkpoint.pt"
    best_path = CHECKPOINT_DIR / "best_model.pt"
    
    if not checkpoint_path.exists():
        print("❌ checkpoint.pt introuvable")
        return
    
    if not best_path.exists():
        print("❌ best_model.pt introuvable")
        return
    
    current = load_checkpoint_safe(checkpoint_path)
    best = load_checkpoint_safe(best_path)
    
    print("\n" + "="*70)
    print("⚖️  COMPARAISON CHECKPOINT vs BEST")
    print("="*70)
    
    print(f"\n📊 Val Loss:")
    current_val = current.get('val_loss', float('inf'))
    best_val = best.get('val_loss', float('inf'))
    
    print(f"   Checkpoint actuel: {current_val:.4f}")
    print(f"   Meilleur modèle:   {best_val:.4f} 🏆")
    
    if current_val < best_val:
        improvement = best_val - current_val
        print(f"\n✅ AMÉLIORATION: -{improvement:.4f} ({improvement/best_val*100:.2f}%)")
        print(f"💡 Le meilleur modèle sera mis à jour à la prochaine époque")
    elif current_val > best_val:
        degradation = current_val - best_val
        print(f"\n⚠️  DÉGRADATION: +{degradation:.4f} ({degradation/best_val*100:.2f}%)")
        print(f"💡 Continuez l'entraînement ou rechargez best_model.pt")
    else:
        print(f"\n➡️  IDENTIQUE")
    
    print(f"\n📈 Progression:")
    print(f"   Checkpoint: Époque {current.get('epoch', 0)}, Batch {current.get('batch_idx', 0)}")
    print(f"   Best:       Époque {best.get('epoch', 0)}")
    
    print(f"\n📅 Dates:")
    print(f"   Checkpoint: {current.get('timestamp', 'unknown')}")
    print(f"   Best:       {best.get('timestamp', 'unknown')}")
    
    print("="*70 + "\n")

def validate_integrity():
    """Valide l'intégrité de tous les fichiers"""
    print("\n" + "="*70)
    print("🔍 VALIDATION INTÉGRITÉ")
    print("="*70)
    
    files_to_check = {
        "checkpoint.pt": CHECKPOINT_DIR / "checkpoint.pt",
        "best_model.pt": CHECKPOINT_DIR / "best_model.pt",
        "model.pt": MODEL_DIR / "model.pt",
        "lora_weights.pt": MODEL_DIR / "lora_weights.pt"
    }
    
    results = {}
    
    for name, path in files_to_check.items():
        print(f"\n📄 Vérification: {name}")
        
        if not path.exists():
            print(f"   ❌ Fichier absent")
            results[name] = "missing"
            continue
        
        try:
            data = load_checkpoint_safe(path)
            
            # Vérifications spécifiques
            if name in ["checkpoint.pt", "best_model.pt"]:
                required_keys = ['epoch', 'lora_state_dict', 'optimizer_state_dict']
                missing = [k for k in required_keys if k not in data]
                
                if missing:
                    print(f"   ⚠️  Clés manquantes: {missing}")
                    results[name] = "incomplete"
                else:
                    print(f"   ✅ Structure valide")
                    print(f"   📊 Loss: {data.get('val_loss', 0):.4f}")
                    results[name] = "valid"
            
            elif name == "model.pt":
                num_params = sum(p.numel() for p in data.values())
                print(f"   ✅ Valide ({num_params:,} paramètres)")
                results[name] = "valid"
            
            elif name == "lora_weights.pt":
                if 'lora_layers' in data:
                    num_params = sum(p.numel() for p in data['lora_layers'].values())
                    print(f"   ✅ Valide ({num_params:,} paramètres LoRA)")
                    results[name] = "valid"
                else:
                    print(f"   ⚠️  Format invalide")
                    results[name] = "invalid"
        
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results[name] = "error"
    
    # Résumé
    print(f"\n{'='*70}")
    print("📋 RÉSUMÉ")
    print("="*70)
    
    valid_count = sum(1 for v in results.values() if v == "valid")
    total_count = len(results)
    
    for name, status in results.items():
        emoji = {
            "valid": "✅",
            "incomplete": "⚠️ ",
            "invalid": "❌",
            "missing": "❌",
            "error": "❌"
        }.get(status, "❓")
        
        print(f"   {emoji} {name}: {status}")
    
    print(f"\n📊 Score: {valid_count}/{total_count} fichiers valides")
    
    if valid_count == total_count:
        print(f"✅ SYSTÈME COMPLET ET FONCTIONNEL")
    elif results.get("checkpoint.pt") == "valid":
        print(f"⚠️  SYSTÈME PARTIEL - Reprise possible")
    else:
        print(f"❌ SYSTÈME INCOMPLET - Nouvel entraînement requis")
    
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Gestion des checkpoints avec noms fixes"
    )
    
    parser.add_argument("--info", action="store_true", 
                       help="Info sur checkpoint.pt")
    parser.add_argument("--best", action="store_true", 
                       help="Info sur best_model.pt")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare checkpoint vs best")
    parser.add_argument("--validate", action="store_true", 
                       help="Valide l'intégrité de tous les fichiers")
    
    args = parser.parse_args()
    
    if args.info:
        checkpoint_info()
    elif args.best:
        best_checkpoint_info()
    elif args.compare:
        compare_checkpoints()
    elif args.validate:
        validate_integrity()
    else:
        # Par défaut, afficher checkpoint info
        print("💡 Usage: python manage_checkpoints.py [--info|--best|--compare|--validate]")
        print("\nAffichage par défaut:\n")
        checkpoint_info()

if __name__ == "__main__":
    main()
