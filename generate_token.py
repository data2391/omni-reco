"""
api/generate_token.py — Générateur de token OMNI-RECO API

Usage :
  cd api
  python generate_token.py

Génère un token Bearer aléatoire sécurisé,
l'ajoute à tokens.txt (stocké en hash SHA-256),
et l'affiche en clair pour l'utiliser dans ton app.
"""

import secrets, hashlib
from pathlib import Path

TOKENS_FILE = Path(__file__).parent / "tokens.txt"

def generate_token() -> str:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKENS_FILE, "a") as f:
        f.write(token_hash + "\n")

    print("=" * 60)
    print("  OMNI-RECO API — Nouveau token généré")
    print("=" * 60)
    print(f"  Token (à copier) : {token}")
    print(f"  Hash stocké      : {token_hash[:16]}...")
    print(f"  Fichier          : {TOKENS_FILE}")
    print("=" * 60)
    print("  Usage dans ton app :")
    print(f'  Authorization: Bearer {token}')
    print("=" * 60)
    return token

if __name__ == "__main__":
    generate_token()
