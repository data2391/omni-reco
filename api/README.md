# OMNI-RECO API v1.0

## Installation

```bat
pip install fastapi uvicorn python-multipart
```

## Démarrage

```bat
cd api
python generate_token.py     # génère ton token Bearer
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

| Méthode | URL | Auth | Description |
|---|---|---|---|
| GET | `/health` | Non | Statut moteur |
| POST | `/compare` | Bearer token | Compare 2 photos |

## Exemple cURL

```bash
curl -X POST http://localhost:8000/compare \
  -H "Authorization: Bearer TON_TOKEN" \
  -F "photo_a=@photo1.jpg" \
  -F "photo_b=@photo2.jpg"
```

## Exemple réponse

```json
{
  "match": true,
  "score": 0.8631,
  "confidence": 86.3,
  "method": "arcface+geometry+texture",
  "detail": {
    "arcface_sim": 0.8453,
    "geo_score": 0.71,
    "texture_score": 0.9818,
    "severity": "high",
    "weights": {"arcface": 0.65, "geometry": 0.22, "texture": 0.13}
  },
  "elapsed_s": 14.2
}
```

## Sécurité

- Les tokens sont stockés **hashés SHA-256** dans `tokens.txt`
- Le token en clair n'est jamais stocké sur disque
- Utilise HTTPS en production (nginx reverse proxy recommandé)
