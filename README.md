# 🕵️‍♂️ OMNI-RECO v1
**Moteur de Triple Fusion Biométrique : ArcFace | Géométrie 3D | Texture Gabor**
## 🚀 Présentation
**OMNI-RECO** n'est pas un énième script de reconnaissance faciale basé sur une simple comparaison de distance. C'est un moteur de fusion conçu pour les environnements dégradés, les visages partiellement masqués et les angles extrêmes.
Alors que les solutions standards s'effondrent dès qu'un pixel manque, **OMNI-RECO** maintient son intégrité en croisant trois couches de données indépendantes.
## 🧠 La Triple Fusion (A/G/T)
Le système calcule un **Score Fusionné** basé sur trois piliers :
 1. **ArcFace Embedding (512D)** : Identification de l'identité globale via Deep Learning.
 2. **Géométrie 3D (MediaPipe Legacy)** : Analyse structurelle de 468 landmarks. Extraction de **18 ratios osseux** (distance inter-oculaire, courbure des sourcils, inclinaison nasale, etc.).
   * *Bypasse les occlusions (main devant le visage, masque chirurgical).*
 3. **Texture Gabor** : Analyse du grain de peau via 40 filtres fréquentiels. Détecte si la texture dermique correspond, même si la géométrie est altérée.
## 🛠️ Installation (Zero-Config)
Le système est verrouillé pour garantir la stabilité. L'installateur déploie automatiquement un environnement isolé **Python 3.10** pour éviter les conflits d'API avec les versions récentes de MediaPipe (≥ 0.10.14).
```bash
git clone https://github.com/data2391/omni-reco.git
cd OMNI-RECO-v2
setup.bat

```
## 💻 Utilisation
```bash
python compare.py cible.jpg suspect.png --verbose

```
### Exemple de verdict :
> **[GEOMETRY]** 18 ratios calculés (Score: 0.9311)
> **[TEXTURE]** Analyse Gabor terminée (Score: 0.9989)
> **[RESULT]** ✅ **MATCH FOUND (85.1%)** > *Détail : Identité confirmée via structure osseuse malgré occlusion nasale.*
> 
## 🛡️ Pourquoi cette version ?
Cette version FINAL_9FIXES résout les problèmes critiques rencontrés par la communauté :
 * **Fix API MediaPipe** : Rétablissement de solutions.face_mesh via versioning 0.10.9.
 * **Yaw Correction** : Gestion dynamique des poids (A/G/T) pour les profils extrêmes (jusqu'à -175°).
 * **Sharpening Pipeline** : Prétraitement CLAHE et Denoising NlMeans intégré pour les photos de basse qualité.
## 📖 Dans la même série
Ce projet fait partie de l'écosystème **data2391** et accompagne l'ouvrage :
*"Insubmersible: Mode d'emploi contre un système liberticide"*

## 📜 FAQ

# Pourquoi OMNI-RECO est différent des autres outils de Face-Match ?

La plupart des outils (Dlib, InsightFace seul) échouent dès qu'un visage est partiellement masqué ou de profil. OMNI-RECO utilise une Triple Fusion Biométrique + Oreilles :
	Deep Learning (ArcFace 512D) : identité globale via embedding cosinus
	Géométrie 3D (MediaPipe custom) : analyse de la structure osseuse via 18 ratios mathématiques — fonctionne même si le bas du visage est caché
	Analyse de Texture (Gabor) : comparaison du grain de peau au niveau poreux
	Biométrie auriculaire (YOLOv8-pose) : signature HOG + Gabor de l'oreille, quasi-invariante à l'expression faciale
Si une composante est inutilisable, la fusion redistribue automatiquement son poids sur les zones exploitables.

# Pourquoi l'installateur force-t-il Python 3.10 ?

Parce que MediaPipe ≥ 0.10.14 a supprimé les API solutions.face_mesh utilisées pour le calcul des landmarks 3D.
Pour garantir un score de précision chirurgicale, l'installateur déploie un environnement scellé en Python 3.10.x avec les dépendances exactes figées dans requirements_v2.txt.
Si vous êtes sur Linux et que votre distribution n'inclut pas Python 3.10, setup-linux.sh le compile depuis les sources automatiquement.

# Ça marche vraiment si la personne porte un masque ou a une main devant ?

Oui. Grâce à la pondération adaptative :
	Si le bas du visage est masqué → la géométrie se concentre sur les ratios yeux/sourcils/front
	Si le visage est de profil (|yaw| > 90°) → la géométrie est exclue, les oreilles prennent le relais
	Si la qualité est médiocre → ArcFace réduit son poids, la texture compensie
Si les ratios géométriques des yeux matchent à 95%, le système valide l'identité même sans voir la bouche ou le nez.

# Pourquoi mon score de confiance varie selon les photos ?

Le score est honnête, pas artificellement lissé.
Score	Interprétation
0.90 – 1.00	Conditions studio parfaites, match certain
0.80 – 0.89	Match quasi certain, angles complexes
0.65 – 0.79	Piste sérieuse — qualité médiocre ou yaw extrême
0.50 – 0.64	Incertain — vérification manuelle recommandée
< 0.50	Structures différentes, pas la même personne

Un score de 0.87 sur une photo de profil en basse résolution est plus fiable qu'un score de 0.92 obtenu par un outil qui n'analyse que l'embedding ArcFace.

# Puis-je l'intégrer dans une application web ou mobile ?

Oui. L'API REST est prête à l'emploi :
# Démarrage
api/launch-api.bat       # Windows
bash api/launch-api.sh   # Linux

# Appel depuis n'importe quelle app
curl -X POST http://localhost:8000/compare \
  -H "Authorization: Bearer TON_TOKEN" \
  -F "photo_a=@image1.jpg" \
  -F "photo_b=@image2.jpg"

L'authentification est par Bearer token SHA-256. Pour générer un token : python api/generate_token.py.
Documentation Swagger disponible sur http://localhost:8000/docs.

# Est-ce que ça tourne sur GPU ?
La version actuelle utilise CPUExecutionProvider (ONNX Runtime). Pour activer le GPU :
pip uninstall onnxruntime
pip install onnxruntime-gpu

InsightFace détectera automatiquement CUDA si disponible. Sur CPU, compter ~15-20s par comparaison. Sur GPU (RTX 3060+), ~2-3s.

# Pourquoi médiapipe==0.10.9 exactement ?

La version 0.10.9 est la dernière qui expose l'API mp.solutions.face_mesh.FaceMesh avec refine_landmarks=True (nécessaire pour les 478 landmarks iris inclus).
Les versions ≥ 0.10.14 ont migré vers une API Tasks complètement différente et incompatible avec le pipeline actuel.

Le FutureWarning InsightFace rcond est-il dangereux ?

Non. C'est un avertissement bénin de NumPy concernant un changement futur dans np.linalg.lstsq. Il est supprimé proprement dans OMNI-RECO v2.1 (FIX-9). Il n'affecte pas les calculs.

# Puis-je contribuer ?

Les Pull Requests sont ouvertes. Conditions :
1.	Comprendre la différence entre une distance cosinus et une projection 3D
2.	Lancer python core/regression_tests.py — tous les tests doivent passer
3.	Ne pas modifier requirements_v2.txt sans justification documentée

Est-ce légal ?
OMNI-RECO est un outil de recherche OSINT et de biométrie, conçu pour les audits de réputation numérique et la vérification d'identité.
Son utilisation est soumise aux lois applicables dans votre juridiction — notamment le RGPD en Europe et les législations locales sur la biométrie.
L'utilisateur est seul responsable de l'usage qu'il fait de cet outil. Pirate par nécessité, libre par principe.

**⚠️ Avertissement :** *Cet outil est fourni à des fins de recherche OSINT. L'auteur décline toute responsabilité quant à l'utilisation détournée de cette technologie.*
