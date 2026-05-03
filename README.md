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
git clone https://github.com/ton-username/OMNI-RECO-v2.git
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
*(Insère ici la FAQ que je t'ai donnée précédemment)*
**⚠️ Avertissement :** *Cet outil est fourni à des fins de recherche OSINT. L'auteur décline toute responsabilité quant à l'utilisation détournée de cette technologie.*
