# 📷 Image Compression via DCT & Sparse Matrices (CSR)

Ce projet propose une implémentation personnalisée de la compression d'image inspirée de la norme **JPEG**, utilisant la **Transformée en Cosinus Discrète (DCT)** et une optimisation du stockage via le format **CSR (Compressed Sparse Row)**.

L'application est interactive et développée avec **Streamlit**.

Le projet existe désormais en deux versions :

| Version | Dossier | Contexte | Auteurs |
|---|---|---|---|
| **Python** (application Streamlit) | [`python/`](python) | Projet MAM3, janvier 2026 | Romain Ben, Evrard Lecureur, Zouhair Saitout |
| **C++20** (ligne de commande) | [`cpp/`](cpp) | Projet C++ MAM4, septembre 2026 | Romain Ben, Karim Zrig |

## 🚀 Aperçu du projet
L'objectif est de démontrer comment la mise à zéro de fréquences spécifiques dans le domaine fréquentiel (DCT) permet de créer une matrice "creuse" (sparse), que l'on peut ensuite stocker de manière beaucoup plus compacte qu'une image brute.



## 🛠️ Fonctionnement Technique

L'algorithme suit les étapes rigoureuses du traitement d'image :
1. **Découpage en blocs** : L'image est traitée par blocs de $8 \times 8$ pixels sur les trois canaux **RGB**.
2. **DCT-2** : Passage de l'espace spatial à l'espace fréquentiel via une matrice de passage $P$.
3. **Quantification & Seuillage** : 
   - Division par une matrice de quantification standard $Q$.
   - Application d'un seuil réglable : les coefficients inférieurs au seuil sont mis à zéro.
   - Suppression des hautes fréquences (tronquage de la matrice $D$).
4. **Stockage Sparse** : Conversion des matrices denses en format **CSR** (Compressed Sparse Row) pour ne conserver que les valeurs non nulles.
5. **Reconstruction** : Application de la DCT inverse ($P^T D P$) pour visualiser l'image reconstruite.



## 📊 Analyse de la Compression
L'application affiche en temps réel des métriques pour comparer l'efficacité de l'algorithme :
* **Données RAM** : Le poids de l'image "dépliée" en mémoire vive (pixel par pixel).
* **Taille CSR** : La taille réelle occupée par les matrices compressées (données utiles + indices).
* **Ratio de Gain** : Le facteur de réduction entre le volume brut et le stockage optimisé.

> **💡 Note technique :** La différence entre le fichier original (ex: PNG de 200 Ko) et la "Taille RAM" (ex: 50 Mo) est normale. L'original est déjà compressé par des codecs systèmes. Mon algorithme travaille sur les données brutes pour démontrer le gain mathématique du format CSR.

## 🔗 Démo en ligne
👉 [Compresser une image](https://jpeg-csr-compression.streamlit.app/)

Pour lancer l'application en local :

```bash
cd python
pip install -r requirements.txt
streamlit run app.py
```

## ⚙️ Version C++

La version C++ reprend tout le fonctionnement de la version Python, sans numpy ni scipy : DCT, quantification avec un facteur de qualité $\alpha$, seuil, suppression des hautes fréquences, matrices CSR écrites à la main et fichier binaire `.csr` (équivalent du `.npz`). Elle ajoute en options les analyses du rapport : matrices de quantification alternatives, troncature triangulaire du sujet, bruit poivre et sel.

```bash
cd cpp
make test                                            # tests unitaires
make                                                 # construit ./jpeg_csr
./jpeg_csr compress images/astronaut.png --alpha 5   # compression + métriques
./jpeg_csr decompress resultats/astronaut.csr relue.png
```

Sous Windows avec MSYS2, utiliser `mingw32-make` à la place de `make`.

Sur une image $512 \times 512$, les deux versions produisent les mêmes coefficients (25 écarts d'arrondi flottant sur 786 432) et la version C++ est environ **11 fois plus rapide**. Architecture, options et résultats : [`cpp/README.md`](cpp/README.md).

## 🗂️ Structure du dépôt

```text
jpeg-compression/
├── python/                   version Python (MAM3)
│   ├── app.py                application Streamlit
│   ├── jpeg_compression.py   compression et décompression
│   ├── requirements.txt      dépendances (versions figées)
│   ├── requirements-dev.txt  dépendances de test
│   ├── tests/                tests pytest
│   └── docs/                 rapport et présentation
├── cpp/                      version C++ (MAM4)
│   ├── include/  src/        code source
│   ├── tests/                tests unitaires
│   ├── third_party/          stb_image et stb_image_write
│   ├── scripts/              comparaison avec Python, figures
│   └── docs/                 rapport de synthèse et présentation
├── .github/                  workflows GitHub Actions et Dependabot
└── SECURITY.md               politique de sécurité
```

## ✅ Tests et qualité

À chaque Pull Request et à chaque push sur `main`, GitHub Actions lance :
* **Python tests** : `pytest` sur les fonctions de compression et sur le démarrage de l'application ;
* **C++ tests** : `make test` sous AddressSanitizer et UBSan, puis `make demo` ;
* **Dependency review** : bloque une Pull Request qui ajoute une dépendance vulnérable ;
* **CodeQL** : analyse de sécurité du code Python, C++ et des workflows.

**Dependabot** surveille les dépendances Python et les GitHub Actions. Les mises à jour patch et minor sont fusionnées automatiquement dès que les vérifications obligatoires de `main` sont passées. Voir aussi la [politique de sécurité](SECURITY.md) et le [Wiki](https://github.com/BnRomain/jpeg-compression/wiki).

## 📄 Rapport & Présentation

Si vous êtes intéressé par les détails théoriques et l'analyse complète de ce projet, vous pouvez consulter :

- **📑 Rapport complet** : [Voir le rapport](python/docs/Rapport.pdf)  
- **📊 Présentation Slides** : [Voir la présentation](python/docs/Presentation.pdf)  

Ces documents détaillent :
- L'algorithme DCT & CSR utilisé  
- Les résultats et métriques de compression  
- Les illustrations et comparaisons visuelles  

Pour la version C++ :

- **📑 Rapport de synthèse (2 pages)** : [Voir le rapport](cpp/docs/rapport.pdf)  
- **📊 Présentation** : [Voir la présentation](cpp/docs/presentation.pdf)  
