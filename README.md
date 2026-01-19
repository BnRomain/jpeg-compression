# 📷 Image Compression via DCT & Sparse Matrices (CSR)

Ce projet propose une implémentation personnalisée de la compression d'image inspirée de la norme **JPEG**, utilisant la **Transformée en Cosinus Discrète (DCT)** et une optimisation du stockage via le format **CSR (Compressed Sparse Row)**.

L'application est interactive et développée avec **Streamlit**.

## 🚀 Aperçu du projet
L'objectif est de démontrer comment la mise à zéro de fréquences spécifiques dans le domaine fréquentiel (DCT) permet de créer une matrice "creuse" (sparse), que l'on peut ensuite stocker de manière beaucoup plus compacte qu'une image brute.



## 🛠️ Fonctionnement Technique

L'algorithme suit les étapes rigoureuses du traitement d'image :
1. **Découpage en blocs** : L'image est traitée par blocs de 8 x 8 pixels sur les trois canaux **RGB**.
2. **DCT-2** : Passage de l'espace spatial à l'espace fréquentiel via une matrice de passage P.
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
👉 [Démo en ligne](https://jpeg-csr-compression.streamlit.app/)
