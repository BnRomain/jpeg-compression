# Script oral : projet C++ compression JPEG

Présentation du 23/09/2026, 10 minutes maximum. Slides : [`slides-fr.pdf`](slides-fr.pdf). Le nom de l'orateur est affiché en bas de chaque slide.

| Partie | Slides | Durée visée | Cumul |
|---|---|---|---|
| Romain | 1 à 7 : organisation du code, représentation des données | 4 min 30 | 4 min 30 |
| Karim | 8 à 13 : matrice creuse, ressources, polymorphisme, résultats, bilan | 4 min 30 | 9 min |
| Annexes | 14 à 16 : matrices Q, bruit, utilisation | pour les questions seulement | |

Environ 1 minute de marge. Si le temps presse : raccourcir la slide 3 (algorithme) et la slide 9 (fichier `.csr`), jamais les slides de code.

---

## Partie 1 : Romain

### Slide 1 : Titre (15 s, cumul 0:15)

Bonjour. Nous sommes Romain Ben et Karim Zrig, et nous allons vous présenter notre projet C++ : une compression d'image inspirée de JPEG, avec un stockage des coefficients en matrices creuses.

### Slide 2 : Du projet Python au C++ (40 s, cumul 0:55)

En MAM3, nous avions codé cet algorithme en Python, avec numpy, scipy et une application Streamlit. Cette semaine, nous l'avons entièrement réécrit en C++20, sans aucune bibliothèque de calcul. Le but n'était pas de changer l'algorithme, mais d'apprendre le C++ sur un problème que nous connaissions déjà. La présentation porte donc surtout sur le code : je présente l'organisation du programme et la façon dont on a représenté les données, puis Karim parlera de la matrice creuse, de la gestion des ressources, du polymorphisme et des résultats.

### Slide 3 : L'algorithme en une slide (40 s, cumul 1:35)

Rapidement, l'algorithme. L'image est découpée en blocs de 8 pixels sur 8, sur chacun des trois canaux. Sur chaque bloc centré, on applique la DCT, qui est un changement de base : D égale P M P transposée. On divise ensuite par la matrice de quantification Q, multipliée par un facteur de qualité alpha, et on tronque. Les petits coefficients et les hautes fréquences sont mis à zéro par un seuil et un masque. Il reste une matrice presque vide, stockée au format CSR. La décompression fait le chemin inverse.

### Slide 4 : Organisation du code (45 s, cumul 2:20)

Côté code, on a suivi la compilation séparée du cours : un en-tête commenté, `jpeg.hpp`, qui déclare tous les types, `jpeg.cpp` qui en donne les définitions dans le même ordre, et `main.cpp` pour le programme. Les douze sections portent les mêmes numéros dans l'en-tête et dans les définitions. Le schéma suit le trajet des données. Une image est lue, puis compressée par la fonction `compress`, qui utilise la DCT, la table de quantification et un masque. Le résultat est un objet `CompressedImage`, qu'on peut écrire dans un fichier `.csr`, relire et décompresser. La bibliothèque stb, qui lit les PNG et les JPEG, n'apparaît que dans un seul fichier.

### Slide 5 : Représenter l'image (50 s, cumul 3:10)

Premier choix de représentation : l'image. En Python, c'était un tableau numpy à trois dimensions. On aurait pu faire un vecteur de vecteurs, mais cela fait une allocation par ligne, et rien n'empêche deux lignes d'avoir des tailles différentes. On range donc toutes les intensités dans un seul `std::vector` de double, ligne par ligne, avec la formule d'indice en bas de la slide. La mémoire est contiguë, et il ne reste qu'un invariant à garantir : la taille du vecteur vaut largeur fois hauteur fois trois. L'accès passe par `operator()`, surchargé deux fois : une version pour écrire, et une version `const` pour lire une image constante. Comme `std::vector`, on a aussi `at`, qui vérifie les bornes.

### Slide 6 : Le bloc 8 par 8 (45 s, cumul 3:55)

Deuxième objet : le bloc 8 par 8. Un bloc de pixels, la matrice P et les coefficients sont tous des matrices 8 par 8, donc un seul type, `Matrix8`. Cette fois on utilise un `std::array` de 64 doubles : sa taille est connue à la compilation, donc aucune allocation, ce qui compte vu qu'une image 512 par 512 contient plus de 12 000 blocs. La classe `Dct` calcule P et sa transposée une seule fois, dans son constructeur. On y a appris un piège : les membres sont initialisés dans l'ordre où ils sont déclarés, donc P doit être déclarée avant sa transposée.

### Slide 7 : Quantification (35 s, cumul 4:30)

Dernier point pour moi : la matrice de quantification. Sa classe impose un invariant, tous les diviseurs valent au moins 1, vérifié dans le constructeur qui lance une exception sinon. Cet invariant fixe le type des données : un coefficient de DCT ne dépasse pas 1024 en valeur absolue, donc après division il tient sur 16 bits. C'est pour cela que la matrice creuse stocke des `int16`. Je laisse la parole à Karim.

---

## Partie 2 : Karim

### Slide 8 : La matrice creuse CSR (55 s, cumul 5:25)

Merci Romain. Après la quantification, la matrice d'un canal contient environ 95 % de zéros. On la stocke dans notre propre classe `SparseMatrix`, qui remplace `csr_matrix` de scipy. Elle contient trois vecteurs : les valeurs non nulles, la colonne de chacune, et les pointeurs de lignes, qui indiquent où commence chaque ligne. Sur l'exemple, la ligne du milieu est vide, donc deux pointeurs consécutifs sont égaux. La classe a deux constructeurs. Le premier part de la matrice dense, qu'il lit seulement, par référence constante. Le second reçoit les trois tableaux relus dans un fichier : il les prend par valeur puis les déplace avec `std::move`, donc aucun élément n'est recopié, et il vérifie l'invariant du format.

### Slide 9 : Image compressée et fichier .csr (35 s, cumul 6:00)

L'image compressée est une composition : les dimensions, la table de quantification et trois matrices creuses, une par canal. On l'écrit dans un fichier binaire `.csr`, dont le contenu est dans le tableau. Pour cela, les fonctions `write` et `read` sont surchargées : le compilateur choisit la bonne selon le type. À la relecture, on vérifie les dimensions avant d'allouer quoi que ce soit, puis chaque constructeur revérifie son invariant : un fichier corrompu est rejeté au lieu de donner une image fausse.

### Slide 10 : Ressources, RAII et règle de zéro (40 s, cumul 6:40)

Côté ressources, toutes nos classes contiennent des `std::vector` ou des `std::array`, qui gèrent leur mémoire. On n'a donc écrit aucun destructeur ni aucune copie : c'est la règle de zéro, et il n'y a aucun `new` ni `delete` dans le programme. La seule exception, c'est le tableau de pixels alloué par la bibliothèque stb. On l'a confié à une petite classe RAII, `StbPixels` : son destructeur libère le tableau, même si une exception est lancée, et la copie est interdite, sinon deux objets libéreraient le même tableau.

### Slide 11 : Polymorphisme (45 s, cumul 7:25)

Pour la troncature, on a deux formes : carrée comme en Python, triangulaire comme dans le sujet. Plutôt qu'un test dans la boucle de compression, on a repris l'idée du TD 6 : une interface abstraite `FrequencyMask`, avec une méthode virtuelle pure `keeps`, et deux classes qui la redéfinissent. La fonction `compress` reçoit une référence constante vers l'interface et ne sait pas quel masque elle utilise. Dans le `main`, le masque est un objet local passé par référence : pas de `new`, pas de copie, et ajouter une nouvelle forme ne modifie pas `compress`.

### Slide 12 : Validation et résultats (45 s, cumul 8:10)

Pour valider, on a 17 fonctions de test, qui reprennent les tests Python et vérifient les invariants de chaque classe ; l'intégration continue les relance sous AddressSanitizer et UBSan. En comparant avec la version Python sur la même image, seuls 25 coefficients sur 786 000 diffèrent, d'une unité, à cause d'un arrondi flottant côté Python. Et la version C++ est environ 11 fois plus rapide. Sur les images, on voit l'effet du facteur alpha : plus il est grand, moins on garde de coefficients, et plus l'image devient une mosaïque de blocs.

### Slide 13 : Bilan (35 s, cumul 8:45)

Pour conclure, ce projet nous a appris trois choses. En C++, le type porte l'information : les hypothèses implicites du Python deviennent des invariants. Chaque copie se voit, donc on réfléchit à ce que coûte chaque ligne. Et la durée de vie des objets structure le code. Pour aller plus loin, on pourrait ajouter le codage entropique pour produire un vrai JPEG, ou des pointeurs intelligents pour gérer plusieurs masques. Merci de votre attention, nous sommes prêts pour vos questions.

---

## Questions probables

- **Pourquoi pas `std::vector<std::vector<double>>` pour l'image ?** Une allocation par ligne, une mémoire éclatée, et aucune garantie que les lignes ont la même taille. Un seul vecteur donne un seul invariant.
- **Pourquoi `std::array` et pas `std::vector` pour `Matrix8` ?** Taille fixe connue à la compilation : pas d'allocation dynamique, alors qu'on crée plusieurs matrices par bloc.
- **Pourquoi pas de `std::unique_ptr` pour les masques ?** Pas vu en cours au moment du projet, et inutile ici : un seul masque vit pendant toute la compression, un objet local suffit.
- **Que se passe-t-il si le constructeur de `StbPixels` lance une exception ?** Le destructeur n'est pas appelé, mais rien n'a été alloué puisque `stbi_load` a échoué.
- **Pourquoi 25 coefficients différents ?** En Python l'image passe par /255 puis ×255 : D/Q vaut 23,999999999999996 au lieu de 24, et la troncature donne 23.
- **D'où vient le facteur 11 ?** Boucles compilées en `-O2`, aucune allocation par bloc, alors que Python appelle numpy bloc par bloc.
