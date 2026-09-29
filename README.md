# car-vision

Pipeline de perception autonome panoptique temps réel avec projection 3D Bird's-Eye View (BEV style Tesla FSD / Waymo), exécuté à 100% en local sur Apple Silicon (GPU Metal / MPS).

---

## 1. Architecture & Principes

L'application traite un flux vidéo pare-brise (dashcam, caméra embarquée ou webcam) et produit en temps réel deux représentations synchronisées :
1. **HUD Pare-brise AR** : Détection d'objets (véhicules, piétons, cyclistes), segmentation de la chaussée carrossable (*drivable area*) et des lignes de marquage au sol, projection de boîtes 3D filaires et alertes anticollision.
2. **BEV Vectoriel 3D (Tesla FSD)** : Reconstitution métrique de l'environnement autour du véhicule hôte (*Ego car* à l'origine $(0, 0)$), anneaux radar de distance (10m, 25m, 50m, 75m), vecteurs vitesse et marquages au sol projetés.

```
                    [ Flux Vidéo Dashcam ]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       [ YOLOPv2 (MPS) ]             [ YOLO11 (MPS) ]
     (Drivable Carpet +             (Acteurs : Voitures,
       Lignes de voies)               Piétons, Cyclistes)
               │                             │
               └──────────────┬──────────────┘
                              ▼
           [ Inverse Perspective Mapping (IPM) ]
          (Unprojection 2D -> 3D métrique sol X, Z)
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
       [ Boîtes 3D ]   [ Passages Piétons ] [ Tracker & TTC ]
               │              │              │
               └──────────────┬──────────────┘
                              ▼
        [ Cockpit Dual-Screen PySide6 (HUD + BEV) ]
```

### Modèles & Tâches
- **YOLOPv2** : Segmentation multitâche de la chaussée (masque 2D de zone roulable) et des lignes de voies (pleines, discontinues).
- **YOLO11n** : Détection rapide (6-8 ms sur M4) et classification des acteurs routiers (`car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person`).
- **Détecteur de passages piétons** : Détection des bandes blanches zébrées sur la chaussée par morphologie directionnelle et regroupement spatial.
- **Tracker spatial métrique** : Association spatiale euclidienne frame-à-frame, lissage de vitesse relative $(\dot{X}, \dot{Z})$ et calcul du Time-To-Collision (TTC).

### Formules géométriques (IPM Monoculaire)
Pour une caméra à une hauteur $h_c$ au-dessus de l'asphalte avec une inclinaison $\theta$ (*pitch*) et focale $f_y$ :
$$Z = \frac{f_y \cdot h_c}{(v - c_y)\cos\theta + f_y\sin\theta}$$
$$X = \frac{(u - c_x) \cdot Z}{f_x}$$
$$TTC = \frac{Z}{\max(-\dot{Z}, 0.1 \text{ m/s})}$$

---

## 2. Installation & Dépendances

Testé sur macOS (Apple Silicon M4).

```bash
# Cloner le dépôt
git clone https://github.com/eliottrouvier/car-vision.git
cd car-vision

# Créer l'environnement virtuel Python 3.12
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install torch torchvision ultralytics opencv-python numpy PySide6 pyopengl scipy requests
```

### Poids des modèles
- `weights/yolopv2.pt` : Poids TorchScript YOLOPv2 (149 Mo)
- `yolo11n.pt` : Poids Ultralytics YOLO11 nano (5.6 Mo)

---

## 3. Lancement

### Commande globale
Le CLI `carvision` est installé dans `~/.local/bin/carvision` :
```bash
carvision
```

### Lancement direct
```bash
./.venv/bin/python main.py
```

### Tests unitaires & benchmarks
```bash
./.venv/bin/python test_carvision.py
```

---

## 4. Fonctionnalités de l'interface

- **Sélecteur de source** : Vidéos de démonstration incluses (`highway.mp4`, `car-detection.mp4`, `person-bicycle-car-detection.mp4`), ouverture de fichier local ou webcam en direct.
- **Contrôles HUD** : Activation/désactivation à la volée du tapis de route, des lignes de voies, des boîtes 3D filaires et des passages piétons.
- **Navigation BEV** :
  - `Vue 3D FSD` : Perspective isométrique 3e personne derrière le véhicule hôte.
  - `Top-Down` : Carte radar orthogonale 90°.
  - Molette souris : Zoom métrique (5m à 120m).
  - Clic-glisser : Déplacement de la vue.
  - Double-clic : Recentrage vue par défaut.
- **Télémétrie en direct** : FPS, latence GPU (MPS), véhicule de tête (distance + TTC), alerte anticollision FCW (*Forward Collision Warning*), et alertes piétons.
