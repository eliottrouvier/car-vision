# car-vision

> **Tech Stack : Python • PyTorch (MPS) • YOLOPv2 • YOLO11 • OpenCV • PySide6**  
> Real-time autonomous panoptic perception pipeline and 3D Bird's-Eye View (Tesla FSD style) cockpit, running 100% locally on Apple Silicon (Metal/MPS).

---

## 1. Architecture & Principes

L'application traite un flux vidéo pare-brise (dashcam, caméra embarquée, fichier glissé-déposé ou webcam) et produit en temps réel deux représentations synchronisées :
1. **HUD Pare-brise AR** : Détection d'objets (véhicules, piétons, cyclistes) avec viseurs 2D tactiques ou cuboids 3D, segmentation panoptique de la chaussée roulable (*drivable carpet*) et des lignes de marquage, couloir de trajectoire projeté et alertes anticollision.
2. **Espace Vectoriel 3D (Tesla FSD)** : Reconstitution métrique 3D autour du véhicule hôte (*Ego car* à l'origine $(0, 0)$), modèles 3D pleins et ombrés avec phares/feux LED, badges HUD flottants (*Distance • Vitesse relative*), couloir de trajectoire dynamique et anneaux radar de distance (10m, 25m, 50m, 75m).

```
                    [ Flux Vidéo Dashcam / Fichier Drag & Drop ]
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
                [ YOLOPv2 (MPS) ]             [ YOLO11 (MPS) ]
              (Drivable Carpet +            (Acteurs : Voitures,
                Lignes de voies)              Piétons, Cyclistes)
                        │                             │
                        └──────────────┬──────────────┘
                                       ▼
                   [ Estimateur de Profondeur Hybride Bayésien ]
                  (Fusion Contact Sol IPM + Prior Hauteur Pinhole)
                                       │
                                       ▼
                       [ Filtre de Kalman 3D + 1€ Filter ]
                     (État [X, Z, vx, vz], Lissage & Coasting)
                                       │
                        ┌──────────────┼──────────────┐
                        ▼              ▼              ▼
                [ Modèles 3D ]  [ Couloir Trajectoire ] [ Badges HUD & TTC ]
                        │              │              │
                        └──────────────┬──────────────┘
                                       ▼
                 [ Cockpit Dual-Screen PySide6 Synchronisé ]
```

### Modèles & Tâches
- **YOLOPv2** : Segmentation multitâche de la chaussée (polygone 3D de surface carrossable) et des lignes de voies (pleines, discontinues).
- **YOLO11s (Défaut)** : Détection haute précision (45-60 FPS sur GPU Apple M4) des acteurs routiers (`car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person`). Bascule à chaud possible vers **YOLO11m** (précision max) ou **YOLO11n** (ultra-rapide).
- **Estimateur de Profondeur Hybride** : Combine géométriquement le contact au sol (IPM) et la hauteur apparente de boîte ($Z \approx f_y \cdot H / h$). Résout définitivement la perte des véhicules lointains près de l'horizon.
- **Tracker Kinématique 3D & 1€ Filter** : Filtre de Kalman 3D avec maintien des pistes en cas d'occlusion (coasting jusqu'à 10 frames), et lissage adaptatif 1€ Filter sans à-coups ni latence.
- **Synchronisation interactive bidirectionnelle** : Le survol de la souris sur un véhicule dans la vidéo de gauche l'illumine instantanément en 3D sur l'écran droit (et inversement).

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
- `weights/yolopv2.pt` : Poids TorchScript YOLOPv2 (149 Mo, téléchargement auto si absent)
- `yolo11s.pt` : Poids Ultralytics YOLO11 small (18.4 Mo, sweet-spot M4 par défaut)
- `yolo11m.pt` : Poids Ultralytics YOLO11 medium (38.8 Mo, mode précision maximale)
- `yolo11n.pt` : Poids Ultralytics YOLO11 nano (5.6 Mo, mode ultra-léger)

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

- **Glisser-Déposer (Drag & Drop)** : Déposez n'importe quelle vidéo (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) directement sur la fenêtre pour l'analyser.
- **Sélecteur de modèles** : Basculez en direct entre YOLO11s (Défaut), YOLO11m (Précision Max) et YOLO11n (Rapide).
- **Préréglages de caméra 3D** :
  - `FSD Conducteur` : Vue immersive 3e personne derrière la voiture hôte.
  - `Hélicoptère 45°` : Vue aérienne tactique surélevée.
  - `Top-Down` : Carte zénithale orthogonale métrique.
- **Capture & Exportation** :
  - Bouton `📸 Capture` : Enregistre instantanément une capture PNG haute résolution du double écran synchronisé dans `captures/`.
- **Bascule de viseurs 2D / 3D** : Viseur tactique par coins (*brackets*), cuboids 3D filaires, ou masquage.
- **Calibrage de caméra** : Curseur d'inclinaison (*pitch*) fin pour s'adapter à toutes les hauteurs de pare-brise.
- **Alertes de sécurité ADAS** : Détection du véhicule de tête (*Lead car*), calcul du Time-To-Collision (TTC) et alerte anticollision FCW clignotante.
