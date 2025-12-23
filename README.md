# 🎮 Flappy Bird AR + Q-Learning AI

Un innovativo gioco Flappy Bird che combina **Computer Vision**, **Realtà Aumentata** e **Reinforcement Learning**. Gli ostacoli vengono generati in tempo reale dagli oggetti fisici mostrati alla webcam, mentre un'AI basata su Q-Learning impara a giocare autonomamente.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)

---

## 🎯 Features

### 🌟 Modalità di Gioco

- **🎥 AR Mode (Realtà Aumentata):** Mostra un oggetto (es. cellulare) alla webcam e l'AI crea ostacoli nella posizione corrispondente
- **🤖 Training Mode:** Genera ostacoli casuali per allenare l'AI a velocità massima
- **⚡ Headless Mode:** Training ultra-veloce (1000-5000 FPS) senza GUI

### 🧠 Intelligenza Artificiale

- **Algoritmo:** Q-Learning con Q-Table discretizzata (1.37M stati)
- **State Space:** bird_y, dist_x, |dist_y|, direzione (4 dimensioni)
- **Actions:** 2 azioni (jump, no-jump)
- **Reward Function:** Focus su allineamento gap con penalità esponenziali
- **Exploration:** Epsilon-greedy con decay automatico

### 🎨 Tecnologie

- **Object Detection:** YOLOv8 per rilevamento oggetti real-time
- **Computer Vision:** OpenCV per elaborazione frame webcam
- **RL:** Q-Learning implementato con NumPy (no framework esterni)
- **GUI:** Visualizzazione live con statistiche (FPS, epsilon, score)

---

## 📦 Installazione

### Prerequisiti

- Python 3.8+
- Webcam (per AR Mode)
- OS: Windows, macOS, Linux

### Setup
```bash
# Clone repository
git clone https://github.com/your-username/flappy-bird-ai.git
cd flappy-bird-ai

# Virtual environment (raccomandato)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installa dipendenze
pip install opencv-python ultralytics numpy mss
```

---

## 🚀 Quick Start

### 🎥 Modalità AR (Realtà Aumentata)
```bash
python prova_flappy.py
```

**Come funziona:**
1. Webcam si attiva
2. Mostra un **cellulare** nella metà **DESTRA** dello schermo
3. Ostacolo appare nella posizione dell'oggetto
4. L'AI impara a evitarlo

**Comandi tastiera:**
- `m` → Passa a Training Mode
- `r` → Reset gioco
- `h` → Toggle Headless Mode
- `Space` → Salto manuale (debug)
- `q` → Esci

---

### 🤖 Training Mode

**Nel codice (`prova_flappy.py`):**
```python
TRAINING_MODE = True   # Attiva training
HEADLESS_MODE = False  # GUI visibile
```

**Avvia:**
```bash
python prova_flappy.py
```

**Risultati attesi:**
```
Ep   100 | Score:   5 | Avg10:   3.2 | Avg100:   3.2 | Max:  12 | ε: 0.9999 | FPS:  45
Ep   500 | Score:  18 | Avg10:  15.1 | Avg100:  12.8 | Max:  35 | ε: 0.9995 | FPS:  47
Ep  1000 | Score:  42 | Avg10:  38.5 | Avg100:  32.1 | Max:  68 | ε: 0.9990 | FPS:  46
💾 Checkpoint @ Ep 1000
```

**Convergenza:** L'AI impara in 2000-5000 episodi (~1-2 ore training normale)

---

### ⚡ Headless Mode (Ultra-Veloce)

**Setup:**
```python
TRAINING_MODE = True
HEADLESS_MODE = True  # Nessuna GUI
```

**Velocità:** 1000-5000 FPS (30-100× più veloce)

**Quando usare:**
- Training lunghi (5000+ episodi)
- Training overnight
- Server remoti senza display

---

## 📊 Architettura

### System Overview
```
┌──────────┐
│ Webcam   │
└────┬─────┘
     │
     ▼
┌──────────┐     ┌────────────┐
│ YOLOv8   │────▶│ Obstacles  │
│ Detect   │     │ Generator  │
└──────────┘     └─────┬──────┘
                       │
                       ▼
                ┌──────────────┐
                │   Physics    │
                │   Engine     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Q-Learning  │
                │    Brain     │
                └──────────────┘
```

### Q-Learning Details

**State Space (4D):**
```
State = (bird_y, dist_x, |dist_y|, direction)

- bird_y:    Altezza bird (0-720) → 73 bins
- dist_x:    Distanza ostacolo (0-1280) → 129 bins
- |dist_y|:  Distanza da gap (0-720) → 73 bins
- direction: Sopra/Sotto gap → 2 bins

Total States: 73 × 129 × 73 × 2 = 1,370,052
Q-Table Size: ~10.5 MB
```

**Actions:**
```
0: NON saltare (cadi per gravità)
1: Saltare (velocità negativa)
```

**Reward Function:**
```python
gap_distance < 20px:   +100 (PERFETTO)
gap_distance < 50px:   +40  (BUONO)
gap_distance < 100px:  +15  (OK)
gap_distance < 200px:  -20  (MALE)
gap_distance > 200px:  -30 a -333 (PESSIMO - esponenziale)

Morte:                 -15000
Sopravvivenza frame:   +2
Bordi schermo:         -10
Velocità eccessiva:    -5
```

**Hyperparameters:**
```python
Learning Rate:      0.2
Discount Factor:    0.9
Exploration Init:   0.0  (caricato da file)
Exploration Decay:  0.99999992
Min Exploration:    0.05
Grid Size:          10 pixels
```

---

## 📁 Struttura File
```
flappy-bird-ai/
│
├── prova_flappy.py          # Main game loop
├── brain_flappy.py          # Q-Learning implementation
├── flappy_brain_numpy.pkl   # Trained Q-table (auto-generato)
├── yolov8n.pt              # YOLOv8 weights (auto-scaricato)
├── README.md               # Questo file
└── requirements.txt        # Dipendenze Python
```

---

## 🎮 Configurazione

### Personalizza Oggetto Trigger
```python
# In prova_flappy.py
TRIGGER_OBJECT = 'cell phone'  # Cambia con: 'cup', 'bottle', 'person', etc.
```

**Oggetti supportati (YOLOv8 COCO):**
- `person`, `bicycle`, `car`, `motorcycle`
- `bottle`, `cup`, `fork`, `knife`, `spoon`
- `laptop`, `mouse`, `keyboard`, `cell phone`
- [Lista completa COCO classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

### Tweaking Fisica
```python
# In prova_flappy.py
gravity = 2          # Forza gravità (default: 2)
jump_strength = -15  # Potenza salto (default: -15)
bird_radius = 20     # Dimensione bird
obstacle_speed = 10  # Velocità scroll ostacoli
```

### Tweaking AI
```python
# In brain_flappy.py
GRID_SIZE = 10                    # Risoluzione discretizzazione (↓ = più preciso, più lento)
learning_rate = 0.2               # Velocità apprendimento
discount_factor = 0.90            # Importanza reward futuri
exploration_decay = 0.99999992    # Velocità riduzione esplorazione
```

---

## 📈 Monitoring Training

### Metriche Chiave

**Episode Count:** Numero totale episodi completati  
**Score:** Ostacoli superati in un episodio  
**Avg10:** Media ultimi 10 episodi (trend breve termine)  
**Avg100:** Media ultimi 100 episodi (trend lungo termine)  
**Max:** Miglior punteggio mai raggiunto  
**ε (epsilon):** Tasso esplorazione corrente (1.0 = tutto casuale, 0.05 = sfrutta conoscenza)  
**FPS:** Frame al secondo (performance)

### Salvataggio

**Automatico:**
- Ogni 50 episodi → `flappy_brain_numpy.pkl`
- Contiene Q-table + epsilon corrente

**Manuale:**
```python
brain.save_brain()  # Forza salvataggio
```

### Caricamento

**Automatico all'avvio:**
```python
brain.load_brain()  # Cerca flappy_brain_numpy.pkl
```

**Se shape incompatibile:**
- Vecchio file → backup automatico (`.pkl_OLD_3D.pkl`)
- Inizia con Q-table nuova

---

## 🐛 Troubleshooting

### Webcam non funziona

**macOS:**
```bash
# Autorizza accesso camera in:
System Preferences → Security & Privacy → Camera → Terminal/Python
```

**Linux:**
```bash
# Verifica dispositivi video
ls /dev/video*

# Cambia indice camera se necessario
cap = cv2.VideoCapture(1)  # Prova 0, 1, 2...
```

### YOLOv8 troppo lento
```python
# Riduci risoluzione webcam
cap.set(3, 640)   # Era 1280
cap.set(4, 480)   # Era 720

# Usa modello più piccolo (già usiamo yolov8n, il più veloce)
```

### Q-table troppo grande
```python
# Aumenta GRID_SIZE (meno stati)
GRID_SIZE = 20  # Era 10 (4× meno celle)
```

### AI non impara

**Verifica:**
1. `exploration_rate` iniziale = 0? → Cambia a 1.0
2. Epsilon decade troppo lento? → Riduci `exploration_decay`
3. Reward range troppo grande? → Normalizza

**Debug:**
```python
# Aggiungi dopo brain.load_brain()
print(f"Epsilon: {brain.exploration_rate}")
print(f"Q-table stats: min={brain.q_table.min()}, max={brain.q_table.max()}")
```

---

## 🎓 Come Funziona l'AI

### 1. Discretizzazione

Il gioco ha **infinite** posizioni possibili, ma Q-Learning richiede **stati discreti**:
```python
# Continuo → Discreto
bird_y = 347.6 px  →  idx = 347 // 10 = 34
dist_x = 543.2 px  →  idx = 543 // 10 = 54
dist_y = -127.8 px →  |dist_y| = 127.8 → idx = 12
                      direction = 0 (negativo = sopra)

State = (34, 54, 12, 0)
```

### 2. Epsilon-Greedy Exploration
```python
if random() < epsilon:
    action = random(0, 1)  # Esplora
else:
    action = argmax(Q[state])  # Sfrutta conoscenza
```

**Inizio:** epsilon = 1.0 → tutto casuale → esplora  
**Fine:** epsilon = 0.05 → 95% decisioni informate → sfrutta

### 3. Q-Learning Update
```python
# Bellman Equation
Q[state][action] += lr * (reward + gamma * max(Q[next_state]) - Q[state][action])
```

**Intuizione:**  
"Aggiorna valore azione basandoti su: reward immediato + miglior valore futuro possibile"

### 4. Convergenza

**Episodio 1-500:** Esplora casualmente, riempie Q-table  
**Episodio 500-2000:** Bilancia esplorazione/sfruttamento  
**Episodio 2000+:** Principalmente sfrutta conoscenza appresa  

**Segni di successo:**
- Avg100 cresce costantemente
- Max score aumenta
- Epsilon scende verso 0.05
- Bird evita pattern stupidi (non salta sempre/mai)

---

## 🔬 Esperimenti Avanzati

### 1. Curriculum Learning

Inizia con ostacoli facili, aumenta difficoltà:
```python
# In trainingMode()
if episode_count < 500:
    block_height = random.randint(150, 350)  # Gap grandi
elif episode_count < 2000:
    block_height = random.randint(120, 380)  # Gap medi
else:
    block_height = random.randint(100, 400)  # Gap difficili
```

### 2. Multi-Object AR

Più oggetti = più ostacoli:
```python
TRIGGER_OBJECTS = ['cell phone', 'bottle', 'cup']

for label in TRIGGER_OBJECTS:
    if detected(label):
        create_obstacle(position)
```

### 3. Transfer Learning

Allena su training mode, testa su AR:
```bash
# 1. Allena
TRAINING_MODE = True
python prova_flappy.py  # 2000 episodi

# 2. Testa
TRAINING_MODE = False
python prova_flappy.py  # Gioca con oggetti reali
```

### 4. Reward Shaping

Sperimenta reward diverse:
```python
# Esempio: Premia efficienza
reward += 5.0 if dist_x < 100 and gap_distance < 30 else 0

# Esempio: Penalizza movimenti bruschi
reward -= abs(bird_velocity - prev_velocity) * 0.5
```

---

## 📚 Risorse

### Q-Learning
- [Sutton & Barto - RL Book](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

### Computer Vision
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### Progetti Simili
- [FlappyBird-DeepRL](https://github.com/yenchenlin/DeepLearningFlappyBird)
- [OpenAI Gym](https://gymnasium.farama.org/)

---

## 🤝 Contribuire

Pull requests benvenute! Per modifiche maggiori:

1. Apri issue per discutere modifiche
2. Fork del repository
3. Crea branch (`git checkout -b feature/AmazingFeature`)
4. Commit (`git commit -m 'Add AmazingFeature'`)
5. Push (`git push origin feature/AmazingFeature`)
6. Apri Pull Request

---

## 🎯 Roadmap Futuro

- [ ] Deep Q-Network (DQN) con neural network
- [ ] Multi-agent training (competizione tra AI)
- [ ] Web interface per training remoto
- [ ] Leaderboard online
- [ ] Mobile app (React Native)
- [ ] VR support

---

## 👤 Autore

**Your Name**  
GitHub: [@Eddicpp](https://github.com/Eddicpp)  
Email: eduardo.pane04@gmail.com

---

## 🙏 Ringraziamenti

- **Ultralytics** per YOLOv8
- **OpenCV** community
- **Reinforcement Learning** community

---

## 📊 Statistics
```
Lines of Code:     ~800
Training Time:     1-3 ore (2000 episodi)
Q-Table Size:      10.5 MB
FPS (GUI):         30-60
FPS (Headless):    1000-5000
Success Rate:      Score 50+ dopo 3000 episodi
```

---

**⭐ Se questo progetto ti è stato utile, lascia una stella su GitHub!**
