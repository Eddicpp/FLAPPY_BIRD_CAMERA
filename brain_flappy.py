import numpy as np
import pickle
import os

# --- COSTANTI ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
GRID_SIZE = 10

# ⭐ FIX: 4 dimensioni invece di 3
VERTICAL_POS_BIRD_GRID_SIZE = int(SCREEN_HEIGHT / GRID_SIZE) + 1           # 0-720
HORIZONTAL_DIST_NEXT_PIPE_GRID_SIZE = int(SCREEN_WIDTH / GRID_SIZE) + 1    # 0-1280
GAP_DISTANCE_GRID_SIZE = int(SCREEN_HEIGHT / GRID_SIZE) + 1                # 0-720 (SOLO distanza assoluta)
DIRECTION_SIZE = 2  # 0 = sopra target (dist_y negativo), 1 = sotto target (dist_y positivo)
ACTIONS_SIZE = 2

class QLearningBrain:
    def __init__(self, learning_rate=0.2, discount_factor=0.90, exploration_rate=0):
        #                                                                        ^^^
        # ⭐ FIX CRITICO: Inizia con esplorazione ALTA!
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.99999992# ⭐ Decay ragionevole
        self.min_exploration_rate = 0.05

        # ⭐ Q-table con 5 dimensioni: [bird_y, dist_x, |dist_y|, direzione, azione]
        self.q_table = np.zeros((VERTICAL_POS_BIRD_GRID_SIZE,
                                 HORIZONTAL_DIST_NEXT_PIPE_GRID_SIZE,
                                 GAP_DISTANCE_GRID_SIZE,      # ← Solo valore assoluto
                                 DIRECTION_SIZE,              # ← Sopra/Sotto
                                 ACTIONS_SIZE))
        
        # Bias leggero
        self.q_table[:, :, :, :, 0] = 0.2  # NON saltare
        self.q_table[:, :, :, :, 1] = -0.2  # Saltare
        
        total_cells = self.q_table.size
        print(f"\n📊 Q-TABLE INIZIALIZZATA:")
        print(f"   Shape: {self.q_table.shape}")
        print(f"   Dimensioni:")
        print(f"     Bird Y: {VERTICAL_POS_BIRD_GRID_SIZE}")
        print(f"     Dist X: {HORIZONTAL_DIST_NEXT_PIPE_GRID_SIZE}")
        print(f"     |Dist Y|: {GAP_DISTANCE_GRID_SIZE}")
        print(f"     Direction: {DIRECTION_SIZE} (0=sopra, 1=sotto)")
        print(f"     Actions: {ACTIONS_SIZE}")
        print(f"   Total cells: {total_cells:,}")
        print(f"   Memory: {total_cells * 8 / 1024 / 1024:.2f} MB")
        print(f"   Exploration: {self.exploration_rate} → {self.min_exploration_rate}")
        print(f"   Decay: {self.exploration_decay}\n")

    def discretize_state(self, bird_y, dist_x, dist_y):
        """Converte stato → indici (SEPARANDO direzione)"""
        
        # Bird Y (0-720)
        idx_bird_y = int(bird_y // GRID_SIZE)
        idx_bird_y = max(0, min(idx_bird_y, VERTICAL_POS_BIRD_GRID_SIZE - 1))
        
        # Distance X (0-1280+)
        idx_dist_x = int(dist_x // GRID_SIZE)
        idx_dist_x = max(0, min(idx_dist_x, HORIZONTAL_DIST_NEXT_PIPE_GRID_SIZE - 1))
        
        # ⭐ Distance Y - VALORE ASSOLUTO (0-720)
        dist_y_abs = abs(dist_y)
        idx_dist_y = int(dist_y_abs // GRID_SIZE)
        idx_dist_y = max(0, min(idx_dist_y, GAP_DISTANCE_GRID_SIZE - 1))
        
        # ⭐ DIREZIONE (informazione esplicita!)
        # 0 = bird SOPRA target (dist_y < 0, deve SCENDERE/non saltare)
        # 1 = bird SOTTO target (dist_y > 0, deve SALIRE/saltare)
        idx_direction = 0 if dist_y < 0 else 1
        
        return (idx_bird_y, idx_dist_x, idx_dist_y, idx_direction)

    def choose_action(self, state):
        """Epsilon-greedy"""
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(0, ACTIONS_SIZE)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """Q-learning update"""
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

        # Decay
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

    def reward_function(self, bird_y, bird_velocity, dist_x, dist_y_gap, game_over, action_taken):
        """
        ═══════════════════════════════════════════════════
        REWARD: FOCUS ASSOLUTO su dist_y_gap (gap_distance)
        ═══════════════════════════════════════════════════
        
        OBIETTIVO PRIMARIO: Mantenere gap_distance il PIÙ BASSO possibile
        Tutto il resto è secondario o ignorato
        """
        
        # ═══ 1. MORTE ═══
        if game_over:
            return -15000.0  # Penalità forte
        
        # ═══ 2. CALCOLO GAP DISTANCE (Distanza dal centro gap) ═══
        gap_distance = abs(dist_y_gap)
        
        #print("gap distance: ")
        #print(gap_distance)

        # ═══ 3. REWARD DOMINANTE: Inversamente proporzionale a gap_distance ═══
        # Più sei vicino al centro gap, più reward ottieni (ESPONENZIALE)
        
        if gap_distance < 20:
            # PERFETTO! (0-20px dal centro)
            reward = 100.0 * (1.0 - gap_distance / 20.0)
            # gap_distance = 0  → reward = 100
            # gap_distance = 10 → reward = 50
            # gap_distance = 20 → reward = 0
            
        elif gap_distance < 50:
            # BUONO (20-50px)
            reward = 40.0 * (1.0 - (gap_distance - 20) / 30.0)
            # gap_distance = 20 → reward = 40
            # gap_distance = 35 → reward = 20
            # gap_distance = 50 → reward = 0
            
        elif gap_distance < 100:
            # ACCETTABILE (50-100px)
            reward = 15.0 * (1.0 - (gap_distance - 50) / 50.0)
            # gap_distance = 50  → reward = 15
            # gap_distance = 75  → reward = 7.5
            # gap_distance = 100 → reward = 0
            
        elif gap_distance < 200:
            # MEDIOCRE (100-200px)
            reward = -20.0
            
        else:
            # PESSIMO (>200px) - PENALITÀ ESPONENZIALE
            overshoot = gap_distance - 200
            reward = -30.0 - (overshoot ** 1.2) * 0.1
            # gap_distance = 250 → reward ≈ -66
            # gap_distance = 300 → reward ≈ -130
            # gap_distance = 400 → reward ≈ -333
        

        if dist_y_gap < -30 and action_taken == 0: reward-=40
        elif dist_y_gap > 30 and action_taken == 1: reward-=1500
        # ═══ 4. BONUS SOPRAVVIVENZA (molto piccolo) ═══
        reward += 2.0
        
        # ═══ 5. PENALITÀ VELOCITÀ ECCESSIVA (minore) ═══
        if abs(bird_velocity) > 15:
            reward -= 5.0
        
        # ═══ 6. PENALITÀ BORDI SCHERMO (minore) ═══
        if bird_y < 50 or bird_y > 670:
            reward -= 10.0
        
        return reward

    def save_brain(self, filename="flappy_brain_numpy.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'exploration_rate': self.exploration_rate,
                    'shape': self.q_table.shape
                }, f)
        except Exception as e:
            print(f"❌ Errore: {e}")

    def load_brain(self, filename="flappy_brain_numpy.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict):
                    loaded_table = data['q_table']
                else:
                    loaded_table = data
                
                # ⭐ Verifica shape
                if loaded_table.shape == self.q_table.shape:
                    self.q_table = loaded_table
                    print(f"✅ Cervello caricato! ε={self.exploration_rate:.4f}\n")
                else:
                    print(f"\n⚠️  SHAPE INCOMPATIBILE!")
                    print(f"    Salvato: {loaded_table.shape}")
                    print(f"    Atteso:  {self.q_table.shape}")
                    print(f"    → Uso cervello NUOVO (vecchio → backup)\n")
                    
                    import shutil
                    backup = filename.replace('.pkl', '_OLD_3D.pkl')
                    shutil.move(filename, backup)
                    print(f"📦 Backup: {backup}\n")
                    
            except Exception as e:
                print(f"❌ Errore: {e}\n")
        else:
            print("⚠️  Nessun cervello salvato\n")
    
    def reset_brain(self):
        self.q_table.fill(0)
        self.q_table[:, :, :, :, 0] = 0.2
        self.q_table[:, :, :, :, 1] = 0.0
        self.exploration_rate = 1.0
        print("☢️  RESET")

    def printTable(self):
        print(self.q_table[:, :, :, :, 0])
        print(self.q_table[:, :, :, :, 1])
        '''for i in range(100):
            for j in range(3):
                  self.q_table[:, :, :, :, 0]
                  self.q_table[:, :, :, :, 1]'''