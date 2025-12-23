import cv2
import numpy as np
from ultralytics import YOLO
import time
import random

from brain_flappy import QLearningBrain

# --- CONFIGURAZIONE ---
print("Caricamento cervello...")
model = YOLO('yolov8n.pt') 

TRIGGER_OBJECT = 'cell phone' 

TRAINING_MODE = False
HEADLESS_MODE = False # True = VELOCE (no GUI), False = NORMALE (con GUI)
RENDER_EVERY_N = 100  # Mostra UI ogni N episodi (solo se HEADLESS_MODE=False)

# Variabili Gioco GLOBALI
bird_y = 400
bird_velocity = 0
gravity = 2
jump_strength = -15
bird_x = 100
bird_radius = 20

obstacles = [] 
obstacle_speed = 10
obstacle_width = 80
last_spawn_time = 0

score = 0
game_over = False

# Dimensioni Schermo Globali (servono per phisics)
width = 1280
height = 720

# Setup Webcam
if not HEADLESS_MODE:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
else:
    cap = None
    print("🚀 MODALITÀ HEADLESS: Training ultra-veloce ATTIVO")

brain = QLearningBrain()
brain.load_brain()

print("GIOCO AVVIATO!")
if HEADLESS_MODE:
    print("⚡ Velocità training: 1000-5000 FPS stimati")
else:
    print(f"MODALITA' 1: Mostra un '{TRIGGER_OBJECT}' a DESTRA.")
    print(f"MODALITA' 2: Premi 'm' per attivare l'Allenamento Automatico.")

def create_initial_obstacle():
    block_height = random.randint(100, 400)
    block_y1 = random.randint(50, 720 - block_height - 50)
    block_y2 = block_y1 + block_height
    return [1280, obstacle_width, block_y1, block_y2] # Start x=1280 (width)


previous_state = None
last_action = 0
episode_count = 0
scores_history = []

# ⭐ FPS Counter
start_time = time.time()
frame_count = 0

# --- FUNZIONI ---

def gestisci_input():
    # Definisco quali variabili globali vado a modificare
    global game_over, bird_y, bird_velocity, obstacles, score, previous_state, last_spawn_time, TRAINING_MODE, HEADLESS_MODE

    if not HEADLESS_MODE:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False # Interrompe il loop
        elif key == ord('r'):
            game_over = False
            bird_y = random.randint(200, 520)
            bird_velocity = 0
            obstacles = []
            obstacles.append(create_initial_obstacle())
            score = 0
            previous_state = None
            last_spawn_time = time.time()
        elif key == ord('m'):
            TRAINING_MODE = not TRAINING_MODE
            obstacles = []
            obstacles.append(create_initial_obstacle())
            print(f"Modalità: {'TRAINING' if TRAINING_MODE else 'AR'}")
        elif key == ord(' ') and not game_over:
            bird_velocity = jump_strength
        elif key == ord('h'):
            HEADLESS_MODE = not HEADLESS_MODE
            if HEADLESS_MODE:
                print("🚀 HEADLESS MODE ON")
                cv2.destroyAllWindows()
            else:
                print("👁️ GUI MODE ON")
    return True # Continua il loop

def phisics():
    # Definisco variabili globali da usare/modificare
    global bird_y, bird_velocity, obstacles, score, game_over, previous_state, last_action, episode_count, scores_history, brain
    
    dist_x = obstacles[0][0] - bird_x if len(obstacles) > 0 else width
    dist_y = 0
    
    if len(obstacles) > 0:
        obs = obstacles[0]
        
        # Dimensioni gap
        gap_top_size = obs[2]
        gap_bottom_size = 720 - obs[3]
        
        # Target (centro gap)
        target_top = gap_top_size / 2
        target_bottom = obs[3] + (gap_bottom_size / 2)
        
        # ⭐ SEMPLICE: Scegli sempre il gap PIÙ GRANDE
        if gap_top_size > gap_bottom_size:
            dist_y = target_top - bird_y
            #print('è più grande quello SOPRA:') 
            #print(dist_y)

        else:
            dist_y = target_bottom - bird_y
            #print('è più grande quello SOTTO')
            #print(dist_y)
    
    current_state = brain.discretize_state(bird_y, dist_x, dist_y)
    
    if not game_over:
        action = brain.choose_action(current_state)
        
        if action == 1:
            bird_velocity = jump_strength # Assegnazione diretta per salto secco
        
        bird_velocity += gravity
        bird_y += bird_velocity
        
        if previous_state is not None:
            reward = brain.reward_function(bird_y, bird_velocity, dist_x, dist_y, False, last_action)
            brain.learn(previous_state, last_action, reward, current_state)
        
        previous_state = current_state
        last_action = action
        
        for obs in obstacles:
            obs[0] -= obstacle_speed
        
        if len(obstacles) > 0 and obstacles[0][0] < -obstacle_width:
            obstacles.pop(0)
            score += 1
        
        collision = False
        if bird_y >= height - bird_radius or bird_y <= bird_radius:
            collision = True
        
        for obs in obstacles:
            ox, ow, oy1, oy2 = obs
            if (bird_x + bird_radius > ox) and (bird_x - bird_radius < ox + ow):
                if (bird_y + bird_radius > oy1) and (bird_y - bird_radius < oy2):
                    collision = True
        
        if collision:
            game_over = True
            episode_count += 1
            scores_history.append(score)
            
            reward = brain.reward_function(bird_y, bird_velocity, dist_x, dist_y, True, last_action)
            brain.learn(previous_state, last_action, reward, current_state)
            
            # LOG
            avg_10 = np.mean(scores_history[-10:]) if len(scores_history) >= 10 else score
            avg_100 = np.mean(scores_history[-100:]) if len(scores_history) >= 100 else avg_10
            max_score = max(scores_history) if scores_history else 0
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"Ep {episode_count:5d} | "
                  f"Score: {score:3d} | "
                  f"Avg10: {avg_10:5.1f} | "
                  f"Avg100: {avg_100:5.1f} | "
                  f"Max: {max_score:3d} | "
                  f"ε: {brain.exploration_rate:.4f} | "
                  f"FPS: {fps:6.0f}")
            
            if episode_count % 50 == 0:
                brain.save_brain()
                print(f"💾 Checkpoint @ Ep {episode_count}")

def normalMode():
    global last_spawn_time, obstacles, game_over, bird_y, bird_velocity, score, previous_state, frame_count
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        # height e width sono globali

        current_time = time.time()

        cv2.line(frame, (640, 0), (640, 720), (255, 255, 255), 2)
        cv2.putText(frame, "AREA SPAWN ->", (660, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        detected = False
        obs_y1, obs_y2 = 0, 0
        results = model(frame, stream=True, verbose=False)
        
        for r in results:   
            for box in r.boxes: 
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                if label == TRIGGER_OBJECT and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if x1 > 640:
                        detected = True
                        obs_y1, obs_y2 = y1, y2
                        cv2.putText(frame, "RILEVATO!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else: 
                        cv2.putText(frame, "SPOSTA A DESTRA ->", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        if detected and (current_time - last_spawn_time > 2):
            obstacles.append([width, obstacle_width, obs_y1, obs_y2])
            last_spawn_time = current_time
        
        phisics()

        if frame is not None:
            mode_text = f"AR ({TRIGGER_OBJECT})"
            color_text = (0, 0, 255) if TRAINING_MODE else (0, 255, 0)
            cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2)
            cv2.putText(frame, f"Episode: {episode_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for obs in obstacles:
                ox, ow, oy1, oy2 = obs
                cv2.rectangle(frame, (ox, oy1), (ox+ow, oy2), (0, 0, 255), -1)
                cv2.rectangle(frame, (ox, oy1), (ox+ow, oy2), (0, 0, 0), 3)
            
            color_bird = (0, 255, 255) if not game_over else (0, 0, 255)
            cv2.circle(frame, (bird_x, int(bird_y)), bird_radius, color_bird, -1)
            
            cv2.putText(frame, f"Score: {score}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            if game_over:
                cv2.putText(frame, "GAME OVER", (width//2 - 200, height//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            
            cv2.imshow('Flappy Bird AR', frame)

        if game_over:
            # Qui il reset manuale lo gestiamo tramite gestisci_input, 
            # ma se vuoi reset automatico in Normal Mode (non consigliato in AR):
            pass 

        frame_count += 1
        continuo = gestisci_input()
        if (not continuo): break

def trainingMode():
    global last_spawn_time, obstacles, game_over, bird_y, bird_velocity, score, previous_state, frame_count

    obstacles.append(create_initial_obstacle())

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        # height, width globali

        current_time = time.time() 

        if not game_over and current_time - last_spawn_time > 2:
            # Generazione random
            block_height = random.randint(100, 400)
            block_y1 = random.randint(50, 720 - block_height - 50)
            block_y2 = block_y1 + block_height
            obstacles.append([width, obstacle_width, block_y1, block_y2])
            last_spawn_time = current_time

        phisics()

        if frame is not None:
            mode_text = "TRAINING MODE"
            color_text = (0, 0, 255)
            cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2)
            cv2.putText(frame, f"Episode: {episode_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for obs in obstacles:
                ox, ow, oy1, oy2 = obs
                cv2.rectangle(frame, (ox, oy1), (ox+ow, oy2), (0, 0, 255), -1)
                cv2.rectangle(frame, (ox, oy1), (ox+ow, oy2), (0, 0, 0), 3)
            
            color_bird = (0, 255, 255) if not game_over else (0, 0, 255)
            cv2.circle(frame, (bird_x, int(bird_y)), bird_radius, color_bird, -1)
            
            cv2.putText(frame, f"Score: {score}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            if game_over:
                cv2.putText(frame, "GAME OVER", (width//2 - 200, height//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            
            cv2.imshow('Flappy Bird AR', frame)

        if game_over:
            # Reset immediato in training
            game_over = False
            bird_y = random.randint(200, 520)
            bird_velocity = 0
            obstacles = []
            obstacles.append(create_initial_obstacle())
            score = 0
            previous_state = None
            last_spawn_time = time.time()

        frame_count += 1
        continuo = gestisci_input()
        if (not continuo): break

def speedMode():
    global last_spawn_time, obstacles, game_over, bird_y, bird_velocity, score, previous_state, frame_count

    obstacles.append(create_initial_obstacle())
    
    # In speed mode non c'è frame, quindi niente width/height da cap
    # usiamo width/height globali
    
    while True:
        current_time = time.time()
        
        # Spawn ostacoli anche qui
        if not game_over and current_time - last_spawn_time > 2: # 1.5s fittizio, in speed mode il tempo vola
            block_height = random.randint(100, 400)
            block_y1 = random.randint(50, 720 - block_height - 50)
            block_y2 = block_y1 + block_height
            obstacles.append([width, obstacle_width, block_y1, block_y2])
            last_spawn_time = current_time
            
        phisics()
        
        if game_over:
             # Reset immediato
            game_over = False
            bird_y = random.randint(200, 520)
            bird_velocity = 0
            obstacles = []
            obstacles.append(create_initial_obstacle())
            score = 0
            previous_state = None
            last_spawn_time = time.time()
            
        frame_count += 1
        # In speed mode non controlliamo i tasti cv2 ogni frame altrimenti rallenta, 
        # ma se vuoi uscire serve un modo. Per ora lasciamo girare veloce.

# --- SELEZIONE MODALITÀ ---

if not TRAINING_MODE and not HEADLESS_MODE:
    time.sleep(2)
    normalMode()
elif TRAINING_MODE and not HEADLESS_MODE:
    time.sleep(2)
    trainingMode()
else:
    # Headless / Speed Mode
    speedMode()

# Cleanup
if cap is not None:
    cap.release()
cv2.destroyAllWindows()

print(f"\n📊 STATISTICHE FINALI:")
print(f"   Episodi totali: {episode_count}")
print(f"   Tempo totale: {time.time() - start_time:.1f}s")
print(f"   FPS medio: {frame_count / (time.time() - start_time):.0f}")
if scores_history:
    print(f"   Score medio: {np.mean(scores_history):.1f}")
    print(f"   Score massimo: {max(scores_history)}")