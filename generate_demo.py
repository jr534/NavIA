import os
import sys
import colorsys
import random
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path so we can import train.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import train

# --- Configuration Visuelle ---
BG_COLOR = (215, 220, 244, 255)       # #D7DCF4
BASE_BLUE = (63, 86, 189)             # #3F56BD
BASE_HUE = 229 / 360.0                 # Hue pour #3F56BD (environ 229 deg)

# Couleurs des panels
PANEL_BG = (240, 243, 255, 255)
CENTER_PANEL_BG = (245, 247, 255, 255)
TEXT_COLOR_DARK = (30, 34, 43, 255)
TEXT_COLOR_MUTED = (90, 100, 120, 255)

# Dimensions de la vidéo
WIDTH = 1920
HEIGHT = 1080
FPS = 24

# --- Helper pour charger une police ---
def get_font(size, bold=False):
    font_names = ["segoeui.ttf", "segoeuib.ttf" if bold else "segoeui.ttf", "arial.ttf", "consolas.ttf"]
    for name in font_names:
        path = os.path.join("C:\\Windows\\Fonts", name)
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    return ImageFont.load_default()

# --- Helper pour les variations de couleur HSL ---
def get_heatmap_color(prob):
    """Génère une couleur de heatmap basée sur la probabilité.
    Garde le même Hue (229°) de #3F56BD, mais fait varier L et S.
    - prob proche de 0 : très clair et peu saturé (mélange avec le fond)
    - prob proche de 1 : très saturé et plus foncé (bleu intense)
    """
    # S varie de 0.20 à 0.95
    s = 0.20 + 0.75 * prob
    # L varie de 0.93 (très clair) à 0.40 (bleu profond)
    l = 0.93 - 0.53 * prob
    r, g, b = colorsys.hls_to_rgb(BASE_HUE, l, s)
    return int(r * 255), int(g * 255), int(b * 255), 255

def get_blue_variant(s_factor, l_factor):
    """Retourne une variante directe de bleu en modifiant S et L."""
    r, g, b = colorsys.hls_to_rgb(BASE_HUE, l_factor, s_factor)
    return int(r * 255), int(g * 255), int(b * 255)

# --- Classes de jeu ---
class GamePlayer:
    def __init__(self, name):
        self.name = name
        self.ships_board = train.Grille()
        # Génère silencieusement la grille de navires
        self.ships_board.gen_grille()
        
        # Copie de la grille initiale des navires (pour affichage de la grille de défense)
        self.initial_ships = [[self.ships_board.grille[y][x] for x in range(10)] for y in range(10)]
        
        # Grille masquée (représente l'état des tirs effectués sur l'adversaire)
        # 0 = non tiré, 1 = touché, -1 = manqué (encodage cohérent avec backend.py)
        self.hiden_grille = [[0 for _ in range(10)] for _ in range(10)]
        
        # Statistiques
        self.shots_count = 0
        self.hits_count = 0
        self.misses_count = 0
        self.last_shot = None # (y, x, result)

    def get_remaining_ship_cells(self):
        return sum(sum(ligne) for ligne in self.ships_board.grille)

    def get_accuracy(self):
        if self.shots_count == 0:
            return 0
        return int((self.hits_count / self.shots_count) * 100)

class BattleshipDemoGenerator:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        
        # Chargement du modèle MLP
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = train.BattleshipMLP()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Modèle chargé avec succès sur {self.device}")

        # Initialisation des polices
        self.font_title = get_font(38, bold=True)
        self.font_subtitle = get_font(20)
        self.font_header = get_font(24, bold=True)
        self.font_stats = get_font(18, bold=True)
        self.font_text = get_font(16)
        self.font_log = get_font(15)
        
        # Initialisation des joueurs
        self.player1 = GamePlayer("IA 1")
        self.player2 = GamePlayer("IA 2")
        
        # Logs de combat
        self.combat_logs = ["Début du combat IA vs IA !"]
        self.tour_number = 1

    def run_inference(self, player):
        """Exécute l'inférence pour obtenir la heatmap de probabilités du joueur."""
        # Encodage conforme à backend.py : 1 pour touché, -1 pour manqué, 0 pour vide
        hidden_flat = torch.tensor(
            [cell for ligne in player.hiden_grille for cell in ligne],
            dtype=torch.float32,
            device=self.device
        )
        
        with torch.no_grad():
            probas, logits = self.model(hidden_flat)
            probas = torch.sigmoid(logits)
            
            # Masquer à -infini les cases déjà tirées pour le calcul du argmax
            probas_masked = probas.clone()
            for y in range(10):
                for x in range(10):
                    if player.hiden_grille[y][x] != 0:
                        probas_masked[y * 10 + x] = -float("inf")
            
            # Sélection de la case cible
            case_idx = probas_masked.argmax().item()
            target_y = case_idx // 10
            target_x = case_idx % 10
            
            # Retourne la heatmap brute (convertie en liste) et les coordonnées cibles
            return probas.cpu().tolist(), target_y, target_x

    def add_log(self, text):
        self.combat_logs.append(text)
        if len(self.combat_logs) > 6:
            self.combat_logs.pop(0)

    def draw_panel_card(self, draw, x1, y1, x2, y2, fill_color, border_color, radius=15):
        """Dessine une carte de panneau avec des coins arrondis et une bordure subtile."""
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill_color)
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=None, outline=border_color, width=2)

    def draw_grid(self, draw, start_x, start_y, cell_size, gap, hiden_grille, heatmap, target_cell=None, highlight=False):
        """Dessine la grille principale de tir (10x10) avec heatmap et tirs."""
        # Normalisation locale de la heatmap pour un rendu visuel optimal
        min_p = min(heatmap)
        max_p = max(heatmap)
        diff_p = max_p - min_p if (max_p - min_p) > 1e-6 else 1.0
        
        for y in range(10):
            for x in range(10):
                val = hiden_grille[y][x]
                cell_x = start_x + x * (cell_size + gap)
                cell_y = start_y + y * (cell_size + gap)
                
                # Couleur de fond par défaut selon la heatmap
                prob = (heatmap[y * 10 + x] - min_p) / diff_p
                cell_color = get_heatmap_color(prob)
                
                # Dessin du fond de la case
                draw.rounded_rectangle([cell_x, cell_y, cell_x + cell_size, cell_y + cell_size], radius=6, fill=cell_color)
                
                # Si c'est déjà tiré
                if val == -1: # Raté
                    # Fond grisé
                    bg_miss = get_blue_variant(0.05, 0.90) # très délavé et clair
                    draw.rounded_rectangle([cell_x, cell_y, cell_x + cell_size, cell_y + cell_size], radius=6, fill=bg_miss)
                    # Point au centre
                    dot_color = BASE_BLUE
                    r_dot = 4
                    cx, cy = cell_x + cell_size/2, cell_y + cell_size/2
                    draw.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot], fill=dot_color)
                elif val == 1: # Touché
                    # Fond bleu foncé
                    bg_hit = get_blue_variant(0.85, 0.35)
                    draw.rounded_rectangle([cell_x, cell_y, cell_x + cell_size, cell_y + cell_size], radius=6, fill=bg_hit)
                    # Croix rouge d'explosion
                    cx, cy = cell_x + cell_size/2, cell_y + cell_size/2
                    offset = 8
                    draw.line([cx - offset, cy - offset, cx + offset, cy + offset], fill=(255, 82, 82), width=3)
                    draw.line([cx + offset, cy - offset, cx - offset, cy + offset], fill=(255, 82, 82), width=3)

                # Si c'est la case ciblée active
                if highlight and target_cell == (y, x):
                    # Bordure dorée d'avertissement de tir
                    draw.rounded_rectangle(
                        [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                        radius=6,
                        fill=None,
                        outline=(255, 200, 0),
                        width=3
                    )

    def draw_defense_grid(self, draw, start_x, start_y, cell_size, gap, ships_board, initial_ships):
        """Dessine la petite grille de défense (les navires réels et les impacts)."""
        for y in range(10):
            for x in range(10):
                cell_x = start_x + x * (cell_size + gap)
                cell_y = start_y + y * (cell_size + gap)
                
                has_ship = initial_ships[y][x] == 1
                is_hit = has_ship and ships_board.grille[y][x] == 0
                
                if is_hit:
                    # Navire touché/détruit : rouge/sombre
                    cell_color = (255, 82, 82)
                elif has_ship:
                    # Navire intact : bleu thématique
                    cell_color = BASE_BLUE
                else:
                    # Vide
                    cell_color = get_blue_variant(0.10, 0.95)
                
                draw.rounded_rectangle([cell_x, cell_y, cell_x + cell_size, cell_y + cell_size], radius=3, fill=cell_color)

    def generate_image_frame(self, active_player, target_cell=None, highlight_state=None, effect_radius=0):
        """Génère l'image PIL complète d'une frame du jeu."""
        # Création de l'image de fond
        img = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)
        
        # --- EN-TÊTE ---
        draw.text((WIDTH//2, 50), "SIMULATION DE COMBAT BATAILLE NAVALE IA", fill=TEXT_COLOR_DARK, font=self.font_title, anchor="mm")
        draw.text((WIDTH//2, 95), "Affrontement au tour par tour de deux AIs basées sur le modèle MLP", fill=TEXT_COLOR_MUTED, font=self.font_subtitle, anchor="mm")
        
        # --- CALCUL DES HEATMAPS ET CIBLES ---
        # IA 1
        heatmap_p1, target_y_p1, target_x_p1 = self.run_inference(self.player1)
        # IA 2
        heatmap_p2, target_y_p2, target_x_p2 = self.run_inference(self.player2)
        
        # --- PANNEAU DE GAUCHE : IA 1 ---
        card1_x1, card1_y1, card1_x2, card1_y2 = 70, 160, 670, 990
        border_p1 = BASE_BLUE if active_player == 1 else (210, 215, 235)
        self.draw_panel_card(draw, card1_x1, card1_y1, card1_x2, card1_y2, PANEL_BG, border_p1)
        
        # Titres IA 1
        draw.text((370, 200), "IA 1 (Modèle MLP)", fill=TEXT_COLOR_DARK, font=self.font_header, anchor="mm")
        draw.text((370, 230), "Grille d'Attaque (Heatmap de prédiction)", fill=TEXT_COLOR_MUTED, font=self.font_subtitle, anchor="mm")
        
        # Grille d'attaque IA 1 (tire sur IA 2)
        grid1_x, grid1_y = 142, 270
        grid_cell_size = 42
        grid_gap = 4
        self.draw_grid(
            draw, grid1_x, grid1_y, grid_cell_size, grid_gap, 
            self.player1.hiden_grille, heatmap_p1,
            target_cell=target_cell if active_player == 1 else None,
            highlight=(highlight_state is not None and active_player == 1)
        )
        
        # Grille de défense IA 1 (ses propres navires subissant les tirs de IA 2)
        def1_x, def1_y = 271, 785
        draw.text((370, 755), "Grille de Défense (Navires de l'IA 1)", fill=TEXT_COLOR_DARK, font=self.font_stats, anchor="mm")
        self.draw_defense_grid(draw, def1_x, def1_y, 18, 2, self.player1.ships_board, self.player1.initial_ships)

        # --- PANNEAU DE DROITE : IA 2 ---
        card2_x1, card2_y1, card2_x2, card2_y2 = 1250, 160, 1850, 990
        border_p2 = BASE_BLUE if active_player == 2 else (210, 215, 235)
        self.draw_panel_card(draw, card2_x1, card2_y1, card2_x2, card2_y2, PANEL_BG, border_p2)
        
        # Titres IA 2
        draw.text((1550, 200), "IA 2 (Modèle MLP)", fill=TEXT_COLOR_DARK, font=self.font_header, anchor="mm")
        draw.text((1550, 230), "Grille d'Attaque (Heatmap de prédiction)", fill=TEXT_COLOR_MUTED, font=self.font_subtitle, anchor="mm")
        
        # Grille d'attaque IA 2 (tire sur IA 1)
        grid2_x, grid2_y = 1322, 270
        self.draw_grid(
            draw, grid2_x, grid2_y, grid_cell_size, grid_gap,
            self.player2.hiden_grille, heatmap_p2,
            target_cell=target_cell if active_player == 2 else None,
            highlight=(highlight_state is not None and active_player == 2)
        )
        
        # Grille de défense IA 2 (ses propres navires subissant les tirs de IA 1)
        def2_x, def2_y = 1451, 785
        draw.text((1550, 755), "Grille de Défense (Navires de l'IA 2)", fill=TEXT_COLOR_DARK, font=self.font_stats, anchor="mm")
        self.draw_defense_grid(draw, def2_x, def2_y, 18, 2, self.player2.ships_board, self.player2.initial_ships)

        # --- PANNEAU CENTRAL : STATISTIQUES ET FIL DE COMBAT ---
        card3_x1, card3_y1, card3_x2, card3_y2 = 705, 160, 1215, 990
        self.draw_panel_card(draw, card3_x1, card3_y1, card3_x2, card3_y2, CENTER_PANEL_BG, (215, 220, 240))
        
        draw.text((960, 200), "ÉTAT DE LA PARTIE", fill=TEXT_COLOR_DARK, font=self.font_header, anchor="mm")
        draw.text((960, 250), f"TOUR {self.tour_number}", fill=BASE_BLUE, font=get_font(32, bold=True), anchor="mm")
        
        # Stats IA 1 (dans le panneau central)
        draw.text((750, 320), "IA 1", fill=TEXT_COLOR_DARK, font=self.font_stats, anchor="lm")
        p1_cells = self.player1.get_remaining_ship_cells()
        draw.text((1170, 320), f"Navires : {p1_cells}/17 cases", fill=TEXT_COLOR_DARK, font=self.font_text, anchor="rm")
        # Jauge de vie IA 1
        draw.rounded_rectangle([750, 340, 1170, 352], radius=4, fill=(220, 225, 240))
        hp_p1_width = int(420 * (p1_cells / 17))
        if hp_p1_width > 0:
            draw.rounded_rectangle([750, 340, 750 + hp_p1_width, 352], radius=4, fill=BASE_BLUE)
        draw.text((750, 375), f"Tirs : {self.player1.shots_count}   Précision : {self.player1.get_accuracy()}%", fill=TEXT_COLOR_MUTED, font=self.font_text, anchor="lm")
        
        # Séparateur subtil
        draw.line([750, 410, 1170, 410], fill=(210, 215, 235), width=1)
        
        # Stats IA 2 (dans le panneau central)
        draw.text((750, 450), "IA 2", fill=TEXT_COLOR_DARK, font=self.font_stats, anchor="lm")
        p2_cells = self.player2.get_remaining_ship_cells()
        draw.text((1170, 450), f"Navires : {p2_cells}/17 cases", fill=TEXT_COLOR_DARK, font=self.font_text, anchor="rm")
        # Jauge de vie IA 2
        draw.rounded_rectangle([750, 470, 1170, 482], radius=4, fill=(220, 225, 240))
        hp_p2_width = int(420 * (p2_cells / 17))
        if hp_p2_width > 0:
            draw.rounded_rectangle([750, 470, 750 + hp_p2_width, 482], radius=4, fill=BASE_BLUE)
        draw.text((750, 505), f"Tirs : {self.player2.shots_count}   Précision : {self.player2.get_accuracy()}%", fill=TEXT_COLOR_MUTED, font=self.font_text, anchor="lm")
        
        # Séparateur subtil
        draw.line([750, 540, 1170, 540], fill=(210, 215, 235), width=1)

        # Fil d'actualité du combat (Logs)
        draw.text((960, 575), "JOURNAL DE COMBAT", fill=TEXT_COLOR_DARK, font=self.font_stats, anchor="mm")
        
        log_y_start = 615
        for idx, log in enumerate(self.combat_logs):
            text_col = TEXT_COLOR_DARK if idx == len(self.combat_logs)-1 else TEXT_COLOR_MUTED
            # Si c'est le dernier log et que c'est un Touché, on met en rouge
            if idx == len(self.combat_logs)-1 and "TOUCHÉ" in log:
                text_col = (255, 82, 82)
            
            # Encadré pour le dernier message
            if idx == len(self.combat_logs)-1:
                draw.rounded_rectangle([740, log_y_start - 18, 1180, log_y_start + 18], radius=6, fill=(235, 239, 255))
            
            draw.text((960, log_y_start), log, fill=text_col, font=self.font_log, anchor="mm")
            log_y_start += 45

        # Légende de la Heatmap
        draw.text((960, 915), "Probabilité de présence d'un navire ennemi :", fill=TEXT_COLOR_MUTED, font=self.font_text, anchor="mm")
        # Barre de gradient
        grad_x1, grad_y1, grad_x2, grad_y2 = 810, 935, 1110, 947
        draw.rounded_rectangle([grad_x1, grad_y1, grad_x2, grad_y2], radius=4, fill=None, outline=(210, 215, 235), width=1)
        for gx in range(grad_x2 - grad_x1):
            g_prob = gx / float(grad_x2 - grad_x1)
            g_col = get_heatmap_color(g_prob)
            draw.line([grad_x1 + gx, grad_y1, grad_x1 + gx, grad_y2], fill=g_col, width=1)
        draw.text((800, 941), "Min (0%)", fill=TEXT_COLOR_MUTED, font=self.font_log, anchor="rm")
        draw.text((1120, 941), "Max (100%)", fill=TEXT_COLOR_MUTED, font=self.font_log, anchor="lm")

        # --- RENDU DE L'EFFET ANIMÉ DE TIR/IMPACT ---
        if highlight_state is not None and target_cell is not None:
            # IA active tire sur la grille adverse, ce qui correspond à sa propre grille de tir à l'écran
            grid_draw_x = grid1_x if active_player == 1 else grid2_x
            grid_draw_y = grid1_y if active_player == 1 else grid2_y
            
            y_t, x_t = target_cell
            cell_center_x = grid_draw_x + x_t * (grid_cell_size + grid_gap) + grid_cell_size / 2
            cell_center_y = grid_draw_y + y_t * (grid_cell_size + grid_gap) + grid_cell_size / 2
            
            overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            
            if highlight_state == "targeting":
                # Dessiner un réticule de ciblage animé
                r_ret = effect_radius
                # Bounding box
                draw_overlay.rectangle(
                    [cell_center_x - grid_cell_size/2, cell_center_y - grid_cell_size/2,
                     cell_center_x + grid_cell_size/2, cell_center_y + grid_cell_size/2],
                    outline=(255, 200, 0, 255), width=3
                )
                # Réticule central
                draw_overlay.ellipse([cell_center_x - r_ret, cell_center_y - r_ret, cell_center_x + r_ret, cell_center_y + r_ret],
                                     outline=(255, 82, 82, 200), width=2)
                draw_overlay.line([cell_center_x - r_ret - 5, cell_center_y, cell_center_x + r_ret + 5, cell_center_y], fill=(255, 82, 82, 255), width=2)
                draw_overlay.line([cell_center_x, cell_center_y - r_ret - 5, cell_center_x, cell_center_y + r_ret + 5], fill=(255, 82, 82, 255), width=2)
                
            elif highlight_state == "hit_impact":
                # Onde de choc d'explosion (cercles concentriques rouges/oranges)
                draw_overlay.ellipse(
                    [cell_center_x - effect_radius, cell_center_y - effect_radius,
                     cell_center_x + effect_radius, cell_center_y + effect_radius],
                    fill=(255, 200, 50, int(180 * (1 - effect_radius/35)))
                )
                draw_overlay.ellipse(
                    [cell_center_x - effect_radius*0.7, cell_center_y - effect_radius*0.7,
                     cell_center_x + effect_radius*0.7, cell_center_y + effect_radius*0.7],
                    fill=(255, 100, 30, int(220 * (1 - (effect_radius*0.7)/35)))
                )
                draw_overlay.ellipse(
                    [cell_center_x - effect_radius*0.4, cell_center_y - effect_radius*0.4,
                     cell_center_x + effect_radius*0.4, cell_center_y + effect_radius*0.4],
                    fill=(255, 50, 50, int(255 * (1 - (effect_radius*0.4)/35)))
                )
                
            elif highlight_state == "miss_impact":
                # Onde de choc d'eau (cercle blanc/bleuté)
                draw_overlay.ellipse(
                    [cell_center_x - effect_radius, cell_center_y - effect_radius,
                     cell_center_x + effect_radius, cell_center_y + effect_radius],
                    fill=(255, 255, 255, int(150 * (1 - effect_radius/25))),
                    outline=(200, 220, 255, int(200 * (1 - effect_radius/25))),
                    width=2
                )
                
            img = Image.alpha_composite(img, overlay)
            
        return img.convert("RGB")

    def generate_victory_frame(self, winner_name):
        """Génère l'écran de victoire final."""
        img = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)
        
        # En-tête
        draw.text((WIDTH//2, HEIGHT//2 - 180), "FIN DE LA PARTIE", fill=TEXT_COLOR_DARK, font=self.font_title, anchor="mm")
        
        # Victoire
        draw.text((WIDTH//2, HEIGHT//2 - 40), f"VICTOIRE DE {winner_name} !", fill=(255, 82, 82) if winner_name == "IA 1" else BASE_BLUE, 
                  font=get_font(60, bold=True), anchor="mm")
        
        # Sous-titre explicatif
        draw.text((WIDTH//2, HEIGHT//2 + 50), f"Tous les navires de l'adversaire ont été coulés.", fill=TEXT_COLOR_DARK, font=self.font_subtitle, anchor="mm")
        
        # Tableau récapitulatif
        rect_x1, rect_y1, rect_x2, rect_y2 = WIDTH//2 - 350, HEIGHT//2 + 100, WIDTH//2 + 350, HEIGHT//2 + 320
        self.draw_panel_card(draw, rect_x1, rect_y1, rect_x2, rect_y2, CENTER_PANEL_BG, (210, 215, 235), radius=10)
        
        draw.text((rect_x1 + 100, rect_y1 + 50), "Statistiques", fill=TEXT_COLOR_DARK, font=self.font_header, anchor="lm")
        
        draw.text((rect_x1 + 100, rect_y1 + 110), f"Tirs IA 1 : {self.player1.shots_count}", fill=TEXT_COLOR_MUTED, font=self.font_text, anchor="lm")
        draw.text((rect_x1 + 100, rect_y1 + 150), f"Précision IA 1 : {self.player1.get_accuracy()}%", fill=TEXT_COLOR_MUTED, font=self.font_text, anchor="lm")
        
        draw.text((rect_x1 + 450, rect_y1 + 110), f"Tirs IA 2 : {self.player2.shots_count}", fill=TEXT_COLOR_MUTED, font=self.font_text, anchor="lm")
        draw.text((rect_x1 + 450, rect_y1 + 150), f"Précision IA 2 : {self.player2.get_accuracy()}%", fill=TEXT_COLOR_MUTED, font=self.font_text, anchor="lm")
        
        return img.convert("RGB")

    def build_video(self):
        """Boucle principale du jeu qui simule la partie et écrit la vidéo."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.output_path, fourcc, FPS, (WIDTH, HEIGHT))
        
        print("Début de la simulation et de l'enregistrement de la vidéo...")
        
        # --- Introduction (3 secondes / 72 frames statiques du début) ---
        initial_frame = self.generate_image_frame(active_player=1)
        frame_np = np.array(initial_frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        for _ in range(72):
            video.write(frame_bgr)
            
        active_player = 1
        
        # Boucle de jeu
        while self.player1.get_remaining_ship_cells() > 0 and self.player2.get_remaining_ship_cells() > 0:
            attacker = self.player1 if active_player == 1 else self.player2
            defender = self.player2 if active_player == 1 else self.player1
            
            # 1. Calcul de l'inférence et sélection de la cible
            heatmap, ty, tx = self.run_inference(attacker)
            
            # Traduction des coordonnées en format standard (ex: D4)
            col_letter = chr(ord('A') + tx)
            row_num = ty + 1
            coord_str = f"{col_letter}{row_num}"
            
            # --- Animation de ciblage (6 frames) ---
            # Le réticule rétrécit sur la cible
            for f in range(6):
                r_ret = 30 - f * 4  # Rayon de 30 à 6
                img_frame = self.generate_image_frame(
                    active_player=active_player,
                    target_cell=(ty, tx),
                    highlight_state="targeting",
                    effect_radius=r_ret
                )
                f_np = np.array(img_frame)
                f_bgr = cv2.cvtColor(f_np, cv2.COLOR_RGB2BGR)
                video.write(f_bgr)
                
            # 2. Résolution du tir
            is_hit = defender.ships_board.grille[ty][tx] == 1
            if is_hit:
                # C'est un touché !
                attacker.hiden_grille[ty][tx] = 1
                defender.ships_board.grille[ty][tx] = 0 # navire endommagé
                attacker.hits_count += 1
                result_str = "TOUCHÉ"
                log_msg = f"{attacker.name} tire en {coord_str} -> TOUCHÉ !"
                
                # --- Animation d'impact explosion (8 frames) ---
                for f in range(8):
                    r_exp = 5 + f * 3.5 # Rayon grandit de 5 à 30
                    img_frame = self.generate_image_frame(
                        active_player=active_player,
                        target_cell=(ty, tx),
                        highlight_state="hit_impact",
                        effect_radius=r_exp
                    )
                    f_np = np.array(img_frame)
                    f_bgr = cv2.cvtColor(f_np, cv2.COLOR_RGB2BGR)
                    video.write(f_bgr)
            else:
                # C'est raté !
                attacker.hiden_grille[ty][tx] = -1
                attacker.misses_count += 1
                result_str = "RATÉ"
                log_msg = f"{attacker.name} tire en {coord_str} -> Raté"
                
                # --- Animation de splash d'eau (6 frames) ---
                for f in range(6):
                    r_splash = 3 + f * 3 # Rayon de 3 à 18
                    img_frame = self.generate_image_frame(
                        active_player=active_player,
                        target_cell=(ty, tx),
                        highlight_state="miss_impact",
                        effect_radius=r_splash
                    )
                    f_np = np.array(img_frame)
                    f_bgr = cv2.cvtColor(f_np, cv2.COLOR_RGB2BGR)
                    video.write(f_bgr)
                    
            attacker.shots_count += 1
            self.add_log(log_msg)
            print(f"Tour {self.tour_number}: {attacker.name} tire en ({ty}, {tx}) -> {result_str}")
            
            # --- Frame statique du résultat avant transition (4 frames) ---
            result_frame = self.generate_image_frame(active_player=active_player)
            rf_np = np.array(result_frame)
            rf_bgr = cv2.cvtColor(rf_np, cv2.COLOR_RGB2BGR)
            for _ in range(4):
                video.write(rf_bgr)
                
            # Alternance des joueurs
            if active_player == 1:
                active_player = 2
            else:
                active_player = 1
                self.tour_number += 1
                
            # Limite de sécurité pour éviter des vidéos trop longues ou boucles infinies
            if self.tour_number > 100:
                self.add_log("Match nul - limite de tours atteinte !")
                break

        # Déterminer le vainqueur
        if self.player1.get_remaining_ship_cells() == 0:
            winner = "IA 2"
        elif self.player2.get_remaining_ship_cells() == 0:
            winner = "IA 1"
        else:
            winner = "Aucun (Match nul)"
            
        print(f"Fin de partie ! Vainqueur : {winner}")
        self.add_log(f"Victoire de {winner} !")

        # --- Fin de la vidéo : Écran de victoire (5 secondes / 120 frames statiques) ---
        victory_frame = self.generate_victory_frame(winner)
        vf_np = np.array(victory_frame)
        vf_bgr = cv2.cvtColor(vf_np, cv2.COLOR_RGB2BGR)
        for _ in range(120):
            video.write(vf_bgr)
            
        video.release()
        print(f"Vidéo enregistrée avec succès sous : {self.output_path}")

# ============================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "mlp_bataille_navale.pth")
    output_path = os.path.join(base_dir, "demo_bataille_navale.mp4")
    
    # Créer le dossier parent de output_path s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    generator = BattleshipDemoGenerator(model_path, output_path)
    generator.build_video()
    
    # Copier la vidéo dans le dossier d'artefacts pour partage
    artifact_dir = r"C:\Users\prozart\.gemini\antigravity\brain\b74a4f8f-1da4-4366-b0aa-216ab1958494"
    if os.path.exists(artifact_dir):
        import shutil
        dest = os.path.join(artifact_dir, "demo_bataille_navale.mp4")
        shutil.copy2(output_path, dest)
        print(f"Vidéo copiée dans les artefacts : {dest}")
