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

# Dimensions de la vidéo (Format LinkedIn Carré)
WIDTH = 1080
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
    # S varie de 0.20 à 0.95
    s = 0.20 + 0.75 * prob
    # L varie de 0.93 à 0.40
    l = 0.93 - 0.53 * prob
    r, g, b = colorsys.hls_to_rgb(BASE_HUE, l, s)
    return int(r * 255), int(g * 255), int(b * 255), 255

def get_blue_variant(s_factor, l_factor):
    r, g, b = colorsys.hls_to_rgb(BASE_HUE, l_factor, s_factor)
    return int(r * 255), int(g * 255), int(b * 255)

# --- Classes de jeu ---
class GamePlayer:
    def __init__(self, name):
        self.name = name
        self.ships_board = train.Grille()
        self.ships_board.gen_grille()
        
        self.initial_ships = [[self.ships_board.grille[y][x] for x in range(10)] for y in range(10)]
        self.hiden_grille = [[0 for _ in range(10)] for _ in range(10)]
        
        self.shots_count = 0
        self.hits_count = 0
        self.misses_count = 0

    def get_remaining_ship_cells(self):
        return sum(sum(ligne) for ligne in self.ships_board.grille)

    def get_accuracy(self):
        if self.shots_count == 0:
            return 0
        return int((self.hits_count / self.shots_count) * 100)

class LinkedInDemoGenerator:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = train.BattleshipMLP()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Initialisation des polices adaptées à la taille 1080x1080
        self.font_title = get_font(28, bold=True)
        self.font_subtitle = get_font(16)
        self.font_header = get_font(18, bold=True)
        self.font_stats = get_font(14, bold=True)
        self.font_text = get_font(13)
        self.font_log = get_font(13)
        
        self.player1 = GamePlayer("IA 1")
        self.player2 = GamePlayer("IA 2")
        
        self.combat_logs = ["Début du combat IA vs IA !"]
        self.tour_number = 1

    def run_inference(self, player):
        hidden_flat = torch.tensor(
            [cell for ligne in player.hiden_grille for cell in ligne],
            dtype=torch.float32,
            device=self.device
        )
        with torch.no_grad():
            probas, logits = self.model(hidden_flat)
            probas = torch.sigmoid(logits)
            
            probas_masked = probas.clone()
            for y in range(10):
                for x in range(10):
                    if player.hiden_grille[y][x] != 0:
                        probas_masked[y * 10 + x] = -float("inf")
            
            case_idx = probas_masked.argmax().item()
            return probas.cpu().tolist(), case_idx // 10, case_idx % 10

    def add_log(self, text):
        self.combat_logs.append(text)
        if len(self.combat_logs) > 4:
            self.combat_logs.pop(0)

    def draw_panel_card(self, draw, x1, y1, x2, y2, fill_color, border_color, radius=10):
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill_color)
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=None, outline=border_color, width=2)

    def draw_grid(self, draw, start_x, start_y, cell_size, gap, hiden_grille, heatmap, target_cell=None, highlight=False):
        min_p = min(heatmap)
        max_p = max(heatmap)
        diff_p = max_p - min_p if (max_p - min_p) > 1e-6 else 1.0
        
        for y in range(10):
            for x in range(10):
                val = hiden_grille[y][x]
                cell_x = start_x + x * (cell_size + gap)
                cell_y = start_y + y * (cell_size + gap)
                
                prob = (heatmap[y * 10 + x] - min_p) / diff_p
                cell_color = get_heatmap_color(prob)
                
                draw.rounded_rectangle([cell_x, cell_y, cell_x + cell_size, cell_y + cell_size], radius=4, fill=cell_color)
                
                if val == -1: # Miss
                    bg_miss = get_blue_variant(0.05, 0.90)
                    draw.rounded_rectangle([cell_x, cell_y, cell_x + cell_size, cell_y + cell_size], radius=4, fill=bg_miss)
                    cx, cy = cell_x + cell_size/2, cell_y + cell_size/2
                    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=BASE_BLUE)
                elif val == 1: # Hit
                    bg_hit = get_blue_variant(0.85, 0.35)
                    draw.rounded_rectangle([cell_x, cell_y, cell_x + cell_size, cell_y + cell_size], radius=4, fill=bg_hit)
                    cx, cy = cell_x + cell_size/2, cell_y + cell_size/2
                    offset = 6
                    draw.line([cx - offset, cy - offset, cx + offset, cy + offset], fill=(255, 82, 82), width=2)
                    draw.line([cx + offset, cy - offset, cx - offset, cy + offset], fill=(255, 82, 82), width=2)

                if highlight and target_cell == (y, x):
                    draw.rounded_rectangle(
                        [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                        radius=4, fill=None, outline=(255, 200, 0), width=2
                    )

    def draw_defense_grid(self, draw, start_x, start_y, cell_size, gap, ships_board, initial_ships):
        for y in range(10):
            for x in range(10):
                cell_x = start_x + x * (cell_size + gap)
                cell_y = start_y + y * (cell_size + gap)
                
                has_ship = initial_ships[y][x] == 1
                is_hit = has_ship and ships_board.grille[y][x] == 0
                
                if is_hit:
                    cell_color = (255, 82, 82)
                elif has_ship:
                    cell_color = BASE_BLUE
                else:
                    cell_color = get_blue_variant(0.10, 0.95)
                
                draw.rounded_rectangle([cell_x, cell_y, cell_x + cell_size, cell_y + cell_size], radius=2, fill=cell_color)

    def generate_image_frame(self, active_player, target_cell=None, highlight_state=None, effect_radius=0):
        img = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)
        
        # --- EN-TÊTE ---
        draw.text((WIDTH//2, 40), "COMBAT BATAILLE NAVALE : IA vs IA", fill=TEXT_COLOR_DARK, font=self.font_title, anchor="mm")
        draw.text((WIDTH//2, 75), f"Modèle MLP de Deep Learning  •  Tour {self.tour_number}", fill=TEXT_COLOR_MUTED, font=self.font_subtitle, anchor="mm")
        
        heatmap_p1, ty_p1, tx_p1 = self.run_inference(self.player1)
        heatmap_p2, ty_p2, tx_p2 = self.run_inference(self.player2)
        
        # --- GRILLE IA 1 (GAUCHE) ---
        card1_x1, card1_y1, card1_x2, card1_y2 = 40, 110, 520, 690
        border_p1 = BASE_BLUE if active_player == 1 else (210, 215, 235)
        self.draw_panel_card(draw, card1_x1, card1_y1, card1_x2, card1_y2, PANEL_BG, border_p1)
        
        draw.text((280, 140), "IA 1 (Grille d'Attaque)", fill=TEXT_COLOR_DARK, font=self.font_header, anchor="mm")
        
        # Grille principale IA 1
        grid1_x, grid1_y = 86, 175
        grid_cell_size = 36
        grid_gap = 3
        self.draw_grid(
            draw, grid1_x, grid1_y, grid_cell_size, grid_gap,
            self.player1.hiden_grille, heatmap_p1,
            target_cell=target_cell if active_player == 1 else None,
            highlight=(highlight_state is not None and active_player == 1)
        )
        
        # Grille de défense IA 1
        draw.text((280, 585), "Ma Grille (Défense)", fill=TEXT_COLOR_DARK, font=self.font_stats, anchor="mm")
        self.draw_defense_grid(draw, 215, 610, 11, 2, self.player1.ships_board, self.player1.initial_ships)

        # --- GRILLE IA 2 (DROITE) ---
        card2_x1, card2_y1, card2_x2, card2_y2 = 560, 110, 1040, 690
        border_p2 = BASE_BLUE if active_player == 2 else (210, 215, 235)
        self.draw_panel_card(draw, card2_x1, card2_y1, card2_x2, card2_y2, PANEL_BG, border_p2)
        
        draw.text((800, 140), "IA 2 (Grille d'Attaque)", fill=TEXT_COLOR_DARK, font=self.font_header, anchor="mm")
        
        # Grille principale IA 2
        grid2_x, grid2_y = 606, 175
        self.draw_grid(
            draw, grid2_x, grid2_y, grid_cell_size, grid_gap,
            self.player2.hiden_grille, heatmap_p2,
            target_cell=target_cell if active_player == 2 else None,
            highlight=(highlight_state is not None and active_player == 2)
        )
        
        # Grille de défense IA 2
        draw.text((800, 585), "Ma Grille (Défense)", fill=TEXT_COLOR_DARK, font=self.font_stats, anchor="mm")
        self.draw_defense_grid(draw, 735, 610, 11, 2, self.player2.ships_board, self.player2.initial_ships)

        # --- INFOS CENTRALES ET BAS DE PAGE ---
        card3_x1, card3_y1, card3_x2, card3_y2 = 40, 715, 1040, 1040
        self.draw_panel_card(draw, card3_x1, card3_y1, card3_x2, card3_y2, CENTER_PANEL_BG, (215, 220, 240))
        
        # Stats IA 1 (Gauche)
        p1_cells = self.player1.get_remaining_ship_cells()
        draw.text((70, 750), "IA 1", fill=TEXT_COLOR_DARK, font=self.font_header, anchor="lm")
        draw.text((490, 750), f"Navires : {p1_cells}/17", fill=TEXT_COLOR_DARK, font=self.font_text, anchor="rm")
        draw.rounded_rectangle([70, 770, 490, 780], radius=3, fill=(220, 225, 240))
        hp1_w = int(420 * (p1_cells / 17))
        if hp1_w > 0:
            draw.rounded_rectangle([70, 770, 70 + hp1_w, 780], radius=3, fill=BASE_BLUE)
        draw.text((70, 800), f"Tirs : {self.player1.shots_count}   Précision : {self.player1.get_accuracy()}%", fill=TEXT_COLOR_MUTED, font=self.font_text, anchor="lm")

        # Stats IA 2 (Droite)
        p2_cells = self.player2.get_remaining_ship_cells()
        draw.text((590, 750), "IA 2", fill=TEXT_COLOR_DARK, font=self.font_header, anchor="lm")
        draw.text((1010, 750), f"Navires : {p2_cells}/17", fill=TEXT_COLOR_DARK, font=self.font_text, anchor="rm")
        draw.rounded_rectangle([590, 770, 1010, 780], radius=3, fill=(220, 225, 240))
        hp2_w = int(420 * (p2_cells / 17))
        if hp2_w > 0:
            draw.rounded_rectangle([590, 770, 590 + hp2_w, 780], radius=3, fill=BASE_BLUE)
        draw.text((590, 800), f"Tirs : {self.player2.shots_count}   Précision : {self.player2.get_accuracy()}%", fill=TEXT_COLOR_MUTED, font=self.font_text, anchor="lm")

        # Fil d'actualité au milieu en bas
        draw.line([70, 825, 1010, 825], fill=(210, 215, 235), width=1)
        draw.text((WIDTH//2, 845), "JOURNAL DE COMBAT", fill=TEXT_COLOR_DARK, font=self.font_stats, anchor="mm")
        
        log_y = 875
        for idx, log in enumerate(self.combat_logs):
            text_col = TEXT_COLOR_DARK if idx == len(self.combat_logs)-1 else TEXT_COLOR_MUTED
            if idx == len(self.combat_logs)-1 and "TOUCHÉ" in log:
                text_col = (255, 82, 82)
            
            if idx == len(self.combat_logs)-1:
                draw.rounded_rectangle([150, log_y - 12, 930, log_y + 12], radius=4, fill=(235, 239, 255))
                
            draw.text((WIDTH//2, log_y), log, fill=text_col, font=self.font_log, anchor="mm")
            log_y += 32

        # Effets animés
        if highlight_state is not None and target_cell is not None:
            grid_draw_x = grid1_x if active_player == 1 else grid2_x
            grid_draw_y = grid1_y if active_player == 1 else grid2_y
            
            y_t, x_t = target_cell
            cell_center_x = grid_draw_x + x_t * (grid_cell_size + grid_gap) + grid_cell_size / 2
            cell_center_y = grid_draw_y + y_t * (grid_cell_size + grid_gap) + grid_cell_size / 2
            
            overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            
            if highlight_state == "targeting":
                r_ret = effect_radius
                draw_overlay.rectangle(
                    [cell_center_x - grid_cell_size/2, cell_center_y - grid_cell_size/2,
                     cell_center_x + grid_cell_size/2, cell_center_y + grid_cell_size/2],
                    outline=(255, 200, 0, 255), width=2
                )
                draw_overlay.ellipse([cell_center_x - r_ret, cell_center_y - r_ret, cell_center_x + r_ret, cell_center_y + r_ret],
                                     outline=(255, 82, 82, 200), width=2)
                draw_overlay.line([cell_center_x - r_ret - 3, cell_center_y, cell_center_x + r_ret + 3, cell_center_y], fill=(255, 82, 82, 255), width=2)
                draw_overlay.line([cell_center_x, cell_center_y - r_ret - 3, cell_center_x, cell_center_y + r_ret + 3], fill=(255, 82, 82, 255), width=2)
                
            elif highlight_state == "hit_impact":
                draw_overlay.ellipse([cell_center_x - effect_radius, cell_center_y - effect_radius,
                                     cell_center_x + effect_radius, cell_center_y + effect_radius],
                                    fill=(255, 200, 50, int(180 * (1 - effect_radius/35))))
                draw_overlay.ellipse([cell_center_x - effect_radius*0.7, cell_center_y - effect_radius*0.7,
                                     cell_center_x + effect_radius*0.7, cell_center_y + effect_radius*0.7],
                                    fill=(255, 100, 30, int(220 * (1 - (effect_radius*0.7)/35))))
                draw_overlay.ellipse([cell_center_x - effect_radius*0.4, cell_center_y - effect_radius*0.4,
                                     cell_center_x + effect_radius*0.4, cell_center_y + effect_radius*0.4],
                                    fill=(255, 50, 50, int(255 * (1 - (effect_radius*0.4)/35))))
                
            elif highlight_state == "miss_impact":
                draw_overlay.ellipse([cell_center_x - effect_radius, cell_center_y - effect_radius,
                                     cell_center_x + effect_radius, cell_center_y + effect_radius],
                                    fill=(255, 255, 255, int(150 * (1 - effect_radius/25))),
                                    outline=(200, 220, 255, int(200 * (1 - effect_radius/25))), width=2)
                
            img = Image.alpha_composite(img, overlay)
            
        return img.convert("RGB")

    def generate_victory_frame(self, winner_name):
        img = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)
        
        draw.text((WIDTH//2, HEIGHT//2 - 150), "FIN DE LA PARTIE", fill=TEXT_COLOR_DARK, font=self.font_title, anchor="mm")
        draw.text((WIDTH//2, HEIGHT//2 - 40), f"VICTOIRE DE {winner_name} !", fill=(255, 82, 82) if winner_name == "IA 1" else BASE_BLUE, 
                  font=get_font(40, bold=True), anchor="mm")
        draw.text((WIDTH//2, HEIGHT//2 + 30), "Tous les navires adverses ont été coulés.", fill=TEXT_COLOR_MUTED, font=self.font_subtitle, anchor="mm")
        
        rect_x1, rect_y1, rect_x2, rect_y2 = WIDTH//2 - 250, HEIGHT//2 + 90, WIDTH//2 + 250, HEIGHT//2 + 270
        self.draw_panel_card(draw, rect_x1, rect_y1, rect_x2, rect_y2, CENTER_PANEL_BG, (210, 215, 235), radius=8)
        
        draw.text((WIDTH//2, rect_y1 + 30), "Récapitulatif", fill=TEXT_COLOR_DARK, font=self.font_header, anchor="mm")
        draw.text((rect_x1 + 60, rect_y1 + 80), f"IA 1  -  Tirs : {self.player1.shots_count}  •  Précision : {self.player1.get_accuracy()}%", fill=TEXT_COLOR_MUTED, font=self.font_stats, anchor="lm")
        draw.text((rect_x1 + 60, rect_y1 + 130), f"IA 2  -  Tirs : {self.player2.shots_count}  •  Précision : {self.player2.get_accuracy()}%", fill=TEXT_COLOR_MUTED, font=self.font_stats, anchor="lm")
        
        return img.convert("RGB")

    def build_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.output_path, fourcc, FPS, (WIDTH, HEIGHT))
        
        print("Début de l'enregistrement de la vidéo LinkedIn...")
        
        initial_frame = self.generate_image_frame(active_player=1)
        f_np = np.array(initial_frame)
        f_bgr = cv2.cvtColor(f_np, cv2.COLOR_RGB2BGR)
        for _ in range(72):
            video.write(f_bgr)
            
        active_player = 1
        
        while self.player1.get_remaining_ship_cells() > 0 and self.player2.get_remaining_ship_cells() > 0:
            attacker = self.player1 if active_player == 1 else self.player2
            defender = self.player2 if active_player == 1 else self.player1
            
            heatmap, ty, tx = self.run_inference(attacker)
            col_letter = chr(ord('A') + tx)
            row_num = ty + 1
            coord_str = f"{col_letter}{row_num}"
            
            # Ciblage
            for f in range(6):
                r_ret = 30 - f * 4
                img_frame = self.generate_image_frame(
                    active_player=active_player, target_cell=(ty, tx),
                    highlight_state="targeting", effect_radius=r_ret
                )
                video.write(cv2.cvtColor(np.array(img_frame), cv2.COLOR_RGB2BGR))
                
            # Résolution
            is_hit = defender.ships_board.grille[ty][tx] == 1
            if is_hit:
                attacker.hiden_grille[ty][tx] = 1
                defender.ships_board.grille[ty][tx] = 0
                attacker.hits_count += 1
                log_msg = f"{attacker.name} tire en {coord_str} -> TOUCHÉ !"
                
                for f in range(8):
                    r_exp = 5 + f * 3.5
                    img_frame = self.generate_image_frame(
                        active_player=active_player, target_cell=(ty, tx),
                        highlight_state="hit_impact", effect_radius=r_exp
                    )
                    video.write(cv2.cvtColor(np.array(img_frame), cv2.COLOR_RGB2BGR))
            else:
                attacker.hiden_grille[ty][tx] = -1
                attacker.misses_count += 1
                log_msg = f"{attacker.name} tire en {coord_str} -> Raté"
                
                for f in range(6):
                    r_splash = 3 + f * 3
                    img_frame = self.generate_image_frame(
                        active_player=active_player, target_cell=(ty, tx),
                        highlight_state="miss_impact", effect_radius=r_splash
                    )
                    video.write(cv2.cvtColor(np.array(img_frame), cv2.COLOR_RGB2BGR))
                    
            attacker.shots_count += 1
            self.add_log(log_msg)
            
            result_frame = self.generate_image_frame(active_player=active_player)
            rf_bgr = cv2.cvtColor(np.array(result_frame), cv2.COLOR_RGB2BGR)
            for _ in range(4):
                video.write(rf_bgr)
                
            active_player = 2 if active_player == 1 else 1
            if active_player == 1:
                self.tour_number += 1
                
            if self.tour_number > 100:
                break

        winner = "IA 2" if self.player1.get_remaining_ship_cells() == 0 else "IA 1"
        self.add_log(f"Victoire de {winner} !")

        victory_frame = self.generate_victory_frame(winner)
        vf_bgr = cv2.cvtColor(np.array(victory_frame), cv2.COLOR_RGB2BGR)
        for _ in range(120):
            video.write(vf_bgr)
            
        video.release()
        print(f"Vidéo LinkedIn enregistrée sous : {self.output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "mlp_bataille_navale.pth")
    output_path = os.path.join(base_dir, "demo_bataille_navale_linkedin.mp4")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator = LinkedInDemoGenerator(model_path, output_path)
    generator.build_video()
    
    # Copie dans les artefacts
    artifact_dir = r"C:\Users\prozart\.gemini\antigravity\brain\b74a4f8f-1da4-4366-b0aa-216ab1958494"
    if os.path.exists(artifact_dir):
        import shutil
        dest = os.path.join(artifact_dir, "demo_bataille_navale_linkedin.mp4")
        shutil.copy2(output_path, dest)
        print(f"Vidéo copiée dans les artefacts : {dest}")
