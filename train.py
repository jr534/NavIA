import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam

class Grille:
    def __init__(self):
            self.grille = [[0 for _ in range(10)] for _ in range(10)]
    def print_grille(self, grille):
        header = "".join(str(i).rjust(2) for i in range(1, 11))
        print(f"   {header}")
        
        for i, ligne in enumerate(grille):
            print(f"  {'-+'*10}")
            cells = "".join(str(cell).rjust(2) for cell in ligne)
            num_ligne = str(i+1).rjust(2)
            print(f"{num_ligne} {cells}")
    def gen_pose (self):
        postion_Y = random.randint(0, 9)
        postion_X = random.randint(0, 9)
        num_orientation = random.randint(1, 4) 
        match num_orientation :
            case 1:
                str_orientation = "Down"
            case 2:
                str_orientation = "Right"
            case 3:
                str_orientation = "Top"
            case 4:
                str_orientation = "Left"

        return postion_X,postion_Y,str_orientation

    def pose_est_valide (self, len_bataux, postion_X, postion_Y, str_orientation):
    
        posable = False
        # Check haut / bas
        if str_orientation == "Top":
            if postion_Y - len_bataux +1 >= 0: 
                for i in range(len_bataux):
                    if self.grille[postion_Y-i][postion_X] == 1:
                        posable = False
                        break
                    else:
                        posable = True

        if str_orientation == "Down":
            if postion_Y + len_bataux <= 10: 
                for i in range(len_bataux):
                    if self.grille[postion_Y+i][postion_X] == 1:
                        posable = False
                        break
                    else:
                        posable = True

        # Check Droite / Gauche
        if str_orientation == "Right":
            if postion_X + len_bataux <= 10: 
                for i in range(len_bataux):
                    if self.grille[postion_Y][postion_X+i] == 1:
                        posable = False
                        break
                    else:
                        posable = True

        if str_orientation == "Left":
            if postion_X - len_bataux >= 0: 
                for i in range(len_bataux):
                    if self.grille[postion_Y][postion_X-i] == 1:
                        posable = False
                        break
                    else:
                        posable = True
                
        """    if posable:
                print (f"[✓] Le bataux en X={postion_X} et en Y={postion_Y} de longeur={len_bataux}  avec une orientation ver {str_orientation} est POSABLE sur la self.grille de 10*10")
            else:
                print (f"[X] Le bataux en X={postion_X} et en Y={postion_Y} de longeur={len_bataux} avec une orientation ver {str_orientation} est NON POSABLE sur la self.grille de 10*10")
        """

        return posable

    def placée_bataux (self, len_bataux):
        postion_X,postion_Y,str_orientation = self.gen_pose()

        while not self.pose_est_valide(len_bataux,postion_X,postion_Y,str_orientation):
            postion_X,postion_Y,str_orientation = self.gen_pose()

        if str_orientation == "Down":
            for i in range(len_bataux):
                self.grille[postion_Y+i][postion_X] = 1
        if str_orientation == "Top":
            for i in range(len_bataux):
                self.grille[postion_Y-i][postion_X] = 1
        
        if str_orientation == "Left":
            for i in range(len_bataux):
                self.grille[postion_Y][postion_X-i] = 1
        if str_orientation == "Right":
            for i in range(len_bataux):
                self.grille[postion_Y][postion_X+i] = 1
    def gen_grille (self):
        self.grille = [[0 for _ in range(10)] for _ in range(10)]
        for i in range(5):
            match i:
                case 0:
                    len_bataux = 5
                    type_bataux = "Porte-avions"
                case 1:
                    len_bataux = 4
                    type_bataux = "Cuirassé"
                case 2:
                    len_bataux = 3
                    type_bataux = "Contre-torpilleur"
                case 3:
                    len_bataux = 3
                    type_bataux = "Sous-marin"
                case 4:
                    len_bataux = 2
                    type_bataux = "Torpilleur"

                
            print (f"Placement du {type_bataux} sur la self.grille ")
            self.placée_bataux(len_bataux)
                        
            #self.print_grille(self.grille)

class BattleshipMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 100)
        )
    
    def forward(self, x):
        logits = self.fc(x)
        probas = torch.sigmoid(logits)
        return probas, logits
 
# ============================================================
if __name__ == "__main__":
    grille = Grille()
    model = BattleshipMLP()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    Nb_partie = 500

    for i in range(Nb_partie):
        print(f" le model a fini la partie {i+1} en nb de tir : {i+1}")
        hiden_grille = [[0 for _ in range(10)] for _ in range(10)]
        grille.gen_grille()
        nb_tir = 0
        game_loss_total = 0
        while sum(sum(ligne) for ligne in grille.grille) > 0:      

            case_cible = ""
            cose_grille = [[0 for _ in range(10)] for _ in range(10)]

            hidden_grille_flat = torch.tensor(
                [cell for ligne in hiden_grille for cell in ligne], 
                dtype=torch.float32
            )

            
            format_hiden_grille = hidden_grille_flat.shape
            
            probas, logits = model(hidden_grille_flat)

            probas_masked = probas.clone()

            probas_masked = probas.clone()
            logits_masked = logits.clone()
            masque = (hidden_grille_flat != 0)

            logits_masked = logits.clone()

            logits_masked[masque] = -1e9

            probas_masked = torch.sigmoid(logits_masked)

            epsilon = max(0.05, 1.0 - i/Nb_partie) # 100% au début 5% à la fin
            if random.random() < epsilon:
                libres = [idx for y in range(10) for x in range(10) if hiden_grille[y][x]==0 for idx in [y*10+x]]
                case_idx = random.choice(libres)
            else:
                case_idx = probas_masked.argmax().item()        
            y = case_idx // 10
            x = case_idx % 10

            case_cible = f"{y},{x}"


            """      while len(case_cible) != 3:
                        case_cible  = input("sur quelle case voulez vous tirer ? (aux formats: Y,X) : ") 
                        if len(case_cible) != 3:
                            print ("[X] Le format de la case est incorrecte, veuillez entrer un format correct (ex: 1,2)")
                """
            # ============================================ Ferification ==========================
            hit = False
            erreur = False

            if hiden_grille[y][x] != 0:
                erreur = True
                print (f"{i+1}/{Nb_partie} [X] Vous avez déjà tiré sur cette case, veuillez en choisir une autre")
                hiden_grille[y][x] = -5
                cose_grille[y][x] = -5
                
            if grille.grille[y][x] == 1:
                hit = True
                print (f"{i+1}/{Nb_partie} [✓] Vous avez touché un bateaux")
                hiden_grille[y][x] = 5
                cose_grille[y][x] = 5
                grille.grille[y][x] = 0

            elif grille.grille[y][x] == 0:
                if not erreur:
                    print (f"{i+1}/{Nb_partie} [X] Vous avez raté votre tir")
                    hiden_grille[y][x] = 1
                    cose_grille[y][x] = 1


            grille.print_grille(hiden_grille)

            """target = torch.zeros(100)
            target[case_idx] = 1.0 if hit else 0.0
            loss = criterion(logits, target)"""

            target = torch.zeros(100)
            target[case_idx] = 1.0 if (grille.grille[y][x] == 1 or hit) else 0.0
            loss = criterion(logits[case_idx].unsqueeze(0), target[case_idx].unsqueeze(0))

            print (f"Le Modelle a un loss de {loss.item()} sur le tir {nb_tir+1}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            game_loss_total += loss.item()
            nb_tir += 1
            
        print(f"Partie {i+1}/{Nb_partie} - Tirs: {nb_tir} - Loss: {game_loss_total:.4f}")
    
    print(f"\nEntraînement terminé ! Le modèle a joué {Nb_partie} parties.")
    torch.save(model.state_dict(), "models/mlp_bataille_navale.pth")
