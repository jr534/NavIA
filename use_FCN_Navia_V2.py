

import torch
import torch.nn as nn
import train


# ============================== Modèle FCN ==============================

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
        )

    def forward(self, x):
        return self.network(x)  # (batch, 1, 10, 10)


def charger_modele(path="Navia_V2_FCN.pt"):
    """Essaie de charger le FCN. Renvoie None si indisponible (fallback aléatoire)."""
    try:
        model = FCN()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        print(f"[i] Modèle '{path}' chargé, l'IA joue avec NavIA V2.")
        return model
    except Exception as e:
        print(f"[!] Impossible de charger le modèle ({e}). L'IA joue en mode aléatoire.")
        return None
print(charger_modele(path="Navia_V2_FCN.pt").shape())

def construire_tenseur_entree(hit, miss, not_explored):
    """Construit le tenseur (1, 3, 10, 10) attendu par le FCN à partir
    des 3 grilles visibles par l'IA : touché / raté / non exploré."""
    canaux = [hit, miss, not_explored]
    t = torch.tensor(canaux, dtype=torch.float32).unsqueeze(0)  # (1, 3, 10, 10)
    return t


def choisir_case_ia(model, hit, miss, not_explored):
    """Renvoie (y, x) choisi par l'IA parmi les cases non explorées."""
    import random
    cases_dispo = [(y, x) for y in range(10) for x in range(10) if not_explored[y][x] == 1]

    if model is None:
        return random.choice(cases_dispo)

    with torch.no_grad():
        x_in = construire_tenseur_entree(hit, miss, not_explored)
        logits = model(x_in)                     # (1, 1, 10, 10)
        probas = torch.sigmoid(logits).squeeze()  # (10, 10)

        # On masque les cases déjà explorées pour ne jamais les rechoisir
        for y in range(10):
            for x in range(10):
                if not_explored[y][x] == 0:
                    probas[y][x] = -float("inf")

        idx = torch.argmax(probas).item()
        y, x = idx // 10, idx % 10
        return y, x


# ============================== Jeu ==============================

def demander_coordonnees_joueur(hiden_grille_joueur):
    while True:
        user_input = input("Vos coordonnées (ligne,colonne — ex: 3,7) : ").strip()
        try:
            y_str, x_str = user_input.split(",")
            y, x = int(y_str) - 1, int(x_str) - 1
        except ValueError:
            print("[X] Format invalide, utilisez : ligne,colonne (ex: 3,7)")
            continue

        if not (0 <= y <= 9 and 0 <= x <= 9):
            print("[X] Coordonnées hors grille (1 à 10).")
            continue

        if hiden_grille_joueur[y][x] != 0:
            print("[X] Vous avez déjà tiré ici, choisissez une autre case.")
            continue

        return y, x


def compter_bateaux_restants(grille):
    return sum(sum(ligne) for ligne in grille)


def jouer():
    print("=" * 50)
    print(" BATAILLE NAVALE — Joueur vs IA (NavIA)")
    print("=" * 50)

    # Plateau du joueur (où l'IA tire) et plateau de l'IA (où le joueur tire)
    plateau_joueur = train.Grille()
    plateau_joueur.gen_grille()

    plateau_ia = train.Grille()
    plateau_ia.gen_grille()

    # Ce que voit le joueur sur la grille de l'IA
    hiden_grille_joueur = [[0 for _ in range(10)] for _ in range(10)]

    # Ce que voit l'IA sur la grille du joueur : 3 canaux séparés (format attendu par le FCN)
    ia_hit = [[0 for _ in range(10)] for _ in range(10)]
    ia_miss = [[0 for _ in range(10)] for _ in range(10)]
    ia_not_explored = [[1 for _ in range(10)] for _ in range(10)]

    modele = charger_modele()

    tour = 1
    while compter_bateaux_restants(plateau_joueur.grille) > 0 and compter_bateaux_restants(plateau_ia.grille) > 0:
        print(f"\n----- Tour {tour} -----")

        # --- Tour du joueur ---
        print("\n[i] Votre tour — grille de l'IA :")
        plateau_joueur.print_grille(hiden_grille_joueur)
        y, x = demander_coordonnees_joueur(hiden_grille_joueur)

        if plateau_ia.grille[y][x] == 1:
            print("[✓] Touché !")
            hiden_grille_joueur[y][x] = 1
            plateau_ia.grille[y][x] = 0
        else:
            print("[X] Raté.")
            hiden_grille_joueur[y][x] = -1

        if compter_bateaux_restants(plateau_ia.grille) == 0:
            break

        # --- Tour de l'IA ---
        print("\n[i] Tour de l'IA...")
        y, x = choisir_case_ia(modele, ia_hit, ia_miss, ia_not_explored)
        ia_not_explored[y][x] = 0

        if plateau_joueur.grille[y][x] == 1:
            print(f"[✓] L'IA a touché en ({y + 1},{x + 1}) !")
            ia_hit[y][x] = 1
            plateau_joueur.grille[y][x] = 0
        else:
            print(f"[X] L'IA a raté en ({y + 1},{x + 1}).")
            ia_miss[y][x] = 1

        tour += 1

    print("\n" + "=" * 50)
    if compter_bateaux_restants(plateau_ia.grille) == 0:
        print(" 🎉 Vous avez gagné !")
    else:
        print(" 🤖 L'IA a gagné.")
    print("=" * 50)


if __name__ == "__main__":
    jouer()