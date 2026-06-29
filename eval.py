import torch
import train

plataux_joueur = train.Grille()
plataux_joueur.gen_grille()
hiden_grille_joueur = [[0 for _ in range(10)] for _ in range(10)]

plataux_ia = train.Grille()
plataux_ia.gen_grille()
hiden_grille_ia = [[0 for _ in range(10)] for _ in range(10)]
models = train.BattleshipMLP()
models.load_state_dict(torch.load('models/mlp_bataille_navale.pth'))
models.eval()
i=0

while sum(sum(ligne) for ligne in plataux_joueur.grille) > 0 or sum(sum(ligne) for ligne in plataux_ia.grille) > 0:      
    i+=1
    if i%2 == 0:
        print(f"\n[i] Tour du joueur")

        user_input = input("Veuillez entrer les coordonnées de votre tir (au format y,x): ")
        y, x = map(int, user_input.split(','))

        y= y-1
        x= x-1

        if hiden_grille_joueur[y][x] != 0:
                print (f"[X] Vous avez déjà tiré sur cette case, veuillez en choisir une autre")
                hiden_grille_joueur[y][x] = -5
                
        if plataux_joueur.grille[y][x] == 1:
            print (f"[✓] Vous avez touché un bateaux")
            hiden_grille_joueur[y][x] = 1
            plataux_joueur.grille[y][x] = 0

        elif plataux_joueur.grille[y][x] == 0:
            print (f"[X] Vous avez raté votre tir")
            hiden_grille_joueur[y][x] = -1

        plataux_joueur.print_grille(hiden_grille_joueur)
    else:
        print(f"\n[i] Tour de l'IA")
        cose_grille_ia = [[0 for _ in range(10)] for _ in range(10)]

        hidden_grille_flat = torch.tensor(
                [cell for ligne in hiden_grille_ia for cell in ligne], 
                dtype=torch.float32
            )

            
        format_hiden_grille = hidden_grille_flat.shape
        with torch.no_grad():  # Pas besoin de gradients en inférence
            probas, logits = models(hidden_grille_flat)
            probas = torch.sigmoid(logits)
            for y in range(10):
                for x in range(10):
                    if hiden_grille_ia[y][x] != 0:
                        probas[y*10+x] = -float("inf")
            case_idx = probas.argmax().item()
            y = case_idx // 10
            x = case_idx % 10



        
        print(f"IA choisit la case ({y}, {x})")

        hit = False
        if hiden_grille_ia[y][x] != 0:
            print (f"[X] L'IA a déjà tiré sur cette case")
            
            hiden_grille_ia[y][x] = -5
            cose_grille_ia[y][x] = -5
            
        if plataux_ia.grille[y][x] == 1:
            hit = True
            print (f"[✓] L'IA a touché un bateaux")
            hiden_grille_ia[y][x] = 1
            cose_grille_ia[y][x] = 1
            plataux_ia.grille[y][x] = 0

        elif plataux_ia.grille[y][x] == 0:
            print (f"[X] L'IA a raté son tir")
            hiden_grille_ia[y][x] = -1
            cose_grille_ia[y][x] = -1

         
if sum(sum(ligne) for ligne in plataux_joueur.grille) == 0:
    print("\n[i] Le joueur a gagné !")
else:
    print("\n[i] L'IA a gagné !")