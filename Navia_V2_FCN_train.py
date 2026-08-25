import torch
import torch.nn as nn
import train
import random
class FCN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # 3 canaux → 32
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # 32 → 64
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # 64 → 32
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # 32 → 1
            nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=1
            )
        )

    def forward(self, x):
        return self.network(x)

models = FCN() # <-- Instanciation du modèle
optimizer= torch.optim.Adam(models.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss() # initialisation de la fonction de perte
plataux_ia = train.Grille()
nb_tiress = []
losss = []

fenetre = 50
historique_loss = []  # à remplir avec loss.item() de chaque partie
meilleure_loss = float('inf')
for empoque in range(1000000):
    print(f"Entraînement de l'époque {empoque+1}/1000000")
# ====================================  prépration des DATA ============================
    plataux_ia.gen_grille()
    hiden_grille_ia_hit = [[0 for _ in range(10)] for _ in range(10)]
    hiden_grille_ia_miss = [[0 for _ in range(10)] for _ in range(10)]
    hiden_grille_ia_not_explored = [[1 for _ in range(10)] for _ in range(10)]
    nb_bateaux = 17



    target = torch.tensor(
        plataux_ia.grille,
        dtype=torch.float32
    ).unsqueeze(0)
    losss_epoque=[]
    nb_tires = 0
    # ==================================== / prépration des DATA ============================
    while nb_bateaux > 0:

        hidden_grille = torch.tensor([
        hiden_grille_ia_hit,
        hiden_grille_ia_miss,
        hiden_grille_ia_not_explored
        ], dtype=torch.float32)

        hidden_grille = hidden_grille.unsqueeze(0) 

        logits = models(hidden_grille)
        logits = logits.squeeze(1) # <-- suprition du batch
        probas_tensor = torch.sigmoid(logits)

        loss = loss_fn(logits, target) # calcul de la perte

        # =========================== Back prog =========================
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        # ========================== / Back prog =========================
        probas_list = probas_tensor.squeeze(0).tolist() # <-- conversion en liste
        probas_tensor = probas_tensor.squeeze(0)
        masque = torch.tensor(hiden_grille_ia_not_explored, dtype=torch.float32)
        probas_masquees = probas_tensor.clone()
        probas_masquees[masque == 0] = float('-inf')
        cases_restantes = int(masque.sum().item())
        k = min(5, cases_restantes)

        valeurs, indices = torch.topk(
            probas_masquees.flatten(),
            k=k
        )
        case_idx = random.randrange(k)
        indice = indices[case_idx].item()
        y = indice // 10
        x = indice % 10

        #print(f"IA choisit la case ({y}, {x})")
        erreur = False
        hit = False
        if hiden_grille_ia_not_explored[y][x] == 0:
            #print (f"[X] L'IA a déjà tiré sur cette case")
            
            """hiden_grille_ia[y][x] = -5
            cose_grille_ia[y][x] = -5"""
            erreur = True
            
        if plataux_ia.grille[y][x] == 1:
            if erreur == False:
                hit = True
                #print (f"[✓] L'IA a touché un bateaux")
                hiden_grille_ia_hit[y][x] = 1
                hiden_grille_ia_not_explored[y][x] = 0
                nb_bateaux = nb_bateaux - 1

        elif plataux_ia.grille[y][x] == 0:
            if erreur == False:
                #print (f"[X] L'IA a raté son tir")
                hiden_grille_ia_miss[y][x] = 1
                hiden_grille_ia_not_explored[y][x] = 0

        nb_tires = nb_tires + 1
        losss_epoque.append(loss.item())
        moyenne_losss_epoque = sum(losss_epoque) / len(losss_epoque)

        nb_tiress.append(nb_tires)
        moyenne_nb_tiress = sum(nb_tiress) / len(nb_tiress)
    losss.append(moyenne_losss_epoque)
    if moyenne_losss_epoque < meilleure_loss:
        meilleure_loss = moyenne_losss_epoque
        torch.save(models.state_dict(), "Navia_V2_FCN.pt")
        print(f"✓ Nouveau meilleur modèle sauvegardé ! Loss : {meilleure_loss:.6f}")
    moyenne_losss_global = sum(losss) / len(losss)

    historique_loss.append(moyenne_losss_epoque)
    if len(historique_loss) > fenetre:
        historique_loss.pop(0)
    moyenne_glissante = sum(historique_loss) / len(historique_loss)

    print(f"\nLoss moyenne de la partie : {moyenne_losss_epoque} | Nombre de tirs : {nb_tires} | Nombre de tirs moyen : {moyenne_nb_tiress} | Loss moyenne global : {moyenne_losss_global} | Loss moyenne glissante sur {fenetre} parties : {moyenne_glissante}\n")
    print("================================")



