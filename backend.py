import uuid
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

try:
    import train
except ImportError:
    raise ImportError("Le fichier 'train.py' doit être présent dans le même dossier.")

app = FastAPI(
    title="Battleship AI API with Heatmap",
    description="API de bataille navale affichant la carte de chaleur de l'IA",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement du modèle ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = train.BattleshipMLP()
    model.load_state_dict(torch.load('models/mlp_bataille_navale.pth', map_location=device))
    model.eval()
    print("Modèle IA chargé avec succès.")
except Exception as e:
    print(f"Erreur de chargement du modèle : {e}")
    model = None


class GameSession:
    def __init__(self):
        self.plataux_joueur = train.Grille()
        self.plataux_joueur.gen_grille()
        self.hiden_grille_joueur = [[0 for _ in range(10)] for _ in range(10)]

        self.plataux_ia = train.Grille()
        self.plataux_ia.gen_grille()
        self.hiden_grille_ia = [[0 for _ in range(10)] for _ in range(10)]

        self.game_over = False
        self.winner = None


games: Dict[str, GameSession] = {}


# --- Fonction utilitaire pour générer la Heatmap ---
def get_ia_heatmap(game: GameSession) -> List[List[float]]:
    """Calcule la probabilité de présence d'un navire pour chaque case selon le modèle."""
    if model is None:
        return [[0.0 for _ in range(10)] for _ in range(10)]
    
    hidden_grille_flat = torch.tensor(
        [cell for ligne in game.hiden_grille_ia for cell in ligne], 
        dtype=torch.float32,
        device=device
    )
    
    with torch.no_grad():
        probas, logits = model(hidden_grille_flat)
        probas = torch.sigmoid(logits)
        
        # On force à 0% les cases sur lesquelles l'IA a déjà tiré
        for y in range(10):
            for x in range(10):
                if game.hiden_grille_ia[y][x] != 0:
                    probas[y * 10 + x] = 0.0
                    
        # Retourne la matrice 10x10 vers le CPU
        return probas.cpu().view(10, 10).tolist()


# --- Modèles de données ---
class ShootRequest(BaseModel):
    y: int = Field(..., ge=0, le=9)
    x: int = Field(..., ge=0, le=9)

class TurnResult(BaseModel):
    y: int
    x: int
    result: str

class ShootResponse(BaseModel):
    game_id: str
    player_shot: TurnResult
    ai_shot: Optional[TurnResult] = None
    game_over: bool
    winner: Optional[str] = None
    hiden_grille_joueur: List[List[int]]
    hiden_grille_ia: List[List[int]]
    plataux_ia_grille: List[List[int]]
    ia_heatmap: List[List[float]]  # Ajout de la carte de chaleur dans la réponse


@app.get("/")
def get_index():
    return FileResponse("index.html")


@app.post("/game/start", status_code=status.HTTP_201_CREATED)
def start_game():
    game_id = str(uuid.uuid4())
    game = GameSession()
    games[game_id] = game
    
    return {
        "game_id": game_id,
        "message": "Nouvelle partie créée.",
        "hiden_grille_joueur": game.hiden_grille_joueur,
        "hiden_grille_ia": game.hiden_grille_ia,
        "plataux_ia_grille": game.plataux_ia.grille,
        "ia_heatmap": get_ia_heatmap(game)
    }


@app.post("/game/{game_id}/shoot", response_model=ShootResponse)
def player_shoot(game_id: str, request: ShootRequest):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Partie non trouvée.")
    
    game = games[game_id]
    if game.game_over:
        raise HTTPException(status_code=400, detail="La partie est terminée.")

    y, x = request.y, request.x

    # --- 1. TOUR DU JOUEUR ---
    player_outcome = ""
    if game.hiden_grille_joueur[y][x] != 0:
        game.hiden_grille_joueur[y][x] = -5
        player_outcome = "déjà tiré"
    elif game.plataux_joueur.grille[y][x] == 1:
        player_outcome = "touché"
        game.hiden_grille_joueur[y][x] = 1
        game.plataux_joueur.grille[y][x] = 0
    else:
        player_outcome = "raté"
        game.hiden_grille_joueur[y][x] = -1

    player_shot_result = TurnResult(y=y, x=x, result=player_outcome)

    if sum(sum(ligne) for ligne in game.plataux_joueur.grille) == 0:
        game.game_over = True
        game.winner = "Joueur"
        return ShootResponse(
            game_id=game_id,
            player_shot=player_shot_result,
            ai_shot=None,
            game_over=True,
            winner="Joueur",
            hiden_grille_joueur=game.hiden_grille_joueur,
            hiden_grille_ia=game.hiden_grille_ia,
            plataux_ia_grille=game.plataux_ia.grille,
            ia_heatmap=get_ia_heatmap(game)
        )

    # --- 2. TOUR DE L'IA ---
    ai_shot_result = None
    if not game.game_over:
        if model is None:
            raise HTTPException(status_code=500, detail="Modèle IA indisponible.")
        
        hidden_grille_flat = torch.tensor(
            [cell for ligne in game.hiden_grille_ia for cell in ligne], 
            dtype=torch.float32,
            device=device
        )
        
        with torch.no_grad():
            probas, logits = model(hidden_grille_flat)
            probas = torch.sigmoid(logits)
            
            for y_ia in range(10):
                for x_ia in range(10):
                    if game.hiden_grille_ia[y_ia][x_ia] != 0:
                        probas[y_ia * 10 + x_ia] = -float("inf")
            
            case_idx = probas.argmax().item()
            y_ia = case_idx // 10
            x_ia = case_idx % 10

        ai_outcome = ""
        if game.hiden_grille_ia[y_ia][x_ia] != 0:
            game.hiden_grille_ia[y_ia][x_ia] = -5
            ai_outcome = "déjà tiré"
        elif game.plataux_ia.grille[y_ia][x_ia] == 1:
            ai_outcome = "touché"
            game.hiden_grille_ia[y_ia][x_ia] = 1
            game.plataux_ia.grille[y_ia][x_ia] = 0
        else:
            ai_outcome = "raté"
            game.hiden_grille_ia[y_ia][x_ia] = -1

        ai_shot_result = TurnResult(y=y_ia, x=x_ia, result=ai_outcome)

        if sum(sum(ligne) for ligne in game.plataux_ia.grille) == 0:
            game.game_over = True
            game.winner = "IA"

    return ShootResponse(
        game_id=game_id,
        player_shot=player_shot_result,
        ai_shot=ai_shot_result,
        game_over=game.game_over,
        winner=game.winner,
        hiden_grille_joueur=game.hiden_grille_joueur,
        hiden_grille_ia=game.hiden_grille_ia,
        plataux_ia_grille=game.plataux_ia.grille,
        ia_heatmap=get_ia_heatmap(game)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)