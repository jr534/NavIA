import uuid
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

try:
    import train
except ImportError:
    raise ImportError("The 'train.py' file must be present in the same folder.")

app = FastAPI(
    title="Navia AI API with Heatmap (FCN & MLP)",
    description="Battleship AI API showing predictive heatmaps with FCN and MLP support",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & Model Loading ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load MLP Model (V1)
model_mlp = None
try:
    model_mlp = train.BattleshipMLP()
    model_mlp.load_state_dict(torch.load('models/mlp_bataille_navale.pth', map_location=device))
    model_mlp.eval()
    print("MLP Model (V1) successfully loaded.")
except Exception as e:
    print(f"Error loading MLP model: {e}")

# 2. Define and load FCN Model (V2)
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
        return self.network(x)

model_fcn = None
try:
    model_fcn = FCN()
    import os
    fcn_path = 'models/Navia_V2_FCN.pt' if os.path.exists('models/Navia_V2_FCN.pt') else 'Navia_V2_FCN.pt'
    model_fcn.load_state_dict(torch.load(fcn_path, map_location=device, weights_only=True))
    model_fcn.eval()
    print(f"FCN Model (Navia V2) successfully loaded from {fcn_path}.")
except Exception as e:
    print(f"Error loading FCN model: {e}")


class GameSession:
    def __init__(self, model_type: str = "fcn"):
        self.model_type = model_type  # "mlp" or "fcn"
        self.plataux_joueur = train.Grille()
        self.plataux_joueur.gen_grille()
        self.hiden_grille_joueur = [[0 for _ in range(10)] for _ in range(10)]

        self.plataux_ia = train.Grille()
        self.plataux_ia.gen_grille()
        self.hiden_grille_ia = [[0 for _ in range(10)] for _ in range(10)]

        self.game_over = False
        self.winner = None


games: Dict[str, GameSession] = {}


# --- Helper for FCN Input ---
def build_fcn_input(hiden_grille_ia) -> torch.Tensor:
    """Builds the (1, 3, 10, 10) input tensor for the FCN."""
    hit = [[1.0 if cell == 1 else 0.0 for cell in row] for row in hiden_grille_ia]
    miss = [[1.0 if cell == -1 or cell == -5 else 0.0 for cell in row] for row in hiden_grille_ia]
    not_explored = [[1.0 if cell == 0 else 0.0 for cell in row] for row in hiden_grille_ia]
    
    canaux = [hit, miss, not_explored]
    t = torch.tensor(canaux, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 10, 10)
    return t


# --- Helper for Heatmap Generation ---
def get_ia_heatmap(game: GameSession) -> List[List[float]]:
    """Calculates ship presence probabilities based on selected model."""
    if game.model_type == "fcn":
        if model_fcn is None:
            return [[0.0 for _ in range(10)] for _ in range(10)]
        
        x_in = build_fcn_input(game.hiden_grille_ia)
        with torch.no_grad():
            logits = model_fcn(x_in)                     # (1, 1, 10, 10)
            probas = torch.sigmoid(logits).squeeze()     # (10, 10)
            
            # Force already targeted cells to 0%
            for y in range(10):
                for x in range(10):
                    if game.hiden_grille_ia[y][x] != 0:
                        probas[y][x] = 0.0
            return probas.cpu().tolist()
    else:
        # MLP
        if model_mlp is None:
            return [[0.0 for _ in range(10)] for _ in range(10)]
        
        hidden_grille_flat = torch.tensor(
            [cell for ligne in game.hiden_grille_ia for cell in ligne], 
            dtype=torch.float32,
            device=device
        )
        
        with torch.no_grad():
            probas, logits = model_mlp(hidden_grille_flat)
            probas = torch.sigmoid(logits)
            
            # Force already targeted cells to 0%
            for y in range(10):
                for x in range(10):
                    if game.hiden_grille_ia[y][x] != 0:
                        probas[y * 10 + x] = 0.0
                        
            return probas.cpu().view(10, 10).tolist()


# --- Data Models ---
class ShootRequest(BaseModel):
    y: int = Field(..., ge=0, le=9)
    x: int = Field(..., ge=0, le=9)

class TurnResult(BaseModel):
    y: int
    x: int
    result: str

class ShootResponse(BaseModel):
    game_id: str
    model_type: str
    player_shot: TurnResult
    ai_shot: Optional[TurnResult] = None
    game_over: bool
    winner: Optional[str] = None
    hiden_grille_joueur: List[List[int]]
    hiden_grille_ia: List[List[int]]
    plataux_ia_grille: List[List[int]]
    ia_heatmap: List[List[float]]


@app.get("/")
def get_index():
    return FileResponse("index.html")


@app.post("/game/start", status_code=status.HTTP_201_CREATED)
def start_game(model_type: str = "fcn"):
    if model_type not in ["mlp", "fcn"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'mlp' or 'fcn'.")
        
    game_id = str(uuid.uuid4())
    game = GameSession(model_type=model_type)
    games[game_id] = game
    
    return {
        "game_id": game_id,
        "model_type": game.model_type,
        "message": f"New game created with {game.model_type.upper()} model.",
        "hiden_grille_joueur": game.hiden_grille_joueur,
        "hiden_grille_ia": game.hiden_grille_ia,
        "plataux_ia_grille": game.plataux_ia.grille,
        "ia_heatmap": get_ia_heatmap(game)
    }


@app.post("/game/{game_id}/shoot", response_model=ShootResponse)
def player_shoot(game_id: str, request: ShootRequest):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found.")
    
    game = games[game_id]
    if game.game_over:
        raise HTTPException(status_code=400, detail="Game is already over.")

    y, x = request.y, request.x

    # --- 1. PLAYER'S TURN ---
    player_outcome = ""
    if game.hiden_grille_joueur[y][x] != 0:
        game.hiden_grille_joueur[y][x] = -5
        player_outcome = "already shot"
    elif game.plataux_joueur.grille[y][x] == 1:
        player_outcome = "hit"
        game.hiden_grille_joueur[y][x] = 1
        game.plataux_joueur.grille[y][x] = 0
    else:
        player_outcome = "miss"
        game.hiden_grille_joueur[y][x] = -1

    player_shot_result = TurnResult(y=y, x=x, result=player_outcome)

    if sum(sum(ligne) for ligne in game.plataux_joueur.grille) == 0:
        game.game_over = True
        game.winner = "Player"
        return ShootResponse(
            game_id=game_id,
            model_type=game.model_type,
            player_shot=player_shot_result,
            ai_shot=None,
            game_over=True,
            winner="Player",
            hiden_grille_joueur=game.hiden_grille_joueur,
            hiden_grille_ia=game.hiden_grille_ia,
            plataux_ia_grille=game.plataux_ia.grille,
            ia_heatmap=get_ia_heatmap(game)
        )

    # --- 2. AI'S TURN ---
    ai_shot_result = None
    if not game.game_over:
        y_ia, x_ia = 0, 0
        if game.model_type == "fcn":
            if model_fcn is None:
                raise HTTPException(status_code=500, detail="AI FCN model is unavailable.")
            
            x_in = build_fcn_input(game.hiden_grille_ia)
            with torch.no_grad():
                logits = model_fcn(x_in)
                probas = torch.sigmoid(logits).squeeze()
                
                # Mask out explored cells
                for y_temp in range(10):
                    for x_temp in range(10):
                        if game.hiden_grille_ia[y_temp][x_temp] != 0:
                            probas[y_temp][x_temp] = -float("inf")
                
                case_idx = probas.argmax().item()
                y_ia = case_idx // 10
                x_ia = case_idx % 10
        else:
            # MLP
            if model_mlp is None:
                raise HTTPException(status_code=500, detail="AI MLP model is unavailable.")
            
            hidden_grille_flat = torch.tensor(
                [cell for ligne in game.hiden_grille_ia for cell in ligne], 
                dtype=torch.float32,
                device=device
            )
            
            with torch.no_grad():
                probas, logits = model_mlp(hidden_grille_flat)
                probas = torch.sigmoid(logits)
                
                for y_ia_temp in range(10):
                    for x_ia_temp in range(10):
                        if game.hiden_grille_ia[y_ia_temp][x_ia_temp] != 0:
                            probas[y_ia_temp * 10 + x_ia_temp] = -float("inf")
                
                case_idx = probas.argmax().item()
                y_ia = case_idx // 10
                x_ia = case_idx % 10

        ai_outcome = ""
        if game.hiden_grille_ia[y_ia][x_ia] != 0:
            game.hiden_grille_ia[y_ia][x_ia] = -5
            ai_outcome = "already shot"
        elif game.plataux_ia.grille[y_ia][x_ia] == 1:
            ai_outcome = "hit"
            game.hiden_grille_ia[y_ia][x_ia] = 1
            game.plataux_ia.grille[y_ia][x_ia] = 0
        else:
            ai_outcome = "miss"
            game.hiden_grille_ia[y_ia][x_ia] = -1

        ai_shot_result = TurnResult(y=y_ia, x=x_ia, result=ai_outcome)

        if sum(sum(ligne) for ligne in game.plataux_ia.grille) == 0:
            game.game_over = True
            game.winner = "IA"

    return ShootResponse(
        game_id=game_id,
        model_type=game.model_type,
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