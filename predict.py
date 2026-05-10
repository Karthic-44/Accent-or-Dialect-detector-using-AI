import torch 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from model import model as Model
from load import load
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "model/model.pth"

def load_model(model_path : str = model_path):
    checkpoint = torch.load(model_path, map_location=device)

    le = LabelEncoder()
    le.classes_ = np.array(checkpoint["label_classes"])

    model = Model(num_classes=checkpoint["num_classes"], freeze_base= True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model,le

_model , _label_encoder = load_model()

def predict(file_path: str) -> dict:
    audio = load(file_path).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(_model(audio),dim=1).squeeze(0)

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    accent = _label_encoder.inverse_transform([pred_idx])[0]
    
    top3_idx = probs.topk(min(3,len(probs))).indices.tolist()
    top3 = [{
        "accent": _label_encoder.inverse_transform([i])[0],
        "confidence": round(probs[i].item(),4)
    }
    for i in top3_idx
    ]

    return {"accent": accent, "confidence": round(confidence,4),"top3": top3}
    