import os 
import torch 
import numpy as np
import pandas as pd
import time #

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from load import load
from model import model as Model

DATA_PATH = r"F:\dataset\cv-corpus\en\clips"
LABELS_FILE =r"F:\processed dataset\preprocessed.csv"
MODEL_OUT = r"model\model.pth"

BATCH_SIZE = 16
EPOCHS=15
LR=3e-4
VAL_SPLIT=0.15
SEED=42

class dataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder, data_path : str ):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.data_path = data_path
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx:int):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_path,row["file"])
        audio = load(path)
        label = int(self.label_encoder.transform([row["accent"]])[0])
        return audio,label
    
def evaluate(model, loader,device):
    model.eval()
    all_preds, all_labels = [],[]
    
    with torch.no_grad():
        for audio, label in loader:
            audio, label = audio.to(device), label.to(device)
            output = model(audio)
            all_preds.extend(output.argmax(dim=1).cpu().tolist())
            all_labels.extend(label.cpu().tolist())
        accuracy = sum(p==l for p , l in zip(all_preds,all_labels)) / len(all_labels)
    return accuracy
    
def train():
    torch.manual_seed(SEED)
    df = pd.read_csv(LABELS_FILE)
    le = LabelEncoder()
    le.fit(df["accent"])
    num_classes = len(le.classes_)
    class_names = le.classes_.tolist()
    
    train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED, stratify=df["accent"])
    train_loader = DataLoader(dataset(train_df,le,DATA_PATH),
                                batch_size=BATCH_SIZE, shuffle=True, num_workers=4,pin_memory=True)
    
    val_loader = DataLoader(dataset(val_df,le,DATA_PATH),
                            batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(num_classes=num_classes).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable,lr=LR,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()
    best = 0.0
    
    total_start = time.time()
    for epoch in range(1,EPOCHS+1):
        epoch_start = time.time() #
        model.train()
        running_loss = 0.0
        
        for audio,label in train_loader:
            audio,label = audio.to(device),label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(audio),label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,max_norm=1.0)
            optimizer.step()
            running_loss +=loss.item()
        
        scheduler.step()
        val_acc = evaluate(model,val_loader,device)
        avg_train = running_loss/len(train_loader)
        
        #######
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - total_start
        remaining = (EPOCHS -epoch) *epoch_time
        
        print(f"Epoch: {epoch}/{EPOCHS} \n Time: {epoch_time:.2f} \n ETA:{remaining/60:.2f} \n average_train: {avg_train:.4f} \n val Acc: {val_acc:.4f}")
        
        #######
        if val_acc>best:
            best = val_acc
            os.makedirs(os.path.dirname(MODEL_OUT),exist_ok=True)
            torch.save({"model_state": model.state_dict(),
                        "label_classes": class_names,
                        "num_classes": num_classes}, MODEL_OUT)

if __name__ == "__main__":
    train()