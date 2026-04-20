import os 
import pandas as pd

def preprocess():
    tsv = ""
    out =""
    min_sample = 100
    
    df = pd.read_csv(tsv, sep="\t",  low_memory= False)
    df = df[df["accents"].notna() & (df["accents"].str.strip()!="")]
    df = df.rename(columns={"path":"file", "accents":"accent"})
    df["accent"] = df["accent"].str.strip().str.lower()
    counts = df["accent"].value_counts()
    valid = counts[counts >=min_sample].index
    df = df[df["accent"].isin(valid)]["file","accent"]
    
    os.makedirs(os.path.dirname(out),exist_ok=True)
    df.to_csv(out, index=False)
    
