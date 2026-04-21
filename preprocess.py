import os 
import pandas as pd

import numpy as np

def preprocess(tsv : str, out : str):
    min_sample = 100
    
    df = pd.read_csv(tsv, sep="\t",  low_memory= False)
    df = df[df["accents"].notna() & (df["accents"].str.strip()!="")]
    df = df.rename(columns={"path":"file", "accents":"accent"})
    
    df["accent"] = df["accent"].str.strip().str.lower()
    
    df["accent"] = (df["accent"].str.replace("united states english", "usa", regex=False)
                    .str.replace("england english","british",regex=False)
                    .str.replace("india and south asia (india, pakistan, sri lanka)", "south asia", regex=False))
                    
    
    df["accent"] = df["accent"].str.strip().str.lower()
    counts = df["accent"].value_counts()
    valid = counts[counts >=min_sample].index
    df = df[df["accent"].isin(valid)][["file","accent"]]
    
    os.makedirs(os.path.dirname(out),exist_ok=True)
    df.to_csv(out, index=False)
    
        
if __name__ == "__main__":
    preprocess(r"F:\dataset\cv-corpus\en\dev.tsv",r"F:\processed dataset\preprocessed.csv")
