import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_files(path:str)-> list[str]:
    """
    get path of all files in the folder
    """
    allfiles = []
    for root,_,files in os.walk(path):
        for file in files:
            if not file[0]=='.':
                allfiles.append(os.path.join(root,file))
    return allfiles

# get file path
basepath = "data/train"
filepaths = get_files(basepath)
filepaths = [f.replace(basepath,"").strip("/") for f in filepaths]

# get target
targets = [f.split("/")[0] for f in filepaths]

# make dataframe of file and target
df = pd.DataFrame({
    'file':filepaths,
    'target':targets})

# Initialize the StratifiedKFold object with 5 folds
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create a new column 'kfold' in the DataFrame and initialize it to -1
df['kfold'] = -1

# Get the values of the 'target' column to use for stratified splitting
y = df['target'].values

# Loop through the splits and assign the fold indices to the 'kfold' column
for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y)):
    df.loc[val_idx, 'kfold'] = fold

# set weight: 2 for fields and fields_roads, 1 otherwise
df["weight"] = [1]*len(df)

for idx in df.loc[df["target"]=="fields"].index:
    df.at[idx,"weight"] = 2

for idx in df.loc[df["target"]=="fields_roads"].index:
    df.at[idx,"weight"] = 2

# save data information to excel file
df.to_excel("data.xlsx",index=False)
