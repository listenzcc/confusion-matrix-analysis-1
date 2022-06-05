# %%
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

# %%
folder = Path(__file__).parent.joinpath('private/data')

# %%


def parse_file_name(name):
    string = name[13:-4]
    area, subject, folder = [int(e) for e in string.split('-')]
    return area, subject, folder


# %%
record = []

for file in tqdm(folder.iterdir()):
    if not file.is_file:
        continue

    if not file.name.endswith('.npy'):
        continue

    with open(file.as_posix(), 'rb') as f:
        y_pred = np.load(f)
        y_true = np.load(f)

    area, subject, folder = parse_file_name(file.name)

    record.append([y_pred, y_true, area, subject, folder])

table = pd.DataFrame(
    record, columns=['pred', 'true', 'area', 'subject', 'folder'])
table

# %%
