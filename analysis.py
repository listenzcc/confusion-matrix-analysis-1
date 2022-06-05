# %%
import numpy as np
import pandas as pd

from sklearn import metrics
from tqdm.auto import tqdm

import plotly.express as px

from load_data import table as TABLE

# %%
TABLE

# %%
# Information

areas = [e for e in TABLE['area'].unique()]
subjects = [e for e in TABLE['subject'].unique()]
folders = [e for e in TABLE['folder'].unique()]

print('Area: {}'.format(areas))
print('Subject: {}'.format(subjects))
print('Folder: {}'.format(folders))

# %%

record = []

for subject in tqdm(subjects):
    for area in tqdm(areas):
        # Analysis confusion matrix for subject and area

        table = TABLE.query('area == {}'.format(area)).query(
            'subject == {}'.format(subject))

        y_pred = np.concatenate(table['pred'].to_numpy())
        y_true = np.concatenate(table['true'].to_numpy())

        report = metrics.classification_report(
            y_pred=y_pred, y_true=y_true, output_dict=True)

        acc = report['accuracy']

        matrix = metrics.confusion_matrix(
            y_pred=y_pred, y_true=y_true, normalize='true')

        matrix

        record.append([subject, area, acc, matrix, report])

table = pd.DataFrame(
    record, columns=['subject', 'area', 'accuracy', 'matrix', 'report'])
table


# %%
title = 'Accuracy by areas'
fig = px.box(table, x='area', color='area', y='accuracy', title=title)
fig.show()

title = 'Accuracy by subjects'
fig = px.box(table, x='subject', color='subject', y='accuracy', title=title)
fig.show()

# %%
n = len(table)

distance = np.zeros((n, n))

for j in range(n):
    for k in range(0, j):
        mat0 = table.iloc[j]['matrix']
        mat1 = table.iloc[k]['matrix']
        distance[j, k] = np.linalg.norm(mat0 - mat1)
        distance[k, j] = distance[j, k]

fig = px.imshow(distance)

fig.update_layout(
    yaxis=dict(
        tickvals=[e for e in range(n)],
        ticktext=table[['subject', 'area']],
    ),

    xaxis=dict(
        tickvals=[e for e in range(n)],
        ticktext=table[['subject', 'area']],
    ),
)

# %%

# %%
