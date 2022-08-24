from pathlib import Path

import numpy as np
from prettytable import PrettyTable


columns = ['model', 'bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']
row = ['MolHGCN']


for dataset in columns[1:]:
    results = []
    for result_path in Path(f'./saved_models/{dataset}').rglob('*.txt'):
        with open(result_path, 'r') as fin:
            results.append(float(fin.readline()) * 100)

    results = np.array(results)
    row.append(f'%.1f \u00B1 %.1f ({len(results)})' % (results.mean(), results.std()))


table = PrettyTable()
table.field_names = columns
table.add_row(row)
print(table)
