import pandas as pd
import os

train = pd.read_csv(os.path.join('data', 'agricultural_train.csv'))
val   = pd.read_csv(os.path.join('data', 'agricultural_val.csv'))
test  = pd.read_csv(os.path.join('data', 'agricultural_test.csv'))

full = pd.concat([train, val, test], ignore_index=True)
full.to_csv(os.path.join('data', 'agricultural_full.csv'), index=False)
print("合并完成，总样本数：", len(full))