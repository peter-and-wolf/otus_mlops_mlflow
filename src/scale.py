import time

import numpy as np
import pandas as pd

def scale(train_in_path: str, 
          test_in_path: str,
          train_out_path: str,
          test_out_path: str) -> None:

  t1 = time.time()
  print('\nscaling...')
  
  train_df = pd.read_csv(train_in_path, header=None, dtype=float)
  test_df = pd.read_csv(test_in_path, header=None, dtype=float)


  train_mean = train_df.values[:, 1:].mean()
  train_std = train_df.values[:, 1:].std()

  print(f'train_mean={train_mean:.2f}, train_std={train_std:.2f}')

  train_df.values[:, 1:] -= train_mean
  train_df.values[:, 1:] /= train_std
  test_df.values[:, 1:] -= train_mean
  test_df.values[:, 1:] /= train_std
  
  np.save(train_out_path, train_df)
  np.save(test_out_path, test_df)

  print(f'done for {time.time() - t1:.2f}s')


if __name__ == '__main__':
  scale()