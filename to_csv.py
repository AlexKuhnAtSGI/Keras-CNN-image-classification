import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_pickle("Train.pkl")
df = df.reshape(60000,-1).astype("float32")
df = df/255

pd.DataFrame(df).to_csv("Train.csv")
# pd.DataFrame(df[0:5]).to_csv("Train_1.csv")

# dfX = pd.read_csv("Train_1.csv")
# for _, row in dfX[0:5].iterrows():
	# row = np.array(row[1:], dtype='float64')
	# plt.imshow(row.reshape(64,128))
	# plt.show()