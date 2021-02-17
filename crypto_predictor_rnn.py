import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque
import random, time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# data from https://www.kaggle.com/jorijnsmit/binance-full-history
# Code is based on Sentdex's tensorflow tutorial (2018)
# you can find more detailed information on pythonprogramming.net

# initial parameters
SEQ_LEN = 60  # the sequence length to base our predictions (in this case, model will predict the next prices based on the last 60 periods' data)
FUTURE_PERIOD_PREDICT = 3 # period count that we will predict (in this case, model will predict the next 3 time periods' prices)
RATIO_TO_PREDICT = "BTC-USDT"  # the ratio that we want to predict the prices of
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED"


def get_rid_of_them_mf_zeroes(df):
	for col in df.columns:
		for i in range(df[col].size):
			if col != 'target':
				if i != 0:
					if df[col][i] == 0:
						df[col][i] = df[col][i-1]
				else:
					df.drop(df.index[0])

	return df

def classify(current, future):
	if float(future) > float(current):
		return 1
	else:
		return 0

def preprocess_df(df):
	scaler = preprocessing.MinMaxScaler()
	df = df.drop(columns='future')
	df = df.astype(float)
	columns = df.columns
	for col in df.columns:
		if col != "target":
			df[col] = df[col].pct_change()
			df.dropna(inplace=True)

	df = pd.DataFrame(scaler.fit_transform(pd.DataFrame(df)))
	df.columns = columns
	df.dropna(inplace=True)

	sequential_data = []
	prev_days = deque(maxlen=SEQ_LEN)

	for i in df.values:
		prev_days.append([n for n in i[:-1]]) #every column except target
		if len(prev_days) == SEQ_LEN:
			sequential_data.append([np.array(prev_days), i[-1]])

	random.shuffle(sequential_data)
	
	buys = []
	sells = []

	for seq, target in sequential_data:
		if target == 0:
			sells.append([seq, target])
		elif target == 1:
			buys.append([seq, target])

	lower = min(len(buys), len(sells))
	buys = buys[:lower]
	sells = sells[:lower]

	sequential_data = buys+sells
	random.shuffle(sequential_data)

	X = []
	y= []

	for seq, target in sequential_data:
		X.append(seq)
		y.append(target)

	return np.array(X), y

# creating dataframe
main_df = pd.DataFrame()
pairs = ["AVAX-USDT", "BCH-USDT", "BTC-USDT", "ETH-USDT", "LTC-USDT", "TRX-USDT"]
for pair in pairs:
	dataset = f"{pair}.parquet"
	df = pd.read_parquet(dataset).drop(columns=['quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'])
	df = df.rename(columns={"close": f"{pair}_close","volume": f"{pair}_volume"})
	df = df.drop(columns=['open', 'high', 'low'])

	if len(main_df) == 0:
		main_df = df
	else:
		main_df = main_df.join(df)


#main_df.div(100)


main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df['future']))

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

main_df = get_rid_of_them_mf_zeroes(main_df)
validation_main_df = get_rid_of_them_mf_zeroes(validation_main_df)

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

model = Sequential()
model.add(LSTM(256, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
			  optimizer=opt,
			  metrics=['accuracy'])

tensorborad = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_loss:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath), monitor='val_loss', verbose=1, save_best_only=True, mode='max')

history = model.fit(
				train_x, train_y,
				batch_size=BATCH_SIZE,
				epochs=EPOCHS,
				validation_data=(validation_x, validation_y),
				callbacks=[tensorborad, checkpoint])
