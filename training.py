from model import BRC
from generator import Generator
import glob
import os
from keras.optimizers import Adam
from random import shuffle
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping

# Get the model
model = BRC()
model.model.summary()

old_weight = 'brc_2.h5'
if os.path.exists(old_weight):
    try:
        model.model.load_weights(old_weight)
        print("Load weight")
    except:
        print("Train from fresh")


# Get the generator
img_list = glob.glob(os.path.join('ground_truth','*.jpg'))
shuffle(img_list)
split = int(0.8*len(img_list))

train = Generator(img_list[0:split],8)
valid = Generator(img_list[split:],8).__getitem__(5)

# Compile model then run
optimizer = Adam(lr = 0.002)
model.model.compile(loss = 'mean_squared_error',optimizer = optimizer)


# Define Callbacks (save model, validation etc)
ckpt = ModelCheckpoint('brc_3.h5',save_best_only=True,mode = 'min')
tsb = TensorBoard(log_dir=os.path.join( os.getcwd(),'logs' ),write_graph=True,histogram_freq=1)
early_stop_cb = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=3,
                           mode='min',
                           verbose=1)
callback = [ckpt,tsb,early_stop_cb]

# Train the model
model.model.fit_generator(generator = train,
                    steps_per_epoch = 150,
                    epochs = 50,
                    verbose = 1,
                    validation_data = valid,
                    callbacks = callback)