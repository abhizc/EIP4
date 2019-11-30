# Assignment 3

Final Validation accuracy for Base Network = "Accuracy on test data is: 82.35"

## Model Defination

model = Sequential()
 
model.add(SeparableConv2D(64, 3, 3, activation='relu', use_bias=False, input_shape=(32, 32, 3))) #30
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(128, 3, 3, activation='relu', use_bias=False)) #28
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(256, 3, 3, activation='relu', use_bias=False)) #26
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(10, 1, 1, activation='relu', use_bias=False)) 
model.add(MaxPooling2D(pool_size=(2, 2)))#13

model.add(SeparableConv2D(64, 3, 3, use_bias=False, activation='relu'))#11
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(SeparableConv2D(128, 3, 3, use_bias=False, activation='relu'))#9
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(256, 3, 3, use_bias=False, activation='relu'))#7
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(10, 1, 1, use_bias=False, activation='relu')) 
model.add(GlobalAveragePooling2D())

model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

# Epoc 50 Logs

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.01.
390/390 [==============================] - 69s 176ms/step - loss: 1.4833 - acc: 0.4628 - val_loss: 2.0395 - val_acc: 0.4697
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0075815011.
390/390 [==============================] - 66s 170ms/step - loss: 1.1025 - acc: 0.6070 - val_loss: 1.9194 - val_acc: 0.4940
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0061050061.
390/390 [==============================] - 66s 170ms/step - loss: 0.9651 - acc: 0.6578 - val_loss: 1.1816 - val_acc: 0.6196
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.005109862.
390/390 [==============================] - 66s 169ms/step - loss: 0.8765 - acc: 0.6907 - val_loss: 0.9791 - val_acc: 0.6655
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0043936731.
390/390 [==============================] - 66s 170ms/step - loss: 0.8059 - acc: 0.7156 - val_loss: 0.8384 - val_acc: 0.7112
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0038535645.
390/390 [==============================] - 66s 170ms/step - loss: 0.7517 - acc: 0.7374 - val_loss: 0.7653 - val_acc: 0.7385
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.003431709.
390/390 [==============================] - 66s 170ms/step - loss: 0.7030 - acc: 0.7528 - val_loss: 0.7703 - val_acc: 0.7398
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0030931024.
390/390 [==============================] - 66s 169ms/step - loss: 0.6700 - acc: 0.7618 - val_loss: 0.7533 - val_acc: 0.7414
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0028153153.
390/390 [==============================] - 66s 170ms/step - loss: 0.6424 - acc: 0.7750 - val_loss: 0.7680 - val_acc: 0.7333
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0025833118.
390/390 [==============================] - 65s 168ms/step - loss: 0.6178 - acc: 0.7821 - val_loss: 0.6564 - val_acc: 0.7740
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0023866348.
390/390 [==============================] - 65s 167ms/step - loss: 0.5982 - acc: 0.7911 - val_loss: 0.6937 - val_acc: 0.7606
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022177866.
390/390 [==============================] - 65s 167ms/step - loss: 0.5808 - acc: 0.7964 - val_loss: 0.6821 - val_acc: 0.7642
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.002071251.
390/390 [==============================] - 65s 167ms/step - loss: 0.5686 - acc: 0.8004 - val_loss: 0.6321 - val_acc: 0.7812
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0019428793.
390/390 [==============================] - 65s 167ms/step - loss: 0.5512 - acc: 0.8075 - val_loss: 0.6280 - val_acc: 0.7789
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018294914.
390/390 [==============================] - 65s 167ms/step - loss: 0.5406 - acc: 0.8096 - val_loss: 0.6365 - val_acc: 0.7814
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017286085.
390/390 [==============================] - 65s 167ms/step - loss: 0.5261 - acc: 0.8151 - val_loss: 0.6195 - val_acc: 0.7924
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00163827.
390/390 [==============================] - 65s 167ms/step - loss: 0.5140 - acc: 0.8202 - val_loss: 0.6159 - val_acc: 0.7918
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015569049.
390/390 [==============================] - 65s 167ms/step - loss: 0.5084 - acc: 0.8208 - val_loss: 0.6055 - val_acc: 0.7935
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0014832394.
390/390 [==============================] - 65s 167ms/step - loss: 0.4969 - acc: 0.8274 - val_loss: 0.6085 - val_acc: 0.7906
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.00141623.
390/390 [==============================] - 65s 167ms/step - loss: 0.4861 - acc: 0.8277 - val_loss: 0.5954 - val_acc: 0.7964
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0013550136.
390/390 [==============================] - 65s 167ms/step - loss: 0.4832 - acc: 0.8303 - val_loss: 0.5961 - val_acc: 0.7902
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.00129887.
390/390 [==============================] - 66s 168ms/step - loss: 0.4759 - acc: 0.8325 - val_loss: 0.6047 - val_acc: 0.7970
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0012471938.
390/390 [==============================] - 66s 169ms/step - loss: 0.4669 - acc: 0.8360 - val_loss: 0.6047 - val_acc: 0.7976
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0011994722.
390/390 [==============================] - 66s 170ms/step - loss: 0.4632 - acc: 0.8379 - val_loss: 0.5716 - val_acc: 0.8043
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.001155268.
390/390 [==============================] - 66s 169ms/step - loss: 0.4577 - acc: 0.8404 - val_loss: 0.5649 - val_acc: 0.8059
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0011142061.
390/390 [==============================] - 66s 169ms/step - loss: 0.4519 - acc: 0.8413 - val_loss: 0.5833 - val_acc: 0.8030
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.001075963.
390/390 [==============================] - 66s 169ms/step - loss: 0.4460 - acc: 0.8429 - val_loss: 0.5546 - val_acc: 0.8125
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.001040258.
390/390 [==============================] - 66s 169ms/step - loss: 0.4420 - acc: 0.8439 - val_loss: 0.6162 - val_acc: 0.7959
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0010068466.
390/390 [==============================] - 66s 169ms/step - loss: 0.4347 - acc: 0.8479 - val_loss: 0.5565 - val_acc: 0.8104
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0009755146.
390/390 [==============================] - 66s 169ms/step - loss: 0.4333 - acc: 0.8478 - val_loss: 0.5721 - val_acc: 0.8086
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0009460738.
390/390 [==============================] - 66s 169ms/step - loss: 0.4294 - acc: 0.8498 - val_loss: 0.5817 - val_acc: 0.8050
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.000918358.
390/390 [==============================] - 66s 169ms/step - loss: 0.4268 - acc: 0.8500 - val_loss: 0.5542 - val_acc: 0.8113
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0008922198.
390/390 [==============================] - 66s 169ms/step - loss: 0.4172 - acc: 0.8540 - val_loss: 0.5624 - val_acc: 0.8102
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0008675284.
390/390 [==============================] - 66s 169ms/step - loss: 0.4198 - acc: 0.8523 - val_loss: 0.5561 - val_acc: 0.8112
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0008441668.
390/390 [==============================] - 66s 169ms/step - loss: 0.4121 - acc: 0.8558 - val_loss: 0.5644 - val_acc: 0.8108
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0008220304.
390/390 [==============================] - 66s 169ms/step - loss: 0.4075 - acc: 0.8563 - val_loss: 0.5669 - val_acc: 0.8133
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0008010253.
390/390 [==============================] - 66s 169ms/step - loss: 0.4085 - acc: 0.8575 - val_loss: 0.5620 - val_acc: 0.8110
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0007810669.
390/390 [==============================] - 66s 169ms/step - loss: 0.3988 - acc: 0.8608 - val_loss: 0.5391 - val_acc: 0.8189
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.000762079.
390/390 [==============================] - 66s 169ms/step - loss: 0.3990 - acc: 0.8600 - val_loss: 0.5571 - val_acc: 0.8175
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0007439923.
390/390 [==============================] - 66s 169ms/step - loss: 0.3940 - acc: 0.8618 - val_loss: 0.5707 - val_acc: 0.8075
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0007267442.
390/390 [==============================] - 66s 169ms/step - loss: 0.3952 - acc: 0.8604 - val_loss: 0.5544 - val_acc: 0.8177
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0007102777.
390/390 [==============================] - 66s 169ms/step - loss: 0.3941 - acc: 0.8615 - val_loss: 0.5446 - val_acc: 0.8191
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0006945409.
390/390 [==============================] - 66s 169ms/step - loss: 0.3894 - acc: 0.8629 - val_loss: 0.5551 - val_acc: 0.8131
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0006794863.
390/390 [==============================] - 66s 169ms/step - loss: 0.3897 - acc: 0.8606 - val_loss: 0.5519 - val_acc: 0.8169
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0006650705.
390/390 [==============================] - 66s 169ms/step - loss: 0.3842 - acc: 0.8638 - val_loss: 0.5497 - val_acc: 0.8182
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0006512537.
390/390 [==============================] - 66s 169ms/step - loss: 0.3848 - acc: 0.8647 - val_loss: 0.5482 - val_acc: 0.8175
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0006379992.
390/390 [==============================] - 66s 169ms/step - loss: 0.3832 - acc: 0.8654 - val_loss: 0.5556 - val_acc: 0.8167
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0006252736.
390/390 [==============================] - 66s 169ms/step - loss: 0.3783 - acc: 0.8678 - val_loss: 0.5388 - val_acc: 0.8252
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0006130456.
390/390 [==============================] - 66s 169ms/step - loss: 0.3751 - acc: 0.8683 - val_loss: 0.5456 - val_acc: 0.8204
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0006012868.
390/390 [==============================] - 66s 169ms/step - loss: 0.3774 - acc: 0.8670 - val_loss: 0.5404 - val_acc: 0.8191
Model took 3294.33 seconds to train
Accuracy on test data is: **81.91**


**Highest Validation Accuracy was 82.52**
