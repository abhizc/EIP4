# Assignment 3
### Base Model
Total params: 1,172,410
Final Validation accuracy for Base Model = "Accuracy on test data is: 82.35"
Highest Validation achieved:
Epoch 40/50
390/390 [==============================] - 8s 19ms/step - loss: 0.3541 - acc: 0.8800 - val_loss: 0.5580 - **val_acc: 0.8294**

## Model using depthwise separable convolution 

### Model defineation

model = Sequential()
 
model.add(SeparableConv2D(64, 3, 3, activation='relu', use_bias=False, input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Dropout(0.15))
#layer output:30x30x64 Receptivefield:1x1

model.add(SeparableConv2D(128, 3, 3, activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(0.15))
#layer output:28x28x128 Receptivefield:3x3

model.add(SeparableConv2D(256, 3, 3, activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(0.15))
#layer output:26x26x256 Receptivefield:5x5

model.add(SeparableConv2D(64, 1, 1, activation='relu', use_bias=False))
#layer output:26x26x64 Receptivefield:5x5 
model.add(MaxPooling2D(pool_size=(2, 2)))
#layer output:13x13x64 Receptivefield:5x5

model.add(SeparableConv2D(64, 3, 3, use_bias=False, activation='relu'))#11
model.add(BatchNormalization())
model.add(Dropout(0.15))
#layer output:11x11x64 Receptivefield:9x9 

model.add(SeparableConv2D(128, 3, 3, use_bias=False, activation='relu'))#9
model.add(BatchNormalization())
model.add(Dropout(0.15))
#layer output:9x9x128 Receptivefield:13x13 

model.add(SeparableConv2D(256, 3, 3, use_bias=False, activation='relu'))#7
model.add(BatchNormalization())
model.add(Dropout(0.15))
#layer output:7x7x256 Receptivefield:17x17

model.add(SeparableConv2D(10, 1, 1, use_bias=False, activation='relu'))
#layer output:7x7x10 Receptivefield:17x17 
model.add(GlobalAveragePooling2D())

#layer output:10 Receptivefield:17x17
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])


**Total params: 113,307**

**Accuracy achieved: Accuracy on test data is: 84.03**

### Epoc Details

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.01.
781/781 [==============================] - 37s 48ms/step - loss: 1.4747 - acc: 0.4594 - val_loss: 2.3451 - val_acc: 0.3951
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0075815011.
781/781 [==============================] - 34s 43ms/step - loss: 1.0659 - acc: 0.6233 - val_loss: 1.3861 - val_acc: 0.5561
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0061050061.
781/781 [==============================] - 34s 43ms/step - loss: 0.9052 - acc: 0.6829 - val_loss: 1.0689 - val_acc: 0.6300
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.005109862.
781/781 [==============================] - 34s 44ms/step - loss: 0.8035 - acc: 0.7187 - val_loss: 0.7637 - val_acc: 0.7359
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0043936731.
781/781 [==============================] - 34s 44ms/step - loss: 0.7381 - acc: 0.7401 - val_loss: 0.7803 - val_acc: 0.7353
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0038535645.
781/781 [==============================] - 34s 43ms/step - loss: 0.6838 - acc: 0.7596 - val_loss: 0.6892 - val_acc: 0.7598
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.003431709.
781/781 [==============================] - 34s 43ms/step - loss: 0.6484 - acc: 0.7725 - val_loss: 0.6391 - val_acc: 0.7756
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0030931024.
781/781 [==============================] - 34s 43ms/step - loss: 0.6113 - acc: 0.7887 - val_loss: 0.6717 - val_acc: 0.7702
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0028153153.
781/781 [==============================] - 34s 43ms/step - loss: 0.5855 - acc: 0.7958 - val_loss: 0.6597 - val_acc: 0.7746
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0025833118.
781/781 [==============================] - 34s 43ms/step - loss: 0.5649 - acc: 0.8033 - val_loss: 0.6156 - val_acc: 0.7931
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0023866348.
781/781 [==============================] - 34s 43ms/step - loss: 0.5439 - acc: 0.8109 - val_loss: 0.5866 - val_acc: 0.8000
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022177866.
781/781 [==============================] - 34s 43ms/step - loss: 0.5255 - acc: 0.8177 - val_loss: 0.5906 - val_acc: 0.7953
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.002071251.
781/781 [==============================] - 34s 43ms/step - loss: 0.5125 - acc: 0.8206 - val_loss: 0.5836 - val_acc: 0.8001
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0019428793.
781/781 [==============================] - 34s 43ms/step - loss: 0.5003 - acc: 0.8245 - val_loss: 0.6142 - val_acc: 0.7901
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018294914.
781/781 [==============================] - 34s 43ms/step - loss: 0.4902 - acc: 0.8290 - val_loss: 0.5274 - val_acc: 0.8199
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017286085.
781/781 [==============================] - 34s 43ms/step - loss: 0.4787 - acc: 0.8334 - val_loss: 0.5332 - val_acc: 0.8222
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00163827.
781/781 [==============================] - 34s 43ms/step - loss: 0.4647 - acc: 0.8367 - val_loss: 0.5749 - val_acc: 0.8061
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015569049.
781/781 [==============================] - 34s 43ms/step - loss: 0.4602 - acc: 0.8388 - val_loss: 0.5108 - val_acc: 0.8261
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0014832394.
781/781 [==============================] - 34s 43ms/step - loss: 0.4486 - acc: 0.8449 - val_loss: 0.5270 - val_acc: 0.8257
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.00141623.
781/781 [==============================] - 34s 43ms/step - loss: 0.4413 - acc: 0.8470 - val_loss: 0.5214 - val_acc: 0.8245
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0013550136.
781/781 [==============================] - 34s 43ms/step - loss: 0.4379 - acc: 0.8466 - val_loss: 0.5151 - val_acc: 0.8214
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.00129887.
781/781 [==============================] - 34s 43ms/step - loss: 0.4301 - acc: 0.8482 - val_loss: 0.5161 - val_acc: 0.8255
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0012471938.
781/781 [==============================] - 34s 43ms/step - loss: 0.4212 - acc: 0.8531 - val_loss: 0.5174 - val_acc: 0.8250
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0011994722.
781/781 [==============================] - 34s 43ms/step - loss: 0.4180 - acc: 0.8521 - val_loss: 0.5061 - val_acc: 0.8281
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.001155268.
781/781 [==============================] - 34s 43ms/step - loss: 0.4106 - acc: 0.8540 - val_loss: 0.5024 - val_acc: 0.8290
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0011142061.
781/781 [==============================] - 34s 43ms/step - loss: 0.4020 - acc: 0.8584 - val_loss: 0.4944 - val_acc: 0.8334
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.001075963.
781/781 [==============================] - 34s 43ms/step - loss: 0.4055 - acc: 0.8574 - val_loss: 0.5291 - val_acc: 0.8179
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.001040258.
781/781 [==============================] - 34s 43ms/step - loss: 0.3927 - acc: 0.8619 - val_loss: 0.4904 - val_acc: 0.8364
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0010068466.
781/781 [==============================] - 34s 43ms/step - loss: 0.3892 - acc: 0.8628 - val_loss: 0.5068 - val_acc: 0.8304
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0009755146.
781/781 [==============================] - 34s 43ms/step - loss: 0.3892 - acc: 0.8639 - val_loss: 0.4943 - val_acc: 0.8327
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0009460738.
781/781 [==============================] - 34s 43ms/step - loss: 0.3785 - acc: 0.8672 - val_loss: 0.4893 - val_acc: 0.8372
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.000918358.
781/781 [==============================] - 34s 43ms/step - loss: 0.3758 - acc: 0.8671 - val_loss: 0.4840 - val_acc: 0.8388
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0008922198.
781/781 [==============================] - 34s 43ms/step - loss: 0.3748 - acc: 0.8671 - val_loss: 0.4942 - val_acc: 0.8369
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0008675284.
781/781 [==============================] - 34s 44ms/step - loss: 0.3726 - acc: 0.8694 - val_loss: 0.4830 - val_acc: 0.8412
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0008441668.
781/781 [==============================] - 34s 43ms/step - loss: 0.3653 - acc: 0.8713 - val_loss: 0.4990 - val_acc: 0.8347
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0008220304.
781/781 [==============================] - 34s 43ms/step - loss: 0.3657 - acc: 0.8713 - val_loss: 0.4951 - val_acc: 0.8333
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0008010253.
781/781 [==============================] - 34s 44ms/step - loss: 0.3625 - acc: 0.8726 - val_loss: 0.4988 - val_acc: 0.8349
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0007810669.
781/781 [==============================] - 34s 43ms/step - loss: 0.3557 - acc: 0.8739 - val_loss: 0.4896 - val_acc: 0.8380
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.000762079.
781/781 [==============================] - 34s 43ms/step - loss: 0.3549 - acc: 0.8747 - val_loss: 0.4788 - val_acc: 0.8393
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0007439923.
781/781 [==============================] - 34s 43ms/step - loss: 0.3528 - acc: 0.8746 - val_loss: 0.4816 - val_acc: 0.8402
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0007267442.
781/781 [==============================] - 34s 43ms/step - loss: 0.3472 - acc: 0.8757 - val_loss: 0.4904 - val_acc: 0.8380
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0007102777.
781/781 [==============================] - 34s 43ms/step - loss: 0.3518 - acc: 0.8767 - val_loss: 0.4853 - val_acc: 0.8387
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0006945409.
781/781 [==============================] - 34s 43ms/step - loss: 0.3425 - acc: 0.8790 - val_loss: 0.4767 - val_acc: 0.8429
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0006794863.
781/781 [==============================] - 34s 43ms/step - loss: 0.3403 - acc: 0.8799 - val_loss: 0.4781 - val_acc: 0.8405
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0006650705.
781/781 [==============================] - 34s 43ms/step - loss: 0.3375 - acc: 0.8814 - val_loss: 0.4965 - val_acc: 0.8374
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0006512537.
781/781 [==============================] - 34s 43ms/step - loss: 0.3415 - acc: 0.8792 - val_loss: 0.4779 - val_acc: 0.8418
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0006379992.
781/781 [==============================] - 34s 43ms/step - loss: 0.3400 - acc: 0.8809 - val_loss: 0.4911 - val_acc: 0.8386
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0006252736.
781/781 [==============================] - 34s 43ms/step - loss: 0.3369 - acc: 0.8802 - val_loss: 0.4737 - val_acc: 0.8438
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0006130456.
781/781 [==============================] - 34s 43ms/step - loss: 0.3339 - acc: 0.8809 - val_loss: 0.5104 - val_acc: 0.8311
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0006012868.
781/781 [==============================] - 34s 43ms/step - loss: 0.3311 - acc: 0.8828 - val_loss: 0.4792 - val_acc: 0.8403
Model took 1698.04 seconds to train
Accuracy on test data is: 84.03

**Maximum Accuracy: 

**Epoch 00048: LearningRateScheduler setting learning rate to 0.0006252736.
781/781 [==============================] - 34s 43ms/step - loss: 0.3369 - acc: 0.8802 - val_loss: 0.4737 - val_acc: 0.8438**


