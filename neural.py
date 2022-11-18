import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam

np.set_printoptions(threshold=sys.maxsize)

diabetes_df = pd.read_csv('../CAD-VLSI/archive/diabetes.csv')

# X is variables dataset, y is only outcome values
X = diabetes_df.drop('Outcome', axis=1).values
y = diabetes_df.Outcome.values

# using train_test_split to make train and test 
# datasets for variables and outcomes 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2) # random state controls shuffling process

# normalize the data
nl = Normalizer()
X_train = nl.transform(X_train)
X_test = nl.transform(X_test)

# converting numpy array to pandas to display example of normalized dataset
df_ex = pd.DataFrame(X_train, columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
print(df_ex.head())

df_y = pd.DataFrame(y_train, columns=['Outcome'])


# defining the neural network
def nn():

    # input 1x8
    inputs = Input(name='inputs', shape=[X_train.shape[1], ])
    
    layer = Dense(128, name='FC1')(inputs)
    layer = Activation('relu', name='Activation1')(layer)
   
    layer = Dense(128, name='FC2')(layer)
    layer = Activation('relu', name='Activation2')(layer)
  
    layer = Dense(128, name='FC3')(layer)
    layer = Activation('relu', name='Activation3')(layer)

    layer = Dense(1, name='OutLayer')(layer)
    layer = Activation('sigmoid', name='sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


model = nn()
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(), metrics=['accuracy'])


history = model.fit(x=X_train, y=y_train, epochs=400)

# model.save_weights('weights.hdf5')

# model.load_weights('weights.hdf5')
x = [[0.084395,0.776438,0.523252,0.000000,0.000000,0.218584,0.001409,0.261626
]]

ans = model.predict(x)
print(ans)

if ans[0][0] >= 0.5:
    print("Yes, you have diabetes")
else:
    print("No, you dont have diabetes")

# code to print weights
# for lay in model.layers:
#     name = lay.name
#     weights = lay.get_weights()
#     print(name)
#     print(" ")
#     print(weights)
#     print(" ")


# print(model.predict(x))


# plt.plot(history.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# # plt.legend(['Validation'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# # plt.legend(['Validation'], loc='upper left')
# plt.show()