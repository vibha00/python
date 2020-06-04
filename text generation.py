#import dependencies
import numpy
import sys
nltk.download('stopwords')
from nltk.tokenize import RegexptTokenizer
from nltk.corpus import stopwords
from keras.models import sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils
from keras.callbacks import Model_Checkpoint
#load data
file=open("frankenstein=2.text").read()
#tokenization
#standardization
def tolenize_words(input):
input=input.lower
tokenizer=RegexptTokenizer(r'\w+')
tokens=tokenizer.tokenize(input)
filtered=filter(lambda token not is stopwords.words('english'),token)
return " ",join(filtered)
processed_inputs=tokenize_words(file)
#chars to numbers
chars=sorted(list(set(processed_input)))
char_to_num=dict(c,i)for i,c is enumerate (chars))
#one_bot encoding
y=np_ulits.to_categorical(y_data)
#creating the model
model=sequential()
model.add(LSTM(256,input_ shape=(X_shape[1],X.shape[2],return_sequences=true))
model.add(LSTM(256,return_sequences=true))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
#compile the model
model.compile(loss='categorical_crossentropy',optimize='adan')
#saving weights
filepath="model_weights_saved.hdfs"
Checkpoint=modelcheckpoint(filepath,monitor='loss',verbose=1,save_best_only=true,mode='min')
desired_callbacks=[checkpoint]
#fit model and let it train
model.fit(X,y,epochs=4,batch_size=256,callbacks=desired_callbacks)
#recompile model with the saved weights
filename="model_weights_saved.hdfs")
model.load_weights(filename)
model.compile(loss='categorical_crossentropy',optimizer='adan')
#output of the model back into characters
num_to_char=dict((i,c)for i,c in enumerate(chars))
#random seed to help generate
start=numpy.random.randict(0,len(x_data)=1)
pattern=x_data(start)
print("randon seed:")
print("\",''join(num_to_char(value)for value in pattern)),"\")
#generate the test
for i in range (1000):
X=numpy.reshape(pattern,(l,len(pattern),1))
x=x/float(vocab_len)
prediction=model.predict(x,verbose=0)
index=numpy.argmax(prediction)
result=num_to_char[index]
seg_in=[num_to_char[value]for value in pattern]
sys.stdout.write(result)
pattern.append(index)
pattern=pattern[l:len(pattern)]

