from classifierTF import split
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import numpy as np
import pickle,random
from keras.models import Sequential, model_from_json
from keras.layers import Dense ,Activation,  Dropout, TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard, History, ReduceLROnPlateau
fh=open("keras/batch_data.csv","w") # This opens the log file to save batches metrics
fh.write("n,loss,acc,fmeasure,precision,recall,val_loss,val_acc,val_fmeasure,val_precision,val_recall\n") # Header
class BatchLogger(Callback): # Typical callback for keras to be called every n batches
    def __init__(self, display):
        self.seen = 0
        self.display = display
    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            pars=self.params['metrics']
            val_acc=logs.get('val_categorical_accuracy')
            acc=logs.get('categorical_accuracy')
            loss=logs.get('loss')
            val_loss=logs.get('val_loss')
            fmeasure=logs.get('fmeasure')
            precision=logs.get('precision')
            recall=logs.get('recall')
            val_fmeasure=logs.get('val_fmeasure')
            val_precision=logs.get('val_precision')
            val_recall=logs.get('val_recall')
            fh.write(str(self.seen)+","+str(loss)+","+str(acc)+","+str(fmeasure)+","+str(precision)+","+str(recall)+","+str(val_loss)+","+str(val_acc)+","+str(val_fmeasure)+","+str(val_precision)+","+str(val_recall)+"\n")
            fh.flush()
######### PARAMETERS TO TUNE
ratio=0.9
EPOCH=4
dropoutProb=0.5
LOAD=False
EMBDIM=300
nb_classes=8
OPTIMIZE=False
#########
texts,labels,labels_factors,words, maxLength=pickle.load(open("data/train_CaseSensitive.pickle","rb")) # Make sure data/train_CaseSensitive.pickle has been downloaded
X_train=texts # Matrix (n x m)
Y_train=labels # Matrix (n x 8)
nb_words_total=len(words)
max_words_per_sentence=maxLength
model = Sequential() # Initial blank model
model.add(Embedding(nb_words_total, EMBDIM, input_length=max_words_per_sentence, mask_zero=True)) # Embed each word into EMBDIM dimensions
model.add(Bidirectional(LSTM(EMBDIM, init='glorot_uniform', inner_init='orthogonal',activation='tanh', inner_activation='hard_sigmoid', return_sequences=False))) # Add EMBDIM neurons in LSTM (bidirectional)
model.add(Dropout(dropoutProb)) # Add dropout regularization
model.add(Dense(nb_classes)) # Normal fully connected output layer of 8
model.add(Activation('softmax')) # Select label using softmax
if LOAD: # If we don't want to train from scratch, load weights
    model.load_weights("keras/keras_lstm_cleaned.h5")
tb = TensorBoard(log_dir='keras/logs') # Tensorboard dir
model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode='categorical',metrics=['categorical_accuracy','fmeasure', 'precision', 'recall']) # Compile model

if not LOAD:
    history = History()
    early_stopping = EarlyStopping(patience=3, verbose=1) # Early cutoff after 3 epochs
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                  patience=2, min_lr=0.001) # Reduce learning rate to 0.2*learning rate after 2 epochs
    checkpointer = ModelCheckpoint(filepath='keras/keras_lstm_cleaned.h5', 
                               verbose=1, 
                               save_best_only=False) # Save weights after each epoch
    out_batch = BatchLogger(display=100) # Save metrics every 100 batches
    log=model.fit(X_train, Y_train, nb_epoch=EPOCH, batch_size=64,show_accuracy=True,shuffle=True,callbacks=[ checkpointer,tb,out_batch,history]) # Train
    fh.close()
    model.save('keras/keras_lstm_cleaned.h5') # Save weights one last time
    with open("keras/model_lstm.json", "w") as json_file:
        json_file.write(model.to_json()) # Save model configuration into .json
    pickle.dump(history.history,open("keras/metrics_lstm.pickle","wb")) # Save epoch metrics
else:
    test=pickle.load(open("data/test_CaseSensitive.pickle","rb")) # Load test set
    print("Loaded predictions from file")
    preds=model.predict_classes(test) # Predict
    pickle.dump(preds,open("data/predictions.pickle","wb")) # Save predicitions
    print("Predictions done")
