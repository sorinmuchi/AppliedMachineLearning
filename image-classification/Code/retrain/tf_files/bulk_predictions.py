import numpy as np
import tensorflow as tf
import glob
import os

modelFullPath = '/tf_files/retrained_graph.pb'
labelsFullPath = '/tf_files/retrained_labels.txt'

preds = []

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.                                                                                                                                                                       
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

if __name__ == '__main__':

    testimages=[str(i).zfill(5)+".jpg" for i in range(0,6600)]

    labels_file = open(labelsFullPath, 'rb')
    lines = labels_file.readlines()
    labels = [str(w).replace("\n", "") for w in lines]

    ## init numpy array to hold all predictions                                                                                                                                                                    
    all_predictions = np.zeros(shape=(len(testimages),40)) ## 121 categories                                                                                                                                      


    # Creates graph from saved GraphDef.                                                                                                                                                                           
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for i in range(len(testimages)):
            image_data1 = tf.gfile.FastGFile(testimages[i], 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data1})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
            score = predictions[top_k[0]]
            answer = labels[top_k[0]]
            if(score >= 0.15):
                preds.append(answer)
            else:
                preds.append("UNKNOWN")
                print('%s: %s (score = %.5f) or %s (score = %.5f)' % (testimages[i], answer, score, labels[top_k[1]], predictions[top_k[1]]))
            
#            if i % 100 == 0:
#              print(str(i) +' of a total of '+ str(len(testimages)))

    import pandas as pd 
    df = pd.DataFrame(preds, columns=["class"])
    df.to_csv("/tf_files/submit.csv")    