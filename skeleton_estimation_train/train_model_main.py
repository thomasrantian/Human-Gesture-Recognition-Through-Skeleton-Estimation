"""
Train the Keras model for the human skeleton detection;
Created by Thomas Tian (tianran@umich.edu)in EECS 504 course
"""
# data generatir
def gen(df):
    while True:
        for i in df.get_data():
            yield i

"""
Import model builder
"""
from model_builder import *
from external_function_lib import *
from keras.models import load_model
from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow import *


if __name__ == '__main__':
    # Need to download the COCO data set from the link before running
    data_path = "../data/skeleton_train_data.lmdb"

    if not os.path.exists(data_path):
        print ("Training data not dectected!")
        print ("Please download a sample saved train data (lmbd file) following readme or download the full COCO data (65GB)")
        exit()


    # pre-trained weights path
    weight_path = "../data/model_weights.h5"
    if not os.path.exists(weight_path):
        print ("Pretrain weights not dectected!")
        print ("Please download the weights from (https://drive.google.com/open?id=1VJiZfLsHz_VhtQBjlekh6bZIfILsnnzI) or set RE_TRAINING = 0")
        exit()

    RE_TRAINING = 1
    # define the batch size
    batch_size = 10
    train_size = 10000

    """
    load the training COCO data 
    """
    print ("Loading the COCO data set ...")
    df = LMDBSerializer.load(data_path, shuffle=False)

    # build the batch generator
    df = BatchData(df, batch_size, use_list=False)
    df = MapData(df, lambda x: ( [x[0], x[1], x[2]],
                                [x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4]])
                                )
    df.reset_state()
    print("Loaded {} training samples!".format(train_size))

    # load a pre-trained model and continue to training
    model = generate_pose_detection_model(6)
    if RE_TRAINING:
        print ("Prepare to train the model with pre-trained weights")
        model.load_weights(weight_path)
    else:
        print("Prepare to train a new model")
        load_keras_vgg_weights(model)
    loss_funcs = get_loss_funcs()
    model.compile(loss=loss_funcs, optimizer = 'sgd', metrics=["accuracy"])
    model.fit_generator(gen(df),
                        steps_per_epoch=100,
                        epochs=2,
                        use_multiprocessing=False,
                        initial_epoch=0)
    model.save('pose_model.h5')