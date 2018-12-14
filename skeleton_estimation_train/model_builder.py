"""
Train the Keras model for the human skeleton detection;
Created by Thomas Tian (tianran@umich.edu)in EECS 504 course
"""
"""
Import the model lib
"""
from model_lib import *

'''
Build the two branch-multi stage CNN model
'''
def generate_pose_detection_model(n_stage):
    # build the input
    inputs = []
    
    img_in = Input( shape = (None, None, 3))
    inputs.append(img_in)
    w_vector_field_in = Input( shape = (None, None, 38))
    inputs.append(w_vector_field_in)
    w_confi_in = Input( shape = (None, None, 19))
    inputs.append(w_confi_in)

    # process the input image
    img_processed = Lambda(lambda x: x / 256 - 0.5)(img_in)

    #run pure vgg19
    vgg_out = vgg_19(img_processed)

    # run state 1
    stage_1_b1_out = pose_model_stage_1(vgg_out, 38, 1)
    w_1_out = Multiply(name = "weight_stage1_L1") ( [stage_1_b1_out, w_vector_field_in])

    # run state 1
    stage_1_b2_out = pose_model_stage_1(vgg_out, 19, 2)
    w_2_out = Multiply(name = "weight_stage1_L2") ( [stage_1_b2_out, w_confi_in])

    x = Concatenate()([stage_1_b1_out, stage_1_b2_out, vgg_out])
    # build output
    outputs = []
    outputs.append(w_1_out)
    outputs.append(w_2_out)

    for stage_id in range(2,n_stage + 1):
        stage_2_b1_out = pose_model_stage_2(x, 38, 1, stage_id)
        w_1_out = Multiply(name = "weight_stage{}_L1".format(stage_id)) ( [stage_2_b1_out, w_vector_field_in])

        stage_2_b2_out = pose_model_stage_2(x, 19, 2, stage_id)
        w_2_out = Multiply(name = "weight_stage{}_L2".format(stage_id)) ( [stage_2_b2_out, w_confi_in])

        outputs.append(w_1_out)
        outputs.append(w_2_out)

        if (stage_id < n_stage):
            x = Concatenate()([stage_2_b1_out, stage_2_b2_out, vgg_out])
    model = Model( inputs = inputs, outputs = outputs)
    return model
'''
Load the Keras VGG weight 
'''
def load_keras_vgg_weights(model):
    vgg_name = {
        'conv1_1': 'block1_conv1',
        'conv1_2': 'block1_conv2',
        'conv2_1': 'block2_conv1',
        'conv2_2': 'block2_conv2',
        'conv3_1': 'block3_conv1',
        'conv3_2': 'block3_conv2',
        'conv3_3': 'block3_conv3',
        'conv3_4': 'block3_conv4',
        'conv4_1': 'block4_conv1',
        'conv4_2': 'block4_conv2'
    }
    keras_vgg = VGG19(include_top=False, weights='imagenet')
    for layer in model.layers:
        if layer.name in vgg_name:
            layer.set_weights(keras_vgg.get_layer(vgg_name[layer.name]).get_weights())