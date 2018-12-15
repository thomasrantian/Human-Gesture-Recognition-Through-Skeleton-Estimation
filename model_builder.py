"""
Import the model lib
"""
from model_lib import *
from scipy.ndimage.filters import gaussian_filter
import numpy as np
def find_key_points(heatmap):
    joints = []
    joints_id = 0
    for part in range(18):
        map_ori = heatmap[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > 0.1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(joints_id, joints_id + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        joints.append(peaks_with_score_and_id)
        joints_id += len(peaks)
    return joints
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
    # model = Model( inputs = inputs, outputs = outputs)
    model = Model(inputs=[img_in], outputs = [stage_2_b1_out, stage_2_b2_out])
    return model