"""
Import the packages from Keras (note, we are using 2.2.0 version)
"""
from packages_lib import *

"""
VGG_19 model
"""

def vgg_19(x):
    filter_size = 64
    x = Conv2D( filter_size, (3, 3), padding = 'same',
                name = "conv_1_1",
                kernel_initializer=random_normal(stddev=0.01),
                    bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size, (3, 3), padding = 'same',
                name = "conv_1_2",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = "pool_1_1")(x)

    filter_size = 128
    x = Conv2D(filter_size, (3, 3), padding = 'same',
            name = "conv_2_1",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size, (3, 3), padding = 'same',
                name = "conv_2_2",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = "pool_2_1")(x)

    filter_size = 256
    x = Conv2D(filter_size, (3, 3), padding = 'same',
            name = "conv_3_1",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size, (3, 3), padding = 'same',
                name = "conv_3_2",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)
   
    x = Conv2D(filter_size, (3, 3), padding = 'same',
            name = "conv_3_3",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size, (3, 3), padding = 'same',
                name = "conv_3_4",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = "pool_3_1")(x)



    filter_size = 512
    x = Conv2D(filter_size, (3, 3), padding = 'same',
            name = "conv_4_1",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size, (3, 3), padding = 'same',
                name = "conv_4_2",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)


    x = Conv2D(256, (3, 3), padding = 'same',
                name = "add_1",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding = 'same',
                name = "add_2",
            kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    return x

"""
stage_1 model, due branch
"""

def pose_model_stage_1(x, parts_number, branch_id):
    filter_size = 128
    x = Conv2D(filter_size, (3, 3), padding = 'same',
                name = "pose_model_stage_1_con1_branch_{}".format(branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    filter_size = 128
    x = Conv2D(filter_size, (3, 3), padding = 'same',
                name = "pose_model_stage_1_con2_branch_{}".format(branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    filter_size = 128
    x = Conv2D(filter_size, (3, 3), padding = 'same',
                name = "pose_model_stage_1_con3_branch_{}".format(branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    filter_size = 512
    x = Conv2D(filter_size, (1, 1), padding = 'same',
                name = "pose_model_stage_1_con4_branch_{}".format(branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    x = Conv2D(parts_number, (1, 1), padding = 'same',
                name = "pose_model_stage_1_con5_branch_{}".format(branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    return x

"""
stage_2 model, due branch
"""
def pose_model_stage_2(x, parts_number, branch_id, stage_id):
    filter_size = 128
    x = Conv2D(filter_size, (7, 7), padding = 'same',
                name = "pose_model_stage_{}_con1_branch_{}".format(stage_id, branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)


    x = Conv2D(filter_size, (7, 7), padding = 'same',
                name = "pose_model_stage_{}_con2_branch_{}".format(stage_id, branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)


    x = Conv2D(filter_size, (7, 7), padding = 'same',
                name = "pose_model_stage_{}_con3_branch_{}".format(stage_id, branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)


    x = Conv2D(filter_size, (7, 7), padding = 'same',
                name = "pose_model_stage_{}_con4_branch_{}".format(stage_id, branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size, (7, 7), padding = 'same',
                name = "pose_model_stage_{}_con5_branch_{}".format(stage_id, branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size, (1, 1), padding = 'same',
                name = "pose_model_stage_{}_con6_branch_{}".format(stage_id, branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    x = Activation('relu')(x)

    x = Conv2D(parts_number, (1, 1), padding = 'same',
                name = "pose_model_stage_{}_con7_branch_{}".format(stage_id, branch_id),
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
    return x