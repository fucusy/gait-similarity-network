__author__ = 'fucus'


class Project:

    # required, your project's absolute path, in other way, it's the absolute path for this file
    project_path = "/home/chenqiang/github/gait-similarity-network/"

    driver_img_list_path = ""

    # required, this path contain the train image in sub-folder, the there are ten sub-folders, c0, c1, c2, c3 .... c9
    train_img_folder_path = "/Users/fucus/Documents/buaa/projects/State_Farm_Distracted_Driver_Detection/data/imgs/train"

    # required, this path contain imgs to be tested
    test_img_folder_path = "/Users/fucus/Documents/buaa/projects/State_Farm_Distracted_Driver_Detection/data/imgs/test"

    # the original training folder downloaded from kaggle
    original_training_folder = ""

    # the original test folder downloaded from kaggle
    original_testing_folder = ""

    # not required, a img path for you exercise program
    test_img_example_path = "/Users/fucus/Documents/buaa/projects/State_Farm_Distracted_Driver_Detection/data/imgs/train/c0/img_27327.jpg"

    # required, result output path
    result_output_path = "./result/"

    # required, save cache or not
    save_cache = False


    vgg_weight_file_path = "/home/chenqiang/kaggle_driver_data/vgg16_weights.h5"

class data:
    train_type = "nm" 
    train_seq = ["01", "02", "03", "04"] 
    train_angle = "090" 
     
    test_type = "nm" 
    test_seq = ["01", "02"] 
    test_angle = "054"

    train_img_dirs = ["/home/chenqiang/data/CASIA_gait_data_GEI"]

class CNN:
    #
    keras_train_weight = "%s/CNN/vgg_try_karea/cache/model_weights_2_vgg_16_2x20.h5" % Project.project_path
    # keras structure files
    keras_structure_files = ""

    fine_tuning_vgg_weight_file_path = ""

    '''/path/to/json/vgg_self_exp1_keras_arch.json'''
    model_arch_file_name = '/home/liuzheng/competition/kaggle/distractedDrivers/vgg_self_exp1_keras_arch.json'

    '''/path/to/h5/vgg16_weights.h5'''
    model_weights_file_name = '/home/liuzheng/competition/kaggle/distractedDrivers/vgg_self_exp1_keras_weights_best_vLoss0.11026_vAcc0.969.h5'

    optimizer = 'sgd'

    train_iter = 50

    test_batch_size = 64

    batch_size = 64

    '''path/to/save/model/'''
    model_save_path = '%s/cache' % Project.project_path

    '''path/to/prediction/and/file/name/'''
    prediction_save_file = '%s/result/prediction.csv' % Project.project_path

    feature_save_path = '%s/cache' % Project.project_path

    load_image_to_memory_every_time = 8192

    load_image_to_memory_every_time_when_test = 8192

    val_every = 50

    lr = 1e-3
    margin = 50
    model_name = 'siamses_test'
