__author__ = 'fucus'

class data:
    train_img_dirs = ["/home/chenqiang/data/CASIA_gait_data_GEI_center_gravity"]
    test_accu = ['nm']

class CNN:
    test_batch_size = 64

    epoch = 3
    batch_size = 128

    load_image_to_memory_every_time = 8192

    load_image_to_memory_every_time_when_test = 8192

    val_every = 800 # [10, 20, 40]
    lr = 1e-3
    margin = 5 # [1, 10, 100, 1000]
    model_name = 'vgg_like'
    K = 5
    output_dim = 2048 # [512, 1024, 2048, 4096]
