import train

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
if __name__ == "__main__":
    config = [
    'configs/mask_rcnn_R_101_FPN_3x_11category_1.yaml',
    # 'configs/mask_rcnn_R_101_FPN_3x_11category_2.yaml',
    # 'configs/mask_rcnn_R_101_FPN_3x_11category_3.yaml',
    # 'configs/mask_rcnn_R_101_FPN_3x_11category_4.yaml',
    # 'configs/mask_rcnn_R_101_FPN_3x_11category_5.yaml',
    # 'configs/mask_rcnn_R_101_FPN_3x_11category_6.yaml',
    # 'configs/mask_rcnn_R_101_FPN_3x_7.yaml',
    # 'configs/mask_rcnn_R_101_FPN_3x_8.yaml',
    # 'configs/mask_rcnn_R_101_FPN_3x_9.yaml',
    # 'configs/mask_rcnn_R_101_FPN_3x_10.yaml'
    ]
    for i in range(0,len(config)):
        args = default_argument_parser().parse_args()
        args.config_file=config[i]
        # if i == 0:
        #     args.resume = False
        # else:
        #     args.resume = True
        args.resume = True
        print(args)
        if i in (0,2,4):

            train.StartImgAug(args)
        else:
        
            train.Start(args)