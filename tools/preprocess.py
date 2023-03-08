from detectron2.data.datasets import register_coco_instances

CATEGORIES = [
    # {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Background"},
    # {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "Text"},
    # {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "Title"},
    # {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "List"},
    # {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "Table"},
    # {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "Figure"},

    # {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": "Caption"},
    # {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": "Footnote"},
    # {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": "Formula"},
    # {"color": [106, 0, 228], "isthing": 1, "id":4, "name": "List-item"},
    # {"color": [0, 60, 100], "isthing": 1, "id": 5, "name": "Page-footer"},
    # {"color": [0, 160, 100], "isthing": 1, "id": 6, "name": "Page-header"},
    # {"color": [0, 160, 177], "isthing": 1, "id": 7, "name": "Picture"},
    # {"color": [221, 100, 177], "isthing": 1, "id": 8, "name": "Section-header"},
    # {"color": [115, 43, 43], "isthing": 1, "id": 9, "name": "Table"},
    # {"color": [184, 104, 43], "isthing": 1, "id": 10, "name": "Text"},
    # {"color": [184, 244, 194], "isthing": 1, "id": 11, "name": "Title"},
    {"color": [184, 244, 194], "isthing": 1, "id": 12, "name": "Document"},

 
]

def _get_publaynet_instances_meta():
    thing_ids = [k["id"] for k in CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 6, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret
def register_dataset():

    # register_coco_instances("wrap_train", {}, "E:/fyp/warpdoc/withcoco/train_50.json", "E:/fyp/warpdoc/WarpDoc/image/rotate/")
    # register_coco_instances("wrap_val", {}, "E:/fyp/warpdoc/withcoco/val_4.json", "E:/fyp/warpdoc/WarpDoc/image/test/")
    #register_coco_instances("doclaynet_train", {}, "E:/fyp/doclaynet/COCO/train_doclaynet.json", "E:/fyp/doclaynet/PNG/")
    # register_coco_instances("doclaynet_val", {}, "E:/fyp/doclaynet/COCO/val_doclaynet.json", "E:/fyp/doclaynet/PNG/")
    # register_coco_instances("doclaynet_train_1", {}, "E:/fyp/doclaynet_rotate/COCO/train_69375_rotate15.json", "E:/fyp/doclaynet_rotate/PNG15/")
    # register_coco_instances("doclaynet_train_2", {}, "E:/fyp/doclaynet_rotate/COCO/train_69375_rotateN15.json", "E:/fyp/doclaynet_rotate/PNGN15/")

    register_coco_instances("doclaynet_train_1c", {}, "E:/fyp/doclaynet_rotate/COCO/train_69375_rotate15_1c.json", "E:/fyp/doclaynet_rotate/PNG15/")

    
    register_coco_instances("publaynet_train", {}, "E:/fyp/publaynet/train_5000_1.json", "E:/fyp/publaynet/train/")
    
    register_coco_instances("publaynet_val", {}, "E:/fyp/publaynet/val_3000_1.json", "E:/fyp/publaynet/train/")
    #register_coco_instances("doclaynet_val", {}, "E:/fyp/doclaynet/COCO/val_500_1.json", "E:/fyp/doclaynet/PNG/")
   
    # register_coco_instances("publaynet_train_1", {}, "E:/fyp/publaynet/train_5000_1.json", "E:/fyp/publaynet/train/")
    # register_coco_instances("publaynet_train_2", {}, "E:/fyp/publaynet/train_5000_2.json", "E:/fyp/publaynet/train/")
    # register_coco_instances("publaynet_train_3", {}, "E:/fyp/publaynet/train_5000_3.json", "E:/fyp/publaynet/train/")
    # register_coco_instances("publaynet_train_4", {}, "E:/fyp/publaynet/train_5000_4.json", "E:/fyp/publaynet/train/")
    # register_coco_instances("publaynet_train_5", {}, "E:/fyp/publaynet/train_5000_5.json", "E:/fyp/publaynet/train/")
    # register_coco_instances("publaynet_train_6", {}, "E:/fyp/publaynet/train_5000_6.json", "E:/fyp/publaynet/train/")


    # register_coco_instances("publaynet_train_7", {}, "E:/fyp/publaynet/train_5000_7.json", "E:/fyp/publaynet/train/")
    # register_coco_instances("publaynet_train_8", {}, "E:/fyp/publaynet/train_5000_8.json", "E:/fyp/publaynet/train/")
    # register_coco_instances("publaynet_train_9", {}, "E:/fyp/publaynet/train_5000_9.json", "E:/fyp/publaynet/train/")
    # register_coco_instances("publaynet_train_10", {}, "E:/fyp/publaynet/train_5000_10.json", "E:/fyp/publaynet/train/")
    # register_coco_instances("publaynet_train_11", {}, "E:/fyp/publaynet/train_5000_11.json", "E:/fyp/publaynet/train/")

    # register_coco_instances("publaynet_train_12", {}, "E:/fyp/publaynet/train_5000_12.json", "E:/fyp/publaynet/train/")

    #register_coco_instances("publaynet_val", {}, "E:/fyp/publaynet/val_3000_1.json", "E:/fyp/publaynet/train/")
    register_coco_instances("doclaynet_val", {}, "E:/fyp/doclaynet/COCO/val_doclaynet.json", "E:/fyp/doclaynet/PNG/")

    register_coco_instances("doclaynet_train", {}, "E:/fyp/doclaynet/COCO/train_doclaynet.json", "E:/fyp/doclaynet/PNG/")
    register_coco_instances("doclaynet_train_title", {}, "E:/fyp/doclaynet/COCO/train_doclaynet_focus_title_formula_footnote_page-footer.json", "E:/fyp/doclaynet/PNG/")
    register_coco_instances("doclaynet_train_sample", {}, "E:/fyp/doclaynet/COCO/train_1.json", "E:/fyp/doclaynet/PNG/")

    register_coco_instances("wrap_train_1", {}, "E:/fyp/warpdoc/withcoco/train_50_rotateN75_1c.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/N75/")
    register_coco_instances("wrap_train_2", {}, "E:/fyp/warpdoc/withcoco/train_50_rotateN45_1c.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/N45/")
    register_coco_instances("wrap_train_3", {}, "E:/fyp/warpdoc/withcoco/train_50_rotateN15_1c.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/N15/")
    register_coco_instances("wrap_train_4", {}, "E:/fyp/warpdoc/withcoco/train_50_rotate75_1c.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/75/")
    register_coco_instances("wrap_train_5", {}, "E:/fyp/warpdoc/withcoco/train_50_rotate45_1c.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/45/")
    register_coco_instances("wrap_train_6", {}, "E:/fyp/warpdoc/withcoco/train_50_rotate15_1c.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/15/")

    register_coco_instances("wrap_train", {}, "E:/fyp/warpdoc/withcoco/train_50_1c.json", "E:/fyp/warpdoc/WarpDoc/image/rotate/")
    register_coco_instances("wrap_val", {}, "E:/fyp/warpdoc/withcoco/val_4_1.json", "E:/fyp/warpdoc/WarpDoc/image/test/")
    # register_coco_instances("wrap_train_1", {}, "E:/fyp/warpdoc/withcoco/train_50_rotateN75.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/N75/")
    # register_coco_instances("wrap_train_2", {}, "E:/fyp/warpdoc/withcoco/train_50_rotateN45.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/N45/")
    # register_coco_instances("wrap_train_3", {}, "E:/fyp/warpdoc/withcoco/train_50_rotateN15.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/N15/")
    # register_coco_instances("wrap_train_4", {}, "E:/fyp/warpdoc/withcoco/train_50_rotate75.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/75/")
    # register_coco_instances("wrap_train_5", {}, "E:/fyp/warpdoc/withcoco/train_50_rotate45.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/45/")
    # register_coco_instances("wrap_train_6", {}, "E:/fyp/warpdoc/withcoco/train_50_rotate15.json", "E:/fyp/warpdoc/WarpDoc/image/transformation/15/")
