from detectron2.data.datasets import register_coco_instances

CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Background"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "Text"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "Title"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "List"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "Table"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "Figure"},

    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": "Caption"},
    {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": "Footnote"},
    {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": "Formula"},
    {"color": [106, 0, 228], "isthing": 1, "id":4, "name": "List-item"},
    {"color": [0, 60, 100], "isthing": 1, "id": 5, "name": "Page-footer"},
    {"color": [0, 160, 100], "isthing": 1, "id": 6, "name": "Page-header"},
    {"color": [0, 160, 177], "isthing": 1, "id": 7, "name": "Picture"},
    {"color": [221, 100, 177], "isthing": 1, "id": 8, "name": "Section-header"},
    {"color": [115, 43, 43], "isthing": 1, "id": 9, "name": "Table"},
    {"color": [184, 104, 43], "isthing": 1, "id": 10, "name": "Text"},
    {"color": [184, 244, 194], "isthing": 1, "id": 11, "name": "Title"},
    # {"color": [184, 244, 194], "isthing": 1, "id": 12, "name": "Document"},

 
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

    register_coco_instances("doclaynet_val", {}, "E:/fyp/doclaynet/COCO/val_doclaynet.json", "E:/fyp/doclaynet/PNG/")
    register_coco_instances("doclaynet_train", {}, "E:/fyp/doclaynet/COCO/train_doclaynet.json", "E:/fyp/doclaynet/PNG/")

    register_coco_instances("wrap_train", {}, "E:/fyp/warpdoc/withcoco/train_50_1c.json", "E:/fyp/warpdoc/WarpDoc/image/rotate/")
    register_coco_instances("wrap_val", {}, "E:/fyp/warpdoc/withcoco/val_4_1.json", "E:/fyp/warpdoc/WarpDoc/image/test/")

