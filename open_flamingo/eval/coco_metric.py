from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


def compute_cider(
    result_path,
    annotations_path,
):
    # create coco object and coco_result object
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_path)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    return coco_eval.eval


def postprocess_captioning_generation(predictions, split_words=['Output:']):
    for split_word in split_words:
        predictions = predictions.split(f"{split_word}", 1)[0]
    return predictions
