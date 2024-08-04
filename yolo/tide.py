from tidecv import TIDE
import tidecv.datasets as datasets

# json文件
annFile = 'annotations.json'  # 数据集标注json文件路径
resFile = '.\\runs\\val\\exp6\\best_predictions.json'  # yolo系列val保存的json结果文件路径
gt = datasets.COCO(annFile)
bbox_results = datasets.COCOResult(resFile)
tide = TIDE()
# 注意mode参数，TIDE.BOX是边界框评估，TIDE.MASK是segmentation所使用.
# name参数为字符串，是画图图片标题，以及在控制台评估内容第一行呈现的
tide.evaluate_range(gt, bbox_results, mode=TIDE.BOX, name='4')
# 打印在控制台中运行的评估
tide.summarize()
# 图像的形式呈现出来
tide.plot()

