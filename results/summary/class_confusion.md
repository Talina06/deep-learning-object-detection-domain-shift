| run                   | KITTI GT   |    n | best-overlap label (F1-opt thr, IoU≥0.5)                |
|:----------------------|:-----------|-----:|:--------------------------------------------------------|
| frcnn_r50v2_kitti_tv  | car        | 5716 | car 76%, (missed) 24%, truck 0%                         |
| frcnn_r50v2_kitti_tv  | van        |  555 | (missed) 81%, car 13%, truck 4%, bus 2%                 |
| frcnn_r50v2_kitti_tv  | truck      |  223 | (missed) 61%, truck 24%, bus 8%, train 7%               |
| frcnn_r50v2_kitti_tv  | pedestrian |  921 | person 58%, (missed) 42%                                |
| frcnn_r50v2_kitti_tv  | cyclist    |  352 | (missed) 77%, person 20%, bicycle 2%, motorcycle 0%     |
| frcnn_r50v2_kitti_tv  | tram       |  117 | (missed) 66%, train 30%, bus 4%                         |
| fcos_r50_kitti_tv     | car        | 5716 | car 73%, (missed) 27%, truck 1%                         |
| fcos_r50_kitti_tv     | van        |  555 | (missed) 52%, car 27%, truck 18%, bus 3%                |
| fcos_r50_kitti_tv     | truck      |  223 | (missed) 49%, truck 34%, bus 8%, train 7%               |
| fcos_r50_kitti_tv     | pedestrian |  921 | (missed) 50%, person 50%                                |
| fcos_r50_kitti_tv     | cyclist    |  352 | (missed) 76%, person 22%, motorcycle 1%, bicycle 1%     |
| fcos_r50_kitti_tv     | tram       |  117 | (missed) 77%, train 20%, bus 3%, truck 1%               |
| yolo11s_kitti_640     | car        | 5716 | car 74%, (missed) 25%, truck 0%, train 0%               |
| yolo11s_kitti_640     | van        |  555 | (missed) 43%, car 32%, truck 19%, bus 5%                |
| yolo11s_kitti_640     | truck      |  223 | (missed) 45%, truck 24%, train 17%, bus 13%             |
| yolo11s_kitti_640     | pedestrian |  921 | person 54%, (missed) 46%, fire hydrant 0%               |
| yolo11s_kitti_640     | cyclist    |  352 | (missed) 78%, person 20%, bicycle 1%, motorcycle 1%     |
| yolo11s_kitti_640     | tram       |  117 | (missed) 75%, train 23%, bus 2%                         |
| yolo11s_kitti_1280    | car        | 5716 | car 76%, (missed) 24%, truck 1%                         |
| yolo11s_kitti_1280    | van        |  555 | (missed) 52%, car 25%, truck 17%, bus 5%                |
| yolo11s_kitti_1280    | truck      |  223 | (missed) 55%, truck 26%, bus 11%, train 8%              |
| yolo11s_kitti_1280    | pedestrian |  921 | person 55%, (missed) 45%                                |
| yolo11s_kitti_1280    | cyclist    |  352 | (missed) 77%, person 20%, bicycle 2%, car 1%            |
| yolo11s_kitti_1280    | tram       |  117 | (missed) 73%, train 24%, bus 3%                         |
| rtdetr_l_kitti_640    | car        | 5716 | car 77%, (missed) 23%, truck 0%                         |
| rtdetr_l_kitti_640    | van        |  555 | (missed) 53%, car 25%, truck 19%, bus 3%                |
| rtdetr_l_kitti_640    | truck      |  223 | (missed) 44%, truck 36%, train 15%, bus 5%              |
| rtdetr_l_kitti_640    | pedestrian |  921 | person 52%, (missed) 48%                                |
| rtdetr_l_kitti_640    | cyclist    |  352 | (missed) 81%, person 18%, bicycle 1%, car 0%            |
| rtdetr_l_kitti_640    | tram       |  117 | (missed) 71%, train 26%, bus 3%                         |
| rtdetr_l_kitti_1280   | car        | 5716 | car 69%, (missed) 31%, truck 0%                         |
| rtdetr_l_kitti_1280   | van        |  555 | (missed) 60%, car 28%, truck 10%, bus 2%                |
| rtdetr_l_kitti_1280   | truck      |  223 | (missed) 64%, truck 28%, bus 8%, train 0%               |
| rtdetr_l_kitti_1280   | pedestrian |  921 | (missed) 52%, person 48%                                |
| rtdetr_l_kitti_1280   | cyclist    |  352 | (missed) 76%, person 21%, bicycle 2%, car 0%            |
| rtdetr_l_kitti_1280   | tram       |  117 | (missed) 79%, bus 10%, train 10%                        |
| yoloworld_s_kitti_640 | car        | 5716 | car 67%, (missed) 30%, van 3%, truck 0%                 |
| yoloworld_s_kitti_640 | van        |  555 | (missed) 37%, van 33%, truck 15%, car 15%               |
| yoloworld_s_kitti_640 | truck      |  223 | truck 66%, (missed) 27%, van 3%, car 2%                 |
| yoloworld_s_kitti_640 | pedestrian |  921 | (missed) 66%, person sitting 22%, pedestrian 12%        |
| yoloworld_s_kitti_640 | cyclist    |  352 | (missed) 81%, person sitting 11%, pedestrian 6%, car 1% |
| yoloworld_s_kitti_640 | tram       |  117 | (missed) 74%, truck 14%, tram 13%                       |