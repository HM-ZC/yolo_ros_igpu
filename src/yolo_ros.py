#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
import threading
import queue
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray
from std_msgs.msg import Float32
import openvino as ov


class YOLOv11DetectorGPU:
    def __init__(self):
        rospy.init_node('yolo11_detector_gpu', anonymous=True)

        # 加载相机标定参数
        self.load_camera_params()

        # 参数配置
        model_path = rospy.get_param('~model_path', '/root/model/basket11n_openvino_model/basket11n')
        self.conf_thres = rospy.get_param('~conf_threshold', 0.5)
        self.iou_thres = rospy.get_param('~iou_threshold', 0.45)
        self.img_topic = rospy.get_param('~image_topic', '/usb_cam/image/compressed')
        self.queue_size = rospy.get_param('~queue_size', 1)
        self.num_threads = rospy.get_param('~num_threads', 1)

        # 初始化OpenVINO模型
        self.core = ov.Core()
        self.core.set_property({'CACHE_DIR': './cache'})  # 启用模型缓存

        # 打印可用设备
        print("Available devices:", self.core.available_devices)

        # 加载并编译模型
        model = self.core.read_model(f'{model_path}.xml', f'{model_path}.bin')
        self.compiled_model = self.core.compile_model(model, 'GPU')
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # 创建线程专用的推理请求池
        self.infer_requests = [self.compiled_model.create_infer_request()
                               for _ in range(self.num_threads)]

        # 图像处理参数
        self.input_shape = self.input_layer.shape  # (1, 3, 640, 640)
        self.img_size = (640, 640)
        self.class_names = ['basket']  # 根据实际情况修改

        # 图像处理工具
        self.bridge = CvBridge()
        self.avg_fps = 0.0

        # 队列系统
        self.input_queue = queue.Queue(maxsize=self.queue_size)
        self.output_queue = queue.Queue(maxsize=self.queue_size)

        # 线程池
        self.workers = []
        for i in range(self.num_threads):
            t = threading.Thread(target=self.worker, args=(i,))
            t.daemon = True
            t.start()
            self.workers.append(t)

        # 结果发布线程
        self.publisher_thread = threading.Thread(target=self.publish_results)
        self.publisher_thread.daemon = True

        # 订阅图像话题
        self.image_sub = rospy.Subscriber(
            self.img_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=512 * 1024
        )

        # 初始化发布器
        self.bbox_pub = rospy.Publisher('/yolo/detections', BoundingBox2DArray, queue_size=5)
        self.debug_pub = rospy.Publisher('/yolo/debug_image', Image, queue_size=2)
        self.fps_pub = rospy.Publisher('/yolo/fps', Float32, queue_size=1)

        self.publisher_thread.start()
        rospy.on_shutdown(self.shutdown_handler)

    def shutdown_handler(self):
        """安全关闭资源"""
        self.image_sub.unregister()
        for t in self.workers:
            if t.is_alive():
                t.join(timeout=1)
        self.publisher_thread.join(timeout=1)
        rospy.loginfo("节点已安全关闭")

    def load_camera_params(self):

        """从参数服务器加载相机标定参数"""
        try:
            # 从ROS参数服务器获取参数
            camera_matrix = rospy.get_param('/camera/camera_matrix')
            dist_coeffs = rospy.get_param('/camera/distortion_coefficients')

            # 转换为NumPy数组
            self.camera_matrix = np.array(camera_matrix['data']).reshape(3, 3)
            self.dist_coeffs = np.array(dist_coeffs['data']).reshape(1, 5)

            # 计算最优新相机矩阵
            self.img_size = rospy.get_param('/camera/image_size', [640, 480])
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                (self.img_size[0], self.img_size[1]),
                alpha=0
            )
            rospy.loginfo("相机参数加载成功")
        except Exception as e:
            rospy.logerr(f"加载相机参数失败: {str(e)}")
            raise

    def undistort_image(self, img):
        """执行畸变矫正"""
        return cv2.undistort(
            img,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix
        )

    def image_callback(self, msg):
        """图像接收回调（添加畸变矫正）"""
        try:
            # 解码压缩图像并进行畸变矫正
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            undistorted_img = self.undistort_image(cv_image)

            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
            self.input_queue.put_nowait((msg.header, undistorted_img))
        except Exception as e:
            rospy.logwarn(f"图像处理失败: {str(e)}")

    def worker(self, thread_id):
        """处理线程（使用OpenVINO GPU推理）"""
        while not rospy.is_shutdown():
            try:
                header, cv_image = self.input_queue.get(timeout=0.5)
                orig_h, orig_w = cv_image.shape[:2]

                # 预处理
                resized = cv2.resize(cv_image, self.img_size)
                input_tensor = self.preprocess(resized)

                # 获取该线程的推理请求
                infer_request = self.infer_requests[thread_id]

                # 执行推理
                infer_request.set_input_tensor(input_tensor)
                infer_request.start_async()
                infer_request.wait()

                # 获取输出
                output = infer_request.get_output_tensor().data[0]

                # 后处理
                candidates = self.postprocess(output, orig_w, orig_h)

                # 投递到输出队列
                self.output_queue.put((header, cv_image, candidates), timeout=0.1)

            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"处理异常: {str(e)}")

    def preprocess(self, img):
        """预处理并返回OpenVINO Tensor对象"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(img, axis=0)
        return ov.Tensor(input_data)

    def postprocess(self, output, orig_w, orig_h):
        """后处理：解析输出并应用NMS"""
        # 假设输出格式为(84, 8400)，其中前4个为坐标，后80为类别概率
        boxes = []
        scores = []
        class_ids = []

        # 遍历所有候选框
        for detection in output.T:  # 转置为(8400, 84)
            scores_all = detection[4:]
            class_id = np.argmax(scores_all)
            confidence = scores_all[class_id]

            if confidence < self.conf_thres:
                continue

            # 转换坐标为原始图像尺寸
            cx = detection[0] * orig_w / self.img_size[0]
            cy = detection[1] * orig_h / self.img_size[1]
            w = detection[2] * orig_w / self.img_size[0]
            h = detection[3] * orig_h / self.img_size[1]

            # 转换为xyxy格式
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(x1 + w)
            y2 = int(y1 + h)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(class_id)

        # 应用NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)

        candidates = []
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            x1, y1, x2, y2 = boxes[idx]
            candidates.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": scores[idx],
                "cls_id": class_ids[idx],
                "label": self.class_names[class_ids[idx]]
            })

        return candidates

    def publish_results(self):
        """结果发布（调整结果处理逻辑）"""
        fps_counter = 0
        fps_start = time.time()

        while not rospy.is_shutdown():
            try:
                header, cv_image, candidates = self.output_queue.get(timeout=0.5)
                debug_image = cv_image.copy()
                img_height = cv_image.shape[0]

                # 创建消息容器
                bbox_array = BoundingBox2DArray()
                bbox_array.header = header

                # 筛选上半部分的候选框
                valid_candidates = []
                for candidate in candidates:
                    center_y = (candidate["y1"] + candidate["y2"]) / 2
                    if center_y < img_height / 2:
                        valid_candidates.append(candidate)

                # 选择置信度最高的候选框
                if valid_candidates:
                    best = max(valid_candidates, key=lambda x: x["conf"])

                    # 填充边界框数据
                    bbox = BoundingBox2D()
                    bbox.center.x = (best["x1"] + best["x2"]) / 2.0
                    bbox.center.y = (best["y1"] + best["y2"]) / 2.0
                    bbox.size_x = best["x2"] - best["x1"]
                    bbox.size_y = best["y2"] - best["y1"]
                    bbox_array.boxes.append(bbox)

                    # 绘制检测结果
                    cv2.rectangle(debug_image,
                                  (best["x1"], best["y1"]),
                                  (best["x2"], best["y2"]),
                                  (0, 255, 0), 2)
                    cv2.putText(debug_image,
                                f"{best['label']}:{best['conf']:.2f}",
                                (best["x1"], best["y1"] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)

                # 计算并发布FPS
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    self.avg_fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()

                cv2.putText(debug_image, f"FPS: {self.avg_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 发布消息
                self.bbox_pub.publish(bbox_array)
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
                self.fps_pub.publish(Float32(self.avg_fps))

            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"发布异常: {str(e)}")


if __name__ == '__main__':
    try:
        detector = YOLOv11DetectorGPU()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass