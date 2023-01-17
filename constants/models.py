class DetectionModelConfig:
    def __init__(self, model_name, img_size, conf_thres, max_det=1):
        self.model_name = model_name
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.max_det = max_det


FP_BLURRING_MODEL = DetectionModelConfig(model_name="fp_i256_e100_b64_220920",
                                         img_size=256,
                                         conf_thres=0.1,
                                         max_det=100)
