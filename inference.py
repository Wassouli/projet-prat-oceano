import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms

from utils.flow_utils import flow_to_image, resize_flow
from utils.torch_utils import restore_model
from models.pwclite import PWCLite
import flowpy
import matplotlib.pyplot as plt
import cv2
from utils.warp_utils import flow_warp

class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/KITTI15/pwclite_ar.tar')
    parser.add_argument('-s', '--test_shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img1.png', 'examples/img2.png'])
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': len(args.img_list),
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    
    t3=args.img_list[0].split('_')
    t0=t3[2].split('.')
    t=args.img_list[1].split('_')
    t1=t[2].split('.')
       
    ts = TestHelper(cfg)

    imgs = [imageio.imread(img).astype(np.float32) for img in args.img_list]
    h, w = imgs[0].shape[:2]

    flow_12 = ts.run(imgs)['flows_fw'][0]

    flow_12 = resize_flow(flow_12, (h, w))
    np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])

    vis_flow = flowpy.flow_to_rgb(np_flow_12)

    cv2.imwrite(t0[0]+ " " +t1[0]+".png", vis_flow)

    im1=cv2.imread("/content/drive/MyDrive/data1/NATL_AN_2007-01-03.png")
    im2=cv2.imread("/content/drive/MyDrive/data1/NATL_AN_2007-01-04.png ")
    
    im1=flow_warp(im2, flow12, pad='border', mode='bilinear'):
    
    
    cv2.imwrite("warped" +t0[0]+ " " +t1[0]+".png", warped)
    def PSNR(original, compressed): 
      mse = np.mean((original - compressed) ** 2) 
      if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                    # Therefore PSNR have no importance. 
          return 100
      max_pixel = 255.0
      psnr = 20 * log10(max_pixel / sqrt(mse)) 
      return psnr 
  # 5. Compute the Structural Similarity Index (SSIM) between the two
  #    images, ensuring that the difference image is returned


    print(PSNR(warped,im2))
