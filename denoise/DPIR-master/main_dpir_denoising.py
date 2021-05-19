import os.path
import logging

import numpy as np
import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util

"""
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
by Kai Zhang (01/August/2020)
"""

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 15                 # set AWGN noise level for noisy image
    noise_level_model = noise_level_img  # set noise level for model
    model_name = 'drunet_color'          # set denoiser model, 'drunet_gray' | 'drunet_color'
    testset_name = 'bsd68'               # set test set,  'bsd68' | 'cbsd68' | 'set12'
    x8 = False                           # default: False, x8 to boost performance

    n_channels = 3                       # fixed
    model_pool = 'model_zoo'             # fixed
    testsets = 'testsets'                # fixed
    noise_set = 'noise'                  # fixed
    results = 'results'                  # fixed
    task_current = 'dn'                  # 'dn' for denoising
    result_name = testset_name + '_' + task_current + '_' + model_name

    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    L_path_noisy = os.path.join(noise_set, result_name) # L_path, for noisy images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    logger.info('model_name:{}, model sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    L_paths_noisy = util.get_image_paths(L_path_noisy)

    for idx, img in enumerate(L_paths_noisy):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        # Add noise without clipping
        #np.random.seed(seed=0)  # for reproducibility
        #img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        img_L = util.single2tensor4(img_L)
        img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
        img_L = img_L.to(device)

        #util.imsave(util.single2uint(img_L), os.path.join(L_path_noisy, img_name+ext))

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        if not x8 and img_L.size(2)//8==0 and img_L.size(3)//8==0:
            img_E = model(img_L)
        elif not x8 and (img_L.size(2)//8!=0 or img_L.size(3)//8!=0):
            img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
        elif x8:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

if __name__ == '__main__':
    main()
