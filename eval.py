import os
import cv2
import yaml
import pickle
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from src.utils.checkpoint import CheckPointer
from train import make_oxford_dataloader
from train import make_coco_dataloader
from train import make_cifar_dataloader
from train import make_flir_adas_dataloader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ModelWrapper(torch.nn.Sequential):
    def __init__(self, *args):
        super(ModelWrapper, self).__init__(*args)

    def predict_homography(self, data):
        for idx, m in enumerate(self):
            data = m.predict_homography(data)
        return data


def destandardize(image, mean=0.443, std=0.129, tile=True, transpose=False):
    image = ((image * std) + mean) * 255.
    image = np.rint(image)
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)
    if transpose:
        image = image.transpose((1, 2, 0))
    if tile:
        image = np.tile(image, (1, 1, 3))
    return image


def draw_rect(image, corners, color='b', thickness=2):
    # RGB space assumption
    if color == 'r':
        rgb = (255, 0, 0)
    if color == 'g':
        rgb = (0, 255, 0)
    if color == 'b':
        rgb = (0, 0, 255)
    image = cv2.line(image, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), rgb, thickness)
    image = cv2.line(image, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), rgb, thickness)
    image = cv2.line(image, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), rgb, thickness)
    image = cv2.line(image, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), rgb, thickness)
    # return cv2.rectangle(image, tuple(corners[0].astype(int)), tuple(corners[2].astype(int)), rgb)
    return image


def evaluate(model: torch.nn.Module, eval_dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.modules.loss._Loss,
             device: str, patch_keys: list, self_supervised=False, visualize=False, postprocess=False, log_filepath=None):

    ###########################################################################
    # Device setting
    ###########################################################################

    if device == 'cuda' and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            print('Multiple GPUs detected. Using DataParallel mode.')
            model = torch.nn.DataParallel(model)
        model.to(device)
    print('Model device: {}'.format(device))

    ###########################################################################
    # Eval
    ###########################################################################

    # Training phase
    model.eval()

    # Time measurement
    model_start = torch.cuda.Event(enable_timing=True)
    model_end = torch.cuda.Event(enable_timing=True)
    model_time = []
    postprocess_start = torch.cuda.Event(enable_timing=True)
    postprocess_end = torch.cuda.Event(enable_timing=True)
    postprocess_time = []

    # Loop for the whole epoch
    batched_mace = []
    with torch.no_grad():
        for iter, data in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):

            # S-COCO VISUALIZATION VALID INDICES:
            # array([159, 630, 665, 1121, 1212, 1248, 1664, 1709, 1771, 2230, 2297])
            # MEAN OF DIFFERENCES:
            # array([0.71827478, 0.3781306 , 0.73596508, 0.31215266, 1.0124368 ,
            #        0.49932163, 0.621531  , 0.96423396, 1.33473603, 0.72880919,
            #        1.43505881])
            # Best 3 indices: 1709, 1771, 2297

            # move data to device
            for key in data:
                data[key] = data[key].to(device, dtype=torch.float)

            # Get homography
            model_start.record()
            delta_hat, _ = model.predict_homography(data)
            model_end.record()
            torch.cuda.synchronize()
            model_time.append(model_start.elapsed_time(model_end))

            # Get ground truth
            delta_gt = data['delta']

            # print(iter, repr(delta_hat.detach().cpu().numpy()))

            # if postprocess:
            #     postprocess_start.record()
            #     homography_refined, delta_hat_refined = model.postprocess(network_output)
            #     postprocess_end.record()
            #     postprocess_time.append(postprocess_start.elapsed_time(postprocess_end))
            #     torch.cuda.synchronize()
            #     delta_hat = delta_hat_refined

            # Calc Mean Average Corner Error
            if self_supervised:
                if torch.is_tensor(delta_gt):
                    delta_gt = delta_gt.detach().cpu().numpy()
                if torch.is_tensor(delta_hat):
                    delta_hat = delta_hat.detach().cpu().numpy()
                mace = np.mean(np.linalg.norm(delta_gt.reshape(-1, 2) -
                                              delta_hat.reshape(-1, 2), axis=-1))

                if log_filepath is not None:
                    with open(log_filepath, 'a') as f:
                        f.write(str(iter) + ',' + str(mace) + '\n')
                batched_mace.append(mace)

            # #####################################################################
            # Visualize warp
            # #####################################################################

            if visualize and self_supervised:

                ###############################################################
                # Imports
                ###############################################################

                import cv2
                import matplotlib.pyplot as plt

                batch_size = data['image_1'].shape[0]
                for idx in range(batch_size):

                    ###############################################################
                    # Retrieve data
                    ###############################################################

                    image_1 = data['image_1'].detach().cpu().numpy()[idx]
                    image_2 = data['image_2'].detach().cpu().numpy()[idx]
                    patch_1 = data['patch_1'].detach().cpu().numpy()[idx]
                    patch_2 = data['patch_2'].detach().cpu().numpy()[idx]
                    corners = data['corners'].detach().cpu().numpy()[idx]

                    ###############################################################
                    # Retrieve mask data if present
                    ###############################################################

                    print(data.keys())
                    if 'patch_1_m' in data:
                        patch_1_m = data['patch_1_m'].detach().cpu().numpy()[idx]
                        patch_1_m = patch_1_m.transpose((1, 2, 0))
                        patch_1_m = np.tile(patch_1_m, (1, 1, 3))
                    if 'patch_2_m' in data:
                        patch_2_m = data['patch_2_m'].detach().cpu().numpy()[idx]
                        patch_2_m = patch_2_m.transpose((1, 2, 0))
                        patch_2_m = np.tile(patch_2_m, (1, 1, 3))

                    ###############################################################
                    # Destandardize
                    ###############################################################

                    image_1 = destandardize(image_1)
                    image_2 = destandardize(image_2)
                    patch_1 = destandardize(patch_1, transpose=True)
                    patch_2 = destandardize(patch_2, transpose=True)
                    image_vis = np.copy(image_2)

                    ###############################################################
                    # Draw rects image
                    ###############################################################

                    image_1 = draw_rect(image_1, corners, 'b')
                    image_2 = draw_rect(image_2, corners, 'b')

                    ###############################################################
                    # Predicted homographies
                    ###############################################################

                    corners_gt = corners - delta_gt[idx].reshape(4, 2)
                    image_vis = draw_rect(image_vis, corners_gt, 'b')

                    # # Orig arch output
                    # delta_orig = np.array([[-26.193892, -22.046888], [-24.314796, -1.6638668], [28.384571, -25.465956],
                    #                        [8.2527275, 4.8997808]])
                    # corners_orig = corners - delta_orig
                    # image_vis = draw_rect(image_vis, corners_orig, 'r')

                    # # AFM arch output
                    # delta_afm = np.array([[-28.005247, -23.422512], [-25.096855, -0.3967885], [26.155422, -25.632349],
                    #                       [8.992535, 5.0495477]])
                    # corners_afm = corners - delta_afm
                    # image_vis = draw_rect(image_vis, corners_afm, 'g')

                    # Orig arch output
                    corners_hat = corners - delta_hat[idx].reshape(-1, 2)
                    image_vis = draw_rect(image_vis, corners_hat, 'r')

                    ###############################################################
                    # Show figure
                    ###############################################################

                    # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(40, 10))
                    # ax1.imshow(image_1)
                    # ax1.set_title('image_1')
                    # ax2.imshow(image_2)
                    # ax2.set_title('image_2')
                    # ax3.imshow(patch_1)
                    # ax3.set_title('patch_1')
                    # ax4.imshow(patch_2)
                    # ax4.set_title('patch_2')
                    # ax5.imshow(image_vis)
                    # ax5.set_title('image_vis')
                    # ax6.imshow(patch_1_m)
                    # ax6.set_title('patch_1_m')
                    # plt.show()
                    #
                    # cv2.imwrite('/data/output/daniel.koguciuk/temp/vis/{}_blob.png'.format(str(iter*batch_size + idx)),
                    #             cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))

                    ###############################################################
                    # GIF: image 1 + image 2 + mask
                    ###############################################################

                    import imageio
                    from src.data.utils import warp_image
                    from src.data.utils import four_point_to_homography

                    c = patch_1.shape[0]
                    corners = np.expand_dims(np.float32([[0, 0], [c, 0], [c, c], [0, c]]), axis=0)
                    homography = four_point_to_homography(corners=corners, deltas=delta_hat[idx].reshape(1, 4, 2),
                                                          crop=False)
                    warped = warp_image(patch_1, homography, target_h=c, target_w=c)

                    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(40, 10))
                    # ax1.imshow(patch_1)
                    # ax2.imshow(patch_2)
                    # ax3.imshow(warped)
                    # plt.show()

                    # Save output
                    gif_1_2_mask_path = '/data/output/daniel.koguciuk/temp/vis/{}_patch_1_2_mask_blob.gif'.format(
                        str(iter * batch_size + idx))

                    # Zhang
                    #imageio.mimsave(gif_1_2_mask_path, [warped, patch_2, patch_1_m], duration=0.5)

                    # Ours
                    patch_1_m_warped = warp_image(patch_1_m, homography, target_h=c, target_w=c)
                    imageio.mimsave(gif_1_2_mask_path, [warped, patch_2, patch_1_m_warped], duration=0.5)

                    np.save('/data/output/daniel.koguciuk/temp/vis/{}_warped.npy'.format(
                        str(iter * batch_size + idx)), warped)
                    np.save('/data/output/daniel.koguciuk/temp/vis/{}_patch_2.npy'.format(
                        str(iter * batch_size + idx)), patch_2)

                    np.save('/data/output/daniel.koguciuk/temp/vis/{}_mask_1.npy'.format(
                        str(iter * batch_size + idx)), patch_1_m_warped)
                    np.save('/data/output/daniel.koguciuk/temp/vis/{}_mask_2.npy'.format(
                        str(iter * batch_size + idx)), patch_2_m)

                    if 'pf' in data:
                        pf = data['pf'].detach().cpu().numpy()[idx]
                        pf = pf.transpose((1, 2, 0))
                        np.save('/data/output/daniel.koguciuk/temp/vis/{}_pf.npy'.format(
                            str(iter * batch_size + idx)), pf)

                    ###############################################################
                    # Commented
                    ###############################################################

                    # if torch.is_tensor(delta_gt):
                    #     delta_gt = delta_gt.detach().cpu().numpy()
                    # if torch.is_tensor(delta_hat):
                    #     delta_hat = delta_hat.detach().cpu().numpy()
                    #
                    # patch_1 = data[patch_keys[0]][0].detach().cpu().numpy().transpose((1, 2, 0))
                    # patch_2 = data[patch_keys[1]][0].detach().cpu().numpy().transpose((1, 2, 0))
                    #
                    # import cv2
                    # import matplotlib.pyplot as plt
                    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(40, 10))
                    # ax1.imshow(np.tile(patch_1, (1, 1, 3)))
                    # ax1.set_title('patch_1')
                    #
                    # # Patch 1 warped with delta_hat
                    # patch_size = patch_1.shape[0]
                    # mat = cv2.getPerspectiveTransform(np.float32([[0, 0], [patch_size, 0], [patch_size, patch_size],
                    #                                               [0, patch_size]]),
                    #                                   np.float32([[0, 0], [patch_size, 0], [patch_size, patch_size],
                    #                                               [0, patch_size]]) + np.float32(delta_hat))
                    # inv_mat = np.linalg.inv(mat)
                    # patch_1_w = np.expand_dims(cv2.warpPerspective(patch_1, inv_mat, dsize=(patch_size, patch_size)), axis=-1)
                    # ax2.imshow(np.tile(patch_1_w, (1, 1, 3)))
                    # ax2.set_title('patch_1 warped_hat')
                    #
                    # # Patch 1 warped with delta_gt
                    # patch_size = patch_1.shape[0]
                    # mat = cv2.getPerspectiveTransform(np.float32([[0, 0], [patch_size, 0], [patch_size, patch_size],
                    #                                               [0, patch_size]]),
                    #                                   np.float32([[0, 0], [patch_size, 0], [patch_size, patch_size],
                    #                                               [0, patch_size]]) + np.float32(delta_gt))
                    # inv_mat = np.linalg.inv(mat)
                    # patch_1_w = np.expand_dims(cv2.warpPerspective(patch_1, inv_mat, dsize=(patch_size, patch_size)), axis=-1)
                    # ax3.imshow(np.tile(patch_1_w, (1, 1, 3)))
                    # ax3.set_title('patch_1 warped_gt')
                    #
                    # ax4.imshow(np.tile(patch_2, (1, 1, 3)))
                    # ax4.set_title('patch_2')
                    # plt.show()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params: {}'.format(pytorch_total_params))

    print("Mean mace: {}".format(np.mean(batched_mace)))

    model_time = model_time[1:]
    avg_model_time = np.mean(model_time)
    print("Mean model time: {}".format(avg_model_time))

    # if postprocess:
    #     postprocess_time = postprocess_time[1:]
    #     avg_postprocess_time = np.mean(postprocess_time)
    #     print("Mean postprocess time: {}".format(avg_postprocess_time))


def main(config_file_path: str, ckpt_file_path: str, batch_size: int, visualize=False, log_filepath=None):

    # Load yaml config file
    with open(config_file_path, 'r') as file:
        config = yaml.full_load(file)

    ###########################################################################
    # Make test data loader
    ###########################################################################

    # Fix numpy seed
    np.random.seed(config['DATA']['SAMPLER']['TEST_SEED'])

    # Dataset fn
    if 'oxford' in config['DATA']['NAME']:
        make_dataloader_fn = make_oxford_dataloader
    elif 'coco' in config['DATA']['NAME']:
        make_dataloader_fn = make_coco_dataloader
    elif 'cifar10' in config['DATA']['NAME']:
        make_dataloader_fn = make_cifar_dataloader
    elif 'flir_adas' in config['DATA']['NAME']:
        make_dataloader_fn = make_flir_adas_dataloader
    else:
        assert False, 'I dont know this dataset yet.'

    # Camera models root
    camera_models_root = (os.path.join(BASE_DIR, config['DATA']['CAMERA_MODELS_ROOT']) if 'CAMERA_MODELS_ROOT' in
                          config['DATA'] is not None else None)

    # Test cache
    test_cache = config['DATA']['DATASET_TEST_CACHE'] if 'DATASET_TEST_CACHE' in config['DATA'] is not None else None

    # Collator
    collator_blob_porosity = config['DATA']['AUGMENT_BLOB_POROSITY'] if 'AUGMENT_BLOB_POROSITY' in config[
        'DATA'] else None
    collator_blobiness = config['DATA']['AUGMENT_BLOBINESS'] if 'AUGMENT_BLOBINESS' in config['DATA'] else None

    # Data sampler mode
    data_sampler_mode = config['DATA']['SAMPLER']['MODE'] if 'MODE' in config['DATA']['SAMPLER'] else None
    data_sampler_frame_dist = config['DATA']['SAMPLER']['PAIR_MAX_FRAME_DIST'] if 'PAIR_MAX_FRAME_DIST'\
                                                                                  in config['DATA']['SAMPLER'] else None

    # Eval dataloader
    eval_dataloader = make_dataloader_fn(dataset_name=config['DATA']['NAME'],
                                         dataset_root=os.path.join(BASE_DIR, config['DATA']['DATASET_ROOT']),
                                         camera_models_root=camera_models_root,
                                         split=os.path.join(BASE_DIR, config['DATA']['TEST_SPLIT']),
                                         transforms=config['DATA']['TEST_TRANSFORM'],
                                         batch_size=batch_size,
                                         samples_per_epoch=config['DATA']['SAMPLER']['TEST_SAMPLES_PER_EPOCH'],
                                         mode=data_sampler_mode,
                                         pair_max_frame_dist=data_sampler_frame_dist,
                                         num_workers=config['DATA']['NUM_WORKERS'],
                                         random_seed=config['DATA']['SAMPLER']['TEST_SEED'],
                                         cache_path=test_cache,
                                         collator_patch_1=config['MODEL']['BACKBONE']['PATCH_KEYS'][0],
                                         collator_patch_2=config['MODEL']['BACKBONE']['PATCH_KEYS'][1],
                                         collator_blob_porosity=collator_blob_porosity,
                                         collator_blobiness=collator_blobiness)

    ###########################################################################
    # Data loaders pickling (for faster debugging)
    ###########################################################################

    # with open('eval_dataloader.pkl', 'wb') as f:
    #     pickle.dump(eval_dataloader, f)
    # exit()

    # with open('eval_dataloader.pkl', 'rb') as f:
    #     eval_dataloader = pickle.load(f)

    ###########################################################################
    # Import and create the model
    ###########################################################################

    # Import model
    backbone_module = importlib.import_module('src.backbones.{}'.format(config['MODEL']['BACKBONE']['NAME']))
    backbone_class_to_call = getattr(backbone_module, 'Model')

    # Create model class
    backbone = backbone_class_to_call(**config['MODEL']['BACKBONE'])

    ###########################################################################
    # Import and create the head
    ###########################################################################

    # Import backbone
    head_module = importlib.import_module('src.heads.{}'.format(config['MODEL']['HEAD']['NAME']))
    head_class_to_call = getattr(head_module, 'Model')

    # Create backbone class
    head = head_class_to_call(backbone, **config['MODEL']['HEAD'])

    ###########################################################################
    # Import and create the head
    ###########################################################################

    model = ModelWrapper(backbone, head)

    ###########################################################################
    # Create training elements
    ###########################################################################

    # Training elements
    if config['SOLVER']['OPTIMIZER'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['SOLVER']['LR'],
                                     betas=(config['SOLVER']['MOMENTUM_1'], config['SOLVER']['MOMENTUM_2']))
    else:
        assert False, 'I do not have this solver implemented yet.'
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['SOLVER']['MILESTONES'],
                                                     gamma=config['SOLVER']['LR_DECAY'])

    try:
        loss_fn = getattr(torch.nn, config['SOLVER']['LOSS'])()
    except:
        loss_fn = config['SOLVER']['LOSS']

    ###########################################################################
    # Checkpoint
    ###########################################################################

    arguments = {"step": 0}
    checkpointer = CheckPointer(model, optimizer, scheduler, config['LOGGING']['DIR'], True, None,
                                device=config['SOLVER']['DEVICE'])
    extra_checkpoint_data = checkpointer.load(f=ckpt_file_path)
    arguments.update(extra_checkpoint_data)

    ###########################################################################
    # NORM OF WEIGHTS
    ###########################################################################

    # def frob(arr):
    #     return np.sqrt(np.sum(np.square(np.abs(arr.reshape(-1)))))
    #
    # # Get feature extractor weigths
    # print('Feature extractor:')
    # fext_weights = model[0].feature_extractor.retrieve_weights()
    # for key in fext_weights:
    #     print(key, fext_weights[key].shape)
    #     #if len(fext_weights[key].shape) == 4 or 'weight' not in key:
    #     #if len(fext_weights[key].shape) == 4 or 'weight' not in key:
    #     #    continue
    #     print(key, ',', frob(fext_weights[key].cpu().detach().numpy()))
    #
    # print('ResNet:')
    # fext_weights = model[0].retrieve_weights()
    # for key in fext_weights:
    #     #print(key, fext_weights[key].shape)
    #     #if len(fext_weights[key].shape) == 4 or 'bn' not in key or 'weight' not in key:
    #     if len(fext_weights[key].shape) == 4:
    #         continue
    #     print(key, ',', frob(fext_weights[key].cpu().detach().numpy()))
    #
    # exit()

    ###########################################################################
    # Do evaluate
    ###########################################################################

    evaluate(model=model, eval_dataloader=eval_dataloader, loss_fn=loss_fn, device=config['SOLVER']['DEVICE'],
             patch_keys=config['MODEL']['BACKBONE']['PATCH_KEYS'], visualize=visualize,
             self_supervised=(data_sampler_mode is None or data_sampler_mode == 'single'),
             postprocess=(config['MODEL']['BACKBONE']['NAME'] == 'Rethinking'),
             log_filepath=log_filepath)
    print('DONE!')


if __name__ == "__main__":

    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Config file with learning settings')
    parser.add_argument('--ckpt', type=str, required=True, help='Model path')
    parser.add_argument('--batch_size', type=int, required=False, default=1, help='Test batch size')
    parser.add_argument('--vis', action='store_true', required=False, help='Should I produce visualization?')
    parser.add_argument('--log', type=str, required=False, help='log filepath')
    args = parser.parse_args()

    # Call main
    main(args.config_file, args.ckpt, args.batch_size, bool(args.vis), args.log)
