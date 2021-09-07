import os
import yaml
import pickle
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.data.transforms as transform_module
from src.utils.checkpoint import CheckPointer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_oxford_dataloader(dataset_name: str, dataset_root: str, camera_models_root: str, split: str, transforms: list,
                           batch_size: int, samples_per_epoch: int, mode: str, pair_max_frame_dist: int,
                           num_workers: int, random_seed=None, cache_path=None):
    """


    Args:
        dataset_name (string): Name of the dataset (name of the folder in src.data dir)
        dataset_root (string): Path to the root of the dataset used.
        camera_models_root (string): Path to the directory with camera models.
        split (string): Path to the file with names of sequences to be loaded.
        transforms (list of callables): What transforms apply to the images?
        batch_size (int): Size of the batch.
        samples_per_epoch (int): How many images should I produce in each epoch?
        mode (str): Should I sample single image from the dataset, pair of images from the same sequence but distant
            in time specified by @pair_max_frame_dist, or triplet of frames with corresponding pose, but captured in
            different sequences?
        pair_max_frame_dist (int): Number of frames we search the positive frame in.
        num_workers: Number of data perp workers.
        random_seed (int): If passed will be used as a seed for numpy random generator.
        cache_path (str): Path to the cached dataset sequences.

    Returns:

    """

    # Import data class
    dataset_module = importlib.import_module('src.data.{}.dataset'.format(dataset_name))
    dataset_class_to_call = getattr(dataset_module, 'Dataset')
    sampler_class_to_call = getattr(dataset_module, 'DatasetSampler')

    # Open sequence names to be loaded
    with open(split, 'r') as f:
        seq_names = f.read().strip().split(',')

    # Compose transforms
    transforms_list = []
    for transform in transforms:
        # Get transform class name and params
        t_name = list(transform.keys())[0]
        t_args = transform[t_name]
        # What class to call?
        t_class_to_call = getattr(transform_module, t_name)
        transforms_list.append(t_class_to_call(*t_args))
    composed_transforms = torchvision.transforms.Compose(transforms_list)

    # Call dataset class
    dataset = dataset_class_to_call(dataset_root=dataset_root, camera_models_root=camera_models_root,
                                    seq_names_to_load=seq_names, transforms=composed_transforms, cache_path=cache_path)

    # Call sampler class
    sampler = sampler_class_to_call(data_source=dataset, batch_size=batch_size, samples_per_epoch=samples_per_epoch,
                                    mode=mode, pair_max_frame_dist=pair_max_frame_dist,
                                    random_seed=random_seed)

    # Return dataloader
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)


def make_coco_dataloader(dataset_name: str, dataset_root: str, split: str, transforms: list, batch_size: int,
                         samples_per_epoch: int, mode: str, num_workers: int, random_seed=None, collator_patch_1=None,
                         collator_patch_2=None, collator_blob_porosity=None, collator_blobiness=None, **kwargs):
    """


    Args:
        dataset_name (string): Name of the dataset (name of the folder in src.data dir)
        dataset_root (string): Path to the root of the dataset used.
        camera_models_root (string): Path to the directory with camera models.
        split (string): Path to the file with names of sequences to be loaded.
        transforms (list of callables): What transforms apply to the images?
        batch_size (int): Size of the batch.
        samples_per_epoch (int): How many images should I produce in each epoch?
        mode (str): Should I sample single image from the dataset, pair of images from the same sequence but distant
            in time specified by @pair_max_frame_dist, or triplet of frames with corresponding pose, but captured in
            different sequences?
        pair_max_frame_dist (int): Number of frames we search the positive frame in.
        num_workers: Number of data perp workers.
        random_seed (int): If passed will be used as a seed for numpy random generator.

    Returns:

    """

    # Import data class
    dataset_module = importlib.import_module('src.data.{}.dataset'.format(dataset_name))
    dataset_class_to_call = getattr(dataset_module, 'Dataset')
    sampler_class_to_call = getattr(dataset_module, 'DatasetSampler')

    # Compose transforms
    transforms_list = []
    for transform in transforms:
        # Get transform class name and params
        t_name = list(transform.keys())[0]
        t_args = transform[t_name]
        # What class to call?
        t_class_to_call = getattr(transform_module, t_name)
        # Add seed!
        transforms_list.append(t_class_to_call(*(t_args + [random_seed])))
    composed_transforms = torchvision.transforms.Compose(transforms_list)

    # Call dataset class
    dataset = dataset_class_to_call(dataset_root=split, transforms=composed_transforms)

    # Call sampler class
    sampler = sampler_class_to_call(data_source=dataset, batch_size=batch_size, samples_per_epoch=samples_per_epoch,
                                    mode=mode, random_seed=random_seed)

    # Return dataloader
    if (collator_patch_1 is None or collator_patch_2 is None or collator_blob_porosity is None or
            collator_blobiness is None):
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    else:
        collator = transform_module.CollatorWithBlobs(patch_1_key=collator_patch_1, patch_2_key=collator_patch_2,
                                                      blob_porosity=collator_blob_porosity,
                                                      blobiness=collator_blobiness, random_seed=random_seed)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collator)


def make_clevr_change_dataloader(dataset_name: str, dataset_root: str, split: str, transforms: list, batch_size: int,
                                 samples_per_epoch: int, mode: str, num_workers: int, random_seed=None, **kwargs):
    """


    Args:
        dataset_name (string): Name of the dataset (name of the folder in src.data dir)
        dataset_root (string): Path to the root of the dataset used.
        camera_models_root (string): Path to the directory with camera models.
        split (string): Path to the file with names of sequences to be loaded.
        transforms (list of callables): What transforms apply to the images?
        batch_size (int): Size of the batch.
        samples_per_epoch (int): How many images should I produce in each epoch?
        mode (str): Should I sample single image from the dataset, pair of images from the same sequence but distant
            in time specified by @pair_max_frame_dist, or triplet of frames with corresponding pose, but captured in
            different sequences?
        pair_max_frame_dist (int): Number of frames we search the positive frame in.
        num_workers: Number of data perp workers.
        random_seed (int): If passed will be used as a seed for numpy random generator.

    Returns:

    """

    # Import data class
    dataset_module = importlib.import_module('src.data.{}.dataset'.format(dataset_name))
    dataset_class_to_call = getattr(dataset_module, 'Dataset')
    sampler_class_to_call = getattr(dataset_module, 'DatasetSampler')

    # Compose transforms
    transforms_list = []
    for transform in transforms:
        # Get transform class name and params
        t_name = list(transform.keys())[0]
        t_args = transform[t_name]
        # What class to call?
        t_class_to_call = getattr(transform_module, t_name)
        transforms_list.append(t_class_to_call(*t_args))
    composed_transforms = torchvision.transforms.Compose(transforms_list)

    # Call dataset class
    dataset = dataset_class_to_call(dataset_root=split, transforms=composed_transforms)

    # Call sampler class
    sampler = sampler_class_to_call(data_source=dataset, batch_size=batch_size, samples_per_epoch=samples_per_epoch,
                                    mode=mode, random_seed=random_seed)

    # Return dataloader
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)


def make_cifar_dataloader(dataset_name: str, dataset_root: str, split: str, transforms: list, batch_size: int,
                          samples_per_epoch: int, mode: str, num_workers: int, random_seed=None, collator_patch_1=None,
                          collator_patch_2=None, collator_blob_porosity=None, collator_blobiness=None, **kwargs):

    # Import data class
    dataset_module = importlib.import_module('src.data.{}.dataset'.format(dataset_name))
    dataset_class_to_call = getattr(dataset_module, 'Dataset')

    # Compose transforms
    transforms_list = []
    for transform in transforms:
        # Get transform class name and params
        t_name = list(transform.keys())[0]
        t_args = transform[t_name]
        # What class to call?
        t_class_to_call = getattr(transform_module, t_name)
        # Add seed!
        transforms_list.append(t_class_to_call(*(t_args + [random_seed])))
    composed_transforms = torchvision.transforms.Compose(transforms_list)

    # Call dataset class
    if 'test' in split:
        train = False
    elif 'train' in split:
        train = True
    dataset = dataset_class_to_call(root=dataset_root, train=train, transform=composed_transforms)

    # Return dataloader

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def make_flir_adas_dataloader(dataset_name: str, dataset_root: str, split: str, transforms: list, batch_size: int,
                              samples_per_epoch: int, mode: str, num_workers: int, random_seed=None,
                              collator_patch_1=None, collator_patch_2=None, collator_blob_porosity=None,
                              collator_blobiness=None, **kwargs):
    """


    Args:
        dataset_name (string): Name of the dataset (name of the folder in src.data dir)
        dataset_root (string): Path to the root of the dataset used.
        camera_models_root (string): Path to the directory with camera models.
        split (string): Path to the file with names of sequences to be loaded.
        transforms (list of callables): What transforms apply to the images?
        batch_size (int): Size of the batch.
        samples_per_epoch (int): How many images should I produce in each epoch?
        mode (str): Should I sample single image from the dataset, pair of images from the same sequence but distant
            in time specified by @pair_max_frame_dist, or triplet of frames with corresponding pose, but captured in
            different sequences?
        pair_max_frame_dist (int): Number of frames we search the positive frame in.
        num_workers: Number of data perp workers.
        random_seed (int): If passed will be used as a seed for numpy random generator.

    Returns:

    """

    # Import data class
    dataset_module = importlib.import_module('src.data.{}.dataset'.format(dataset_name))
    dataset_class_to_call = getattr(dataset_module, 'Dataset')
    sampler_class_to_call = getattr(dataset_module, 'DatasetSampler')

    # Compose transforms
    transforms_list = []
    for transform in transforms:
        # Get transform class name and params
        t_name = list(transform.keys())[0]
        t_args = transform[t_name]
        # What class to call?
        t_class_to_call = getattr(transform_module, t_name)
        # Add seed!
        transforms_list.append(t_class_to_call(*(t_args + [random_seed])))
    composed_transforms = torchvision.transforms.Compose(transforms_list)

    # Call dataset class
    dataset = dataset_class_to_call(dataset_root=split, transforms=composed_transforms)

    # Call sampler class
    sampler = sampler_class_to_call(data_source=dataset, batch_size=batch_size, samples_per_epoch=samples_per_epoch,
                                    mode=mode, random_seed=random_seed)

    # Return dataloader
    if (collator_patch_1 is None or collator_patch_2 is None or collator_blob_porosity is None or
            collator_blobiness is None):
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    else:
        collator = transform_module.CollatorWithBlobs(patch_1_key=collator_patch_1, patch_2_key=collator_patch_2,
                                                      blob_porosity=collator_blob_porosity,
                                                      blobiness=collator_blobiness, random_seed=random_seed)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collator)


def train_one_epoch(model: torch.nn.Sequential,
                    train_dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    gradient_clip: float,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    loss_fn: torch.nn.modules.loss._Loss,
                    epoch: int, steps_per_epoch: int, batch_size: int, device: str,
                    checkpointer: CheckPointer, checkpoint_arguments: dict, log_step: int,
                    summary_writer: torch.utils.tensorboard.SummaryWriter,
                    self_supervised=False, log_verbose=False):

    # Training phase
    model.train()

    # Loop for the whole epoch
    for iter_no, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        # Global Step
        step = epoch*steps_per_epoch + iter_no + 1

        # zero the parameter gradients
        optimizer.zero_grad()

        # move data to device
        for key in data:
            data[key] = data[key].to(device, dtype=torch.float)

        # Add summary writer to data
        if step % log_step == 0:
            data['summary_writer'] = summary_writer
            data['summary_writer_step'] = step

        #######################################################################
        # Loss is the MSE between predicted 4pDelta and ground truth 4pDelta
        # Loss is L1 loss, in which case we have to do additional postprocessing
        if (type(loss_fn) == torch.nn.MSELoss or type(loss_fn) == torch.nn.L1Loss or
                type(loss_fn) == torch.nn.SmoothL1Loss):
            ground_truth, network_output, delta_gt, delta_hat = model(data)
            loss = loss_fn(ground_truth, network_output)

        # Triple loss scenario
        elif type(loss_fn) == str and loss_fn == 'CosineDistance':
            ground_truth, network_output, delta_gt, delta_hat = model(data)
            loss = torch.sum(1 - torch.cosine_similarity(ground_truth, network_output, dim=1))

        # Triple loss scenario
        elif type(loss_fn) == str and (loss_fn == 'TripletLoss' or loss_fn == 'iHomE' or loss_fn == 'biHomE'):

            # # Fix fext
            # model[0].feature_extractor.freeze(True)
            #
            # # Calc loss
            # loss, delta_gt, delta_hat = model(data)
            # print('freezed', loss)
            #
            # # Calc gradients
            # loss.backward()
            #
            # # Retrieve gradients
            # gradient_freezed = {}
            # for name, param in model[0].feature_extractor.named_parameters():
            #     if param.grad is not None:
            #         param_norm = param.grad.data
            #         gradient_freezed[name] = param_norm
            # print(gradient_freezed['layer1.0.weight'])
            #
            # # zero the parameter gradients
            # #optimizer.zero_grad()
            #
            # # Unfix fext
            # model[0].feature_extractor.freeze(False)

            # Calc loss
            loss, delta_gt, delta_hat = model(data)
            # print('unfreezed', loss)
            #
            # # Calc gradients
            # loss.backward()
            #
            # # Retrieve gradients
            # gradient_unfreezed = {}
            # for name, param in model[0].feature_extractor.named_parameters():
            #     if param.grad is not None:
            #         param_norm = param.grad.data
            #         gradient_unfreezed[name] = param_norm
            # print(gradient_unfreezed['layer1.0.weight'])
            #
            # print('OK')
            # exit()

        else:
            assert False, "Do not know the loss: " + str(type(loss_fn))
        #######################################################################

        # calc gradients
        loss.backward()

        # Clip gradients if needed
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Optimize
        optimizer.step()
        scheduler.step()

        # Log
        if step % log_step == 0:

            # Calc norm of gradients
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            # Calc Mean Average Corner Error
            if self_supervised:
                mace = np.mean(np.linalg.norm(delta_gt.detach().cpu().numpy().reshape(-1, 2) -
                                              delta_hat.detach().cpu().numpy().reshape(-1, 2), axis=-1))
                summary_writer.add_scalars('mace', {'train': mace}, step)

            # # Get feature extractor weigths
            # fext_weights = model[0].feature_extractor.retrieve_weights()
            # for key in fext_weights:
            #     summary_writer.add_histogram(key, fext_weights[key].reshape(1, -1), global_step=step)
            #
            #     # Manual save
            #     fpath = os.path.join(summary_writer.get_logdir(), key + '.txt')
            #     with open(fpath, 'a') as f:
            #         weight_str = ','.join([str(e) for e in fext_weights[key].reshape(-1).tolist()])
            #         f.write(str(step) + ',' + weight_str + '\n')

            # Save stats
            summary_writer.add_scalars('loss', {'train': loss.item()}, step)
            summary_writer.add_scalars('lr', {'value': scheduler.get_last_lr()[0]}, step)
            summary_writer.add_scalars('g_norm', {'value': total_norm}, step)
            summary_writer.flush()

            # verbose
            if log_verbose:
                print('Epoch: {} iter: {}/{} loss: {}'.format(epoch, iter_no+1, steps_per_epoch, loss.item()))

    # Save state
    checkpoint_arguments['step'] = step
    checkpointer.save("model_{:06d}".format(step), **checkpoint_arguments)


def eval_one_epoch(model: torch.nn.Sequential,
                   test_dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.modules.loss._Loss,
                   epoch: int, steps_per_epoch: int, batch_size: int, device: str,
                   summary_writer: torch.utils.tensorboard.SummaryWriter,
                   self_supervised=False, log_verbose=False):
    # Training phase
    model.eval()

    # Loop for the whole epoch
    batched_loss = []
    batched_mace = []
    with torch.no_grad():
        for iter_no, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

            # move data to device
            for key in data:
                data[key] = data[key].to(device, dtype=torch.float)

            #######################################################################
            # Loss is the MSE between predicted 4pDelta and ground truth 4pDelta
            # Loss is L1 loss, in which case we have to do additional postprocessing
            if (type(loss_fn) == torch.nn.MSELoss or type(loss_fn) == torch.nn.L1Loss or
                    type(loss_fn) == torch.nn.SmoothL1Loss):
                ground_truth, network_output, delta_gt, delta_hat = model(data)
                loss = loss_fn(ground_truth, network_output)

            # Triple loss scenario
            elif type(loss_fn) == str and loss_fn == 'CosineDistance':
                ground_truth, network_output, delta_gt, delta_hat = model(data)
                loss = torch.sum(1 - torch.cosine_similarity(ground_truth, network_output, dim=1))

            # Triple loss scenario
            elif type(loss_fn) == str and (loss_fn == 'TripletLoss' or loss_fn == 'iHomE' or loss_fn == 'biHomE'):
                loss, delta_gt, delta_hat = model(data)

            else:
                assert False, "Do not know the loss: " + str(type(loss_fn))
            #######################################################################

            # Remember loss
            batched_loss.append(loss.item())

            # Calc Mean Average Corner Error
            if self_supervised:
                mace = np.mean(np.linalg.norm(delta_gt.detach().cpu().numpy().reshape(-1, 2) -
                                              delta_hat.detach().cpu().numpy().reshape(-1, 2), axis=-1))
                batched_mace.append(mace)

            # verbose
            if log_verbose:
                print('Epoch: {} iter: {}/{} loss: {}'.format(epoch, iter_no+1, steps_per_epoch, loss.item()))

    # Save state
    summary_writer.add_scalars('loss', {'test': np.mean(batched_loss)}, (epoch + 1) * steps_per_epoch)
    if self_supervised:
        summary_writer.add_scalars('mace', {'test': np.mean(batched_mace)}, (epoch + 1) * steps_per_epoch)
    summary_writer.flush()


def do_train(model: torch.nn.Sequential,
             train_dataloader: torch.utils.data.DataLoader,
             test_dataloader: torch.utils.data.DataLoader,
             optimizer: torch.optim.Optimizer,
             gradient_clip: float,
             scheduler: torch.optim.lr_scheduler._LRScheduler,
             loss_fn: torch.nn.modules.loss._Loss,
             epochs: int, steps_per_epoch: int, batch_size: int, device: str,
             checkpointer: CheckPointer, checkpoint_arguments: dict, log_dir='logs', log_step=1,
             self_supervised=False, log_verbose=False):

    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################

    summary_writer = SummaryWriter(log_dir)

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
    # Training loop
    ###########################################################################

    start_epoch = checkpoint_arguments['step'] // steps_per_epoch
    for epoch in range(start_epoch, epochs):

        # Train part
        print('Training epoch: {}'.format(epoch))
        train_one_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer,
                        gradient_clip=gradient_clip, scheduler=scheduler, loss_fn=loss_fn, epoch=epoch,
                        steps_per_epoch=steps_per_epoch, batch_size=batch_size, device=device,
                        checkpointer=checkpointer, checkpoint_arguments=checkpoint_arguments, log_step=log_step,
                        summary_writer=summary_writer, self_supervised=self_supervised, log_verbose=log_verbose)

        # Test part
        if test_dataloader is not None:
            print('Testing epoch: {}'.format(epoch))
            eval_one_epoch(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, epoch=epoch,
                           steps_per_epoch=steps_per_epoch, batch_size=batch_size, device=device,
                           summary_writer=summary_writer, self_supervised=self_supervised, log_verbose=log_verbose)


def main(config_file_path: str):

    # Load yaml config file
    with open(config_file_path, 'r') as file:
        config = yaml.full_load(file)

    ###########################################################################
    # Make train/test data loaders
    ###########################################################################

    # Dataset fn
    if 'oxford' in config['DATA']['NAME']:
        make_dataloader_fn = make_oxford_dataloader
    elif 'coco' in config['DATA']['NAME']:
        make_dataloader_fn = make_coco_dataloader
    elif 'clevr_change' in config['DATA']['NAME']:
        make_dataloader_fn = make_clevr_change_dataloader
    elif 'flir_adas' in config['DATA']['NAME']:
        make_dataloader_fn = make_flir_adas_dataloader
    else:
        assert False, 'I dont know this dataset yet.'

    # Camera models root
    camera_models_root = (os.path.join(BASE_DIR, config['DATA']['CAMERA_MODELS_ROOT']) if 'CAMERA_MODELS_ROOT' in
                          config['DATA'] is not None else None)

    # Train/test cache
    train_cache = config['DATA']['DATASET_TRAIN_CACHE'] if 'DATASET_TRAIN_CACHE' in config['DATA'] is not None else None
    test_cache = config['DATA']['DATASET_TEST_CACHE'] if 'DATASET_TEST_CACHE' in config['DATA'] is not None else None

    # Collator
    collator_blob_porosity = config['DATA']['AUGMENT_BLOB_POROSITY'] if 'AUGMENT_BLOB_POROSITY' in config[
        'DATA'] else None
    collator_blobiness = config['DATA']['AUGMENT_BLOBINESS'] if 'AUGMENT_BLOBINESS' in config['DATA'] else None

    # Data sampler mode
    data_sampler_mode = config['DATA']['SAMPLER']['MODE'] if 'MODE' in config['DATA']['SAMPLER'] else None
    data_sampler_frame_dist = config['DATA']['SAMPLER']['PAIR_MAX_FRAME_DIST'] if 'PAIR_MAX_FRAME_DIST'\
                                                                                  in config['DATA']['SAMPLER'] else None

    # Train dataloader
    train_dataloader = make_dataloader_fn(dataset_name=config['DATA']['NAME'],
                                          dataset_root=os.path.join(BASE_DIR, config['DATA']['DATASET_ROOT']),
                                          camera_models_root=camera_models_root,
                                          split=os.path.join(BASE_DIR, config['DATA']['TRAIN_SPLIT']),
                                          transforms=config['DATA']['TRANSFORMS'],
                                          batch_size=config['DATA']['SAMPLER']['BATCH_SIZE'],
                                          samples_per_epoch=config['DATA']['SAMPLER']['TRAIN_SAMPLES_PER_EPOCH'],
                                          mode=data_sampler_mode,
                                          pair_max_frame_dist=data_sampler_frame_dist,
                                          num_workers=config['DATA']['NUM_WORKERS'],
                                          random_seed=config['DATA']['SAMPLER']['TRAIN_SEED'],
                                          cache_path=train_cache,
                                          collator_patch_1=config['MODEL']['BACKBONE']['PATCH_KEYS'][0],
                                          collator_patch_2=config['MODEL']['BACKBONE']['PATCH_KEYS'][1],
                                          collator_blob_porosity=collator_blob_porosity,
                                          collator_blobiness=collator_blobiness)

    # Test dataloader
    test_dataloader = None
    if "TEST_SPLIT" in config['DATA']:
        test_dataloader = make_dataloader_fn(dataset_name=config['DATA']['NAME'],
                                             dataset_root=os.path.join(BASE_DIR, config['DATA']['DATASET_ROOT']),
                                             camera_models_root=camera_models_root,
                                             split=os.path.join(BASE_DIR, config['DATA']['TEST_SPLIT']),
                                             transforms=config['DATA']['TRANSFORMS'],
                                             batch_size=config['DATA']['SAMPLER']['BATCH_SIZE'],
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

    # with open('train_dataloader.pkl', 'wb') as f:
    #     pickle.dump(train_dataloader, f)
    # with open('test_dataloader.pkl', 'wb') as f:
    #     pickle.dump(test_dataloader, f)
    # exit()

    # with open('train_dataloader.pkl', 'rb') as f:
    #     train_dataloader = pickle.load(f)
    # with open('test_dataloader.pkl', 'rb') as f:
    #     test_dataloader = pickle.load(f)

    ###########################################################################
    # DATA LOADERS TEST
    ###########################################################################

    # import numpy as np
    # import matplotlib.pyplot as plt
    # for i_batch, sample_batched in enumerate(train_dataloader):
    #     images = sample_batched[0][0][0]
    #
    #     patch_1, patch_2 = np.split(images.numpy(), 2, axis=0)
    #     target = sample_batched[1][0][0].numpy()
    #
    #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(4, 10))
    #     ax1.imshow(np.tile(patch_1.transpose((1, 2, 0)), (1, 1, 3)))
    #     ax1.set_title('patch_1')
    #
    #     import cv2
    #     mat = cv2.getPerspectiveTransform(np.float32([[0, 0], [128, 0], [128, 128], [0, 128]]),
    #                                       np.float32([[0, 0], [128, 0], [128, 128], [0, 128]]) + np.float32(target))
    #     inv_mat = np.linalg.inv(mat)
    #     patch_1_w = np.expand_dims(cv2.warpPerspective(patch_1.transpose((1, 2, 0)), inv_mat, dsize=(128, 128)), axis=-1)
    #     ax2.imshow(np.tile(patch_1_w, (1, 1, 3)))
    #     ax2.set_title('patch_1 warped')
    #
    #     ax3.imshow(np.tile(patch_2.transpose((1, 2, 0)), (1, 1, 3)))
    #     ax3.set_title('patch_2')
    #
    #     patch_2_w = np.expand_dims(cv2.warpPerspective(patch_2.transpose((1, 2, 0)), mat, dsize=(128, 128)), axis=-1)
    #     ax4.imshow(np.tile(patch_2_w, (1, 1, 3)))
    #     ax4.set_title('patch_2 warped')
    #
    #     plt.show()

    ###########################################################################
    # Import and create the backbone
    ###########################################################################

    # Import backbone
    backbone_module = importlib.import_module('src.backbones.{}'.format(config['MODEL']['BACKBONE']['NAME']))
    backbone_class_to_call = getattr(backbone_module, 'Model')

    # Create backbone class
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

    model = torch.nn.Sequential(backbone, head)

    ###########################################################################
    # Create training elements
    ###########################################################################

    # Training elements
    if config['SOLVER']['OPTIMIZER'] == 'Adam':
        l2_reg = float(config['SOLVER']['L2_WEIGHT_DECAY']) if 'L2_WEIGHT_DECAY' in config['SOLVER'] is not None else 0
        optimizer = torch.optim.Adam(model.parameters(), lr=config['SOLVER']['LR'],
                                     betas=(config['SOLVER']['MOMENTUM_1'], config['SOLVER']['MOMENTUM_2']),
                                     weight_decay=l2_reg)
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
    restart_lr = config['SOLVER']['RESTART_LEARNING_RATE'] if 'RESTART_LEARNING_RATE' in config['SOLVER'] is not None else False
    optim_to_load = optimizer
    if restart_lr:
        optim_to_load = None
    checkpointer = CheckPointer(model, optim_to_load, scheduler, config['LOGGING']['DIR'], True, None,
                                device=config['SOLVER']['DEVICE'])
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    ###########################################################################
    # Load pretrained model
    ###########################################################################

    pretrained_model = config['MODEL']['PRETRAINED'] if 'PRETRAINED' in config['MODEL'] is not None else None
    if pretrained_model is not None:
        checkpoint = torch.load(pretrained_model, map_location=torch.device("cpu"))
        model_ = model
        if isinstance(model_, DistributedDataParallel):
            model_ = model.module
        model_.load_state_dict(checkpoint.pop("model"))
        print('Pretrained model loaded!')

    ###########################################################################
    # Do train
    ###########################################################################

    gradient_clip = config['SOLVER']['GRADIENT_CLIP'] if 'GRADIENT_CLIP' in config['SOLVER'] is not None else -1
    do_train(model=model, device=config['SOLVER']['DEVICE'], train_dataloader=train_dataloader,
             test_dataloader=test_dataloader, optimizer=optimizer, gradient_clip=gradient_clip, scheduler=scheduler,
             loss_fn=loss_fn, batch_size=config['DATA']['SAMPLER']['BATCH_SIZE'], epochs=config['SOLVER']['NUM_EPOCHS'],
             steps_per_epoch=(config['DATA']['SAMPLER']['TRAIN_SAMPLES_PER_EPOCH'] //
                              config['DATA']['SAMPLER']['BATCH_SIZE']),
             log_dir=config['LOGGING']['DIR'], log_step=config['LOGGING']['STEP'], checkpointer=checkpointer,
             checkpoint_arguments=arguments, log_verbose=config['LOGGING']['VERBOSE'],
             self_supervised=(data_sampler_mode is None or data_sampler_mode == 'single'))
    print('DONE!')


if __name__ == "__main__":

    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Config file with learning settings')
    args = parser.parse_args()

    # Call main
    main(args.config_file)
