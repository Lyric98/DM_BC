import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import wandb
import pyiqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # wandb
    if opt["enable_wandb"]:
        project_name = opt["wandb"]["project"]
        print(f"wandb project: {project_name}")
        wandb.init(project=project_name)

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for i, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        #tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    wandb.log({
                        "epoch": current_epoch,
                        "step": current_step,
                        **logs})


                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_lpips = 0.0
                    avg_clip_iqa = 0.0
                    avg_musiq = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)
                        avg_ssim += Metrics.calculate_ssim(sr_img, hr_img)
                        avg_lpips += Metrics.calculate_lpips(sr_img, hr_img)
                        avg_clip_iqa += Metrics.calculate_clip_iqa(sr_img)
                        avg_musiq += Metrics.calculate_musiq(sr_img)

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    avg_lpips /= idx
                    avg_clip_iqa /= idx
                    avg_musiq /= idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    # logger.info('# Validation # PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
                    logger.info('# Validation # PSNR: {:.4e}, SSIM: {:.4e}, LPIPS: {:.4e}, CLIP IQA: {:.4e}, MUSIQ: {:.4e}'.format(
                        avg_psnr, avg_ssim, avg_lpips, avg_clip_iqa, avg_musiq))
                    logger_val = logging.getLogger('val')  # validation logger
                    # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
                    #     current_epoch, current_step, avg_psnr, avg_ssim))
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}, lpips: {:.4e}, clip_iqa: {:.4e}, musiq: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips, avg_clip_iqa, avg_musiq))
                    
                    # Log metrics to wandb
                    wandb.log({
                        "epoch": current_epoch,
                        "step": current_step,
                        "avg_psnr": avg_psnr,
                        "avg_ssim": avg_ssim,
                        "avg_lpips": avg_lpips,
                        "avg_clip_iqa": avg_clip_iqa,
                        "avg_musiq": avg_musiq
                    })
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        avg_clip_iqa = 0.0
        avg_musiq = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_lpips = Metrics.calculate_lpips(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_clip_iqa = Metrics.calculate_clip_iqa(Metrics.tensor2img(visuals['SR'][-1]))
            eval_musiq = Metrics.calculate_musiq(Metrics.tensor2img(visuals['SR'][-1]))

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            avg_lpips += eval_lpips
            avg_clip_iqa += eval_clip_iqa
            avg_musiq += eval_musiq

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_lpips /= idx
        avg_clip_iqa /= idx
        avg_musiq /= idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info('# Validation # LPIPS: {:.4e}'.format(avg_lpips))
        logger.info('# Validation # CLIP IQA: {:.4e}'.format(avg_clip_iqa))
        logger.info('# Validation # MUSIQ: {:.4e}'.format(avg_musiq))

        logger_val = logging.getLogger('val')  # validation logger
        # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
        #     current_epoch, current_step, avg_psnr, avg_ssim))
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}, lpips: {:.4e}, clip_iqa: {:.4e}, musiq: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips, avg_clip_iqa, avg_musiq))
