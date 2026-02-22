import logging
import os
import torch
import torch.nn as nn
import random
import numpy as np
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from loss.supcontrast import SupConLoss
from imgaug import augmenters as iaa
from timm.data.random_erasing import RandomErasing
import torchvision.transforms as T
from torch import autograd


def get_degrated_img_id(weather_img, cfg):
    # 雨雪雾的三种增强在此：需要 from imgaug import augmenters as iaa
    aug_r = iaa.Rain()
    aug_s = iaa.Snowflakes(flake_size=(0.1, 0.7), speed=(0.007, 0.03))
    aug_f = iaa.Fog()
    weather_img_array = weather_img.numpy()
    normalize = T.Compose([
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        # 修改点，雨雪雾图就不要随机擦粗了。以免影响去噪模型学习
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
    ])
    #
    degraded_r1 = aug_r.augment_image(((weather_img_array[0].transpose((1, 2, 0)) * 255).astype(np.uint8)))
    degraded_s1 = aug_s.augment_image(((weather_img_array[1].transpose((1, 2, 0)) * 255).astype(np.uint8)))
    degraded_h1 = aug_f.augment_image(((weather_img_array[2].transpose((1, 2, 0)) * 255).astype(np.uint8)))
    degraded_r2 = aug_r.augment_image(((weather_img_array[0].transpose((1, 2, 0)) * 255).astype(np.uint8)))
    degraded_s2 = aug_s.augment_image(((weather_img_array[1].transpose((1, 2, 0)) * 255).astype(np.uint8)))
    degraded_h2 = aug_f.augment_image(((weather_img_array[2].transpose((1, 2, 0)) * 255).astype(np.uint8)))
    # 有点复杂，嵌套的操作如下：加噪图像先改变下原来的形状，然后转化为tensor,然后从unit8转化类型到float，然后nomalize，然后合并加噪的图像。
    weather_input_1 = torch.stack([normalize(torch.from_numpy(degraded_r1).permute(2, 0, 1).float() / 255.0),
                               normalize(torch.from_numpy(degraded_s1).permute(2, 0, 1).float() / 255.0),
                               normalize(torch.from_numpy(degraded_h1).permute(2, 0, 1).float() / 255.0)], dim=0)
    weather_input_2 = torch.stack([normalize(torch.from_numpy(degraded_r2).permute(2, 0, 1).float() / 255.0),
                               normalize(torch.from_numpy(degraded_s2).permute(2, 0, 1).float() / 255.0),
                               normalize(torch.from_numpy(degraded_h2).permute(2, 0, 1).float() / 255.0)], dim=0)
    return weather_input_1, weather_input_2, normalize(weather_img)
    # #修改点，这里去掉normalize
    # weather_input_1 = torch.stack([(torch.from_numpy(degraded_r1).permute(2, 0, 1).float() / 255.0),
    #                            (torch.from_numpy(degraded_s1).permute(2, 0, 1).float() / 255.0),
    #                            (torch.from_numpy(degraded_h1).permute(2, 0, 1).float() / 255.0)], dim=0)
    # weather_input_2 = torch.stack([(torch.from_numpy(degraded_r2).permute(2, 0, 1).float() / 255.0),
    #                            (torch.from_numpy(degraded_s2).permute(2, 0, 1).float() / 255.0),
    #                            (torch.from_numpy(degraded_h2).permute(2, 0, 1).float() / 255.0)], dim=0)

    # return weather_input_1, weather_input_2, weather_img


def generated_degrated_image(weather_img, device, cfg):
    # 雨雪雾的三种增强在此：需要 from imgaug import augmenters as iaa
    aug_r = iaa.Rain()
    aug_s = iaa.Snowflakes(flake_size=(0.1, 0.7), speed=(0.007, 0.03))
    aug_f = iaa.Fog()
    rain_images = []
    snow_images = []
    fog_images = []
    normalize = T.Compose([
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        # 修改点，雨雪雾图就不要随机擦粗了。以免影响去噪模型学习
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
    ])
    for idx, image in enumerate(weather_img):
        # 将当前图像转换为NumPy数组，并重塑形状
        img_np = image.numpy().transpose((1, 2, 0))
        # 将像素值从浮点数映射到整数范围（0到255）
        img_np = (img_np * 255).astype(np.uint8)
        augmented_image_np_r = aug_r.augment_image(img_np)
        augmented_image_np_s = aug_s.augment_image(img_np)
        augmented_image_np_f = aug_f.augment_image(img_np)

        # 将增强后的NumPy数组转换回PyTorch张量，并重塑形状
        degrade_path = torch.from_numpy(augmented_image_np_r).permute(2, 0, 1).float() / 255.0
        degrade_path = normalize(degrade_path)
        rain_images.append(degrade_path)
        degrade_path = torch.from_numpy(augmented_image_np_s).permute(2, 0, 1).float() / 255.0
        degrade_path = normalize(degrade_path)
        snow_images.append(degrade_path)
        degrade_path = torch.from_numpy(augmented_image_np_f).permute(2, 0, 1).float() / 255.0
        degrade_path = normalize(degrade_path)
        fog_images.append(degrade_path)

        # 将增强后的图像张量堆叠成批量张量
    rain_images = torch.stack(rain_images).cuda().float()
    snow_images = torch.stack(snow_images).cuda().float()
    fog_images = torch.stack(fog_images).cuda().float()
    # print(degraded_images.shape)
    rain_images.to(device), snow_images.to(device), fog_images.to(device)
    return rain_images, snow_images, fog_images


def do_train_stage2(cfg,
                    model,
                    center_criterion,
                    train_loader_stage2,
                    val_loader,
                    optimizer,
                    optimizer_center,
                    optimizer_weather,
                    scheduler,
                    loss_fn,
                    num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    loss_meter_weather = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_weather = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    aug_r = iaa.Rain()
    aug_s = iaa.Snowflakes(flake_size=(0.1, 0.7), speed=(0.007, 0.03))
    aug_f = iaa.Fog()
    for epoch in range(1, epochs + 1):
        if epoch<130:
            continue
        start_time = time.time()
        loss_meter.reset()
        loss_meter_weather.reset()
        acc_meter.reset()
        acc_meter_weather.reset()
        evaluator.reset()
        scheduler.step()
        model.train()

        # with autograd.detect_anomaly():
        for n_iter, (img, vid, target_cam, target_view, weather_img, weather_vid, weather_target_cam, weather_target_view, weather_indices,
                     wea_img) in enumerate(train_loader_stage2):

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            optimizer_weather.zero_grad()
            # 修改点 提前to(device)
            target = vid.to(device)
            weather_vid = weather_vid.to(device)

            normalize = T.Compose([
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                # 修改点，雨雪雾图就不要随机擦粗了。以免影响去噪模型学习
                # RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
            ])

            weather_input_1, weather_input_2, clean_input = get_degrated_img_id(weather_img, cfg)
            weather_input_1 = weather_input_1.to(device)
            weather_input_2 = weather_input_2.to(device)
            clean_input = clean_input.to(device)

            del weather_img

            # Weathernet
            with amp.autocast(enabled=True):
                # 两阶段的weathernet
                if epoch <= 120:
                    # if epoch < 0:
                    # if epoch < cfg.SOLVER.STAGE2.MAX_EPOCHS / 2:
                    _, output, weather_target, _ = model(weather_input_1=weather_input_1, weather_input_2=weather_input_2,
                                                     get_weather_stage1=True)

                    contrast_loss = CE(output, weather_target)
                    weather_aux_loss = contrast_loss
                    # 计算完就删除
                    del _, output, weather_target

                else:

                    restored, output, weather_target = model(weather_input_1=weather_input_1, weather_input_2=weather_input_2,
                                                         get_weather_stage2=True)
                    weathernet_contrast_loss = CE(output, weather_target)

                    l1_loss = l1(restored, clean_input)
                    weather_aux_loss = l1_loss + 0.1 * weathernet_contrast_loss

                    # res_score, res_feat, restored_features = model(x=restored, label=weather_vid)
                    # res_score, res_feat, restored_features = model(x=normalize(restored), label=weather_vid, cam_label=weather_target_cam,
                    #                                                view_label=weather_target_view)
                    res_score, res_feat, restored_features = model(x=restored, label=weather_vid, cam_label=weather_target_cam,
                                                                   view_label=weather_target_view)
                    #
                    # weather_logits = restored_features @ text_features.t().to(torch.float32)
                    # wether_loss包含了IDloss和i2tceloss
                    # weather_loss = loss_fn(res_score, res_feat, weather_vid , weather_target_cam, i2tscore=weather_logits, weather=True)

                    # weather_aux_loss = weather_aux_loss+0.1*weather_loss

                    # weather_aux_loss = weather_aux_loss
                    # if epoch == 9 :
                    #     print(restored)
                    #     print('restored max',restored.max())
                    #     print('restored min',restored.min())
                    #     print('l1_loss',l1_loss)
                    #     print('weathernet_contrast_loss',weathernet_contrast_loss)
                    #   #  print('weather_loss',weather_loss)
                    #     print('let me see see')

                    # 计算完就删除
                    del restored, output, weather_target

            # img在这里to device ，这是因为degraded图像需要先接收cpu上的img
            img = img.to(device)

            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            with ((amp.autocast(enabled=True))):
                # 修改点，若目前属于weather的第二阶段，则加上weather_loss
                score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view)

                # _, _, clean_features = model(x=normalize(clean_input), label=weather_vid, cam_label=weather_target_cam, view_label=weather_target_view)
                _, _, clean_features = model(x=clean_input, label=weather_vid, cam_label=weather_target_cam,
                                             view_label=weather_target_view)

                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits)

                if epoch > 120:

                    # weather_loss 使用 Weathernet，将所有epoch的内容进行去噪后计算reid
                    with torch.no_grad():
                        rain_images, snow_images, fog_images = generated_degrated_image(wea_img, device, cfg)

                        # print('test for big batch_size!!!!!!,now epoch is ',epoch)
                        rain_restored_list = []
                        snow_restored_list = []
                        fog_restored_list = []
                        for i in range(wea_img.size(0)):
                            # 获取当前批次的张量
                            current_img = rain_images[i:i + 1]
                            restored, output, weather_target = model(weather_input_1=current_img, weather_input_2=current_img,
                                                                 get_weather_stage2=True)
                            rain_restored_list.append(restored)
                            current_img = snow_images[i:i + 1]
                            restored, output, weather_target = model(weather_input_1=current_img, weather_input_2=current_img,
                                                                 get_weather_stage2=True)
                            snow_restored_list.append(restored)
                            current_img = fog_images[i:i + 1]
                            restored, output, weather_target = model(weather_input_1=current_img, weather_input_2=current_img,
                                                                 get_weather_stage2=True)
                            fog_restored_list.append(restored)

                            # print('get ',i,'ge restored')
                            # allocated_memory = torch.cuda.memory_allocated()
                            #
                            # print("当前已分配的显存量:", allocated_memory)

                    # 将 restored_list 中的张量堆叠成一个张量
                    rain_restored_list = torch.cat(rain_restored_list, dim=0).detach().to(device)
                    snow_restored_list = torch.cat(snow_restored_list, dim=0).detach().to(device)
                    fog_restored_list = torch.cat(fog_restored_list, dim=0).detach().to(device)

                    rain_res_score, rain_res_feat, rain_restored_features = model(x=rain_restored_list, label=target,
                                                                                  cam_label=target_cam,
                                                                                  view_label=target_view)
                    snow_res_score, snow_res_feat, snow_restored_features = model(x=snow_restored_list, label=target,
                                                                                  cam_label=target_cam,
                                                                                  view_label=target_view)
                    fog_res_score, fog_res_feat, fog_restored_features = model(x=fog_restored_list, label=target,
                                                                               cam_label=target_cam,
                                                                               view_label=target_view)

                    rain_logits = rain_restored_features @ text_features.t().to(torch.float32)
                    snow_logits = snow_restored_features @ text_features.t().to(torch.float32)
                    fog_logits = fog_restored_features @ text_features.t().to(torch.float32)

                    # wether_loss包含了IDloss和i2tceloss
                    rain_loss = loss_fn(rain_res_score, rain_res_feat, target, target_cam, i2tscore=rain_logits)
                    snow_loss = loss_fn(snow_res_score, snow_res_feat, target, target_cam, i2tscore=snow_logits)
                    fog_loss = loss_fn(fog_res_score, fog_res_feat, target, target_cam, i2tscore=fog_logits)

                    l1_features_loss = l1(restored_features, clean_features)
                    weather_aux_loss = weather_aux_loss + l1_features_loss
                    # if epoch >= cfg.SOLVER.STAGE2.MAX_EPOCHS / 2:
                    # selected_features = torch.index_select(image_features, 0, torch.tensor(weather_indices).to(device))
                    loss = loss + rain_loss + snow_loss + fog_loss + 20 * l1_features_loss

            if weather_aux_loss != weather_aux_loss:
                logger.info('weather_aux_loss ==nan :', weather_aux_loss)
                continue
            if loss != loss:
                logger.info('loss ==nan :', loss)
                continue

            scaler.scale(loss).backward(retain_graph=True)
            # scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scaler.scale(weather_aux_loss).backward()
            scaler.step(optimizer_weather)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            if epoch <= 120:
                # if epoch < cfg.SOLVER.STAGE2.MAX_EPOCHS / 2:
                loss_meter.update(loss.item(), img.shape[0])
                loss_meter_weather.update(weather_aux_loss.item(), 3)
                weather_acc = 0.0
            else:
                loss_meter.update(loss.item(), img.shape[0])
                loss_meter_weather.update(weather_aux_loss.item(), 3)
                weather_acc = (fog_logits.max(1)[1] == target).float().mean()
                # print('weather_acc')
            acc_meter.update(acc, 1)
            acc_meter_weather.update(weather_acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss(Weathernet): {:.3f}, Acc: {:.3f}, Acc(Weathernet): {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader_stage2),
                            loss_meter.avg, loss_meter_weather.avg, acc_meter.avg, acc_meter_weather.avg,
                            scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0 and epoch >= 60:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else:
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else:
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

        # 修改weather学习率
        if epoch <= 120:
            lr = 2e-4 * (0.1 ** (epoch // 60))
            for param_group in optimizer_weather.param_groups:
                param_group['lr'] = lr
            print('weathernet 学习率：', lr)
        else:
            lr = 0.0001 * (0.5 ** ((epoch - 120) // 125))
            for param_group in optimizer_weather.param_groups:
                param_group['lr'] = lr
            print('weathernet 学习率：', lr)
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def do_inference_weather(cfg,
                         model,
                         val_loader,
                         num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing for WeatherNet-ReID")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    # normalize = T.Compose([
    #     T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    #     # 修改点，雨雪雾图就不要随机擦粗了。以免影响去噪模型学习
    #     # RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
    # ])
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            # from utils.image_io import save_image_tensor
            # save_image_tensor(img[4:50], '/data1/wangbin/AWDW-ReID/DW-ReID-weathernet/test.png')
            restored = model(weather_input_1=img, weather_input_2=img, get_weather_stage2=True)
            # save_image_tensor(restored[4:50], '/data1/wangbin/AWDW-ReID/DW-ReID-weathernet/restored.png')
            feat = model(restored, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

