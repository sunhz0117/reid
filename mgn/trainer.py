import os
import torch
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap, eval_sysu
from utils.re_ranking import re_ranking
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from IPython import embed
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torchvision

class Trainer():
    def __init__(self, args, model, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.testset
        self.queryset = loader.queryset
        self.loader = loader

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.lr = 0.
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()

        if self.args.model == "MGN":
            for batch, (inputs, labels, attribute) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                attribute = attribute.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels, attribute)
                loss.backward()
                self.optimizer.step()

                self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                    epoch, self.args.epochs,
                    batch + 1, len(self.train_loader),
                    self.loss.display_loss(batch)),
                    end='' if batch+1 != len(self.train_loader) else '\n')

            self.loss.end_log(len(self.train_loader))

        elif self.args.model == "BIMGN":
            for batch, (rgb_img, ir_img, rgb_label, ir_label, rgb_attribute, ir_attribute) in enumerate(self.train_loader):
                rgb_img = rgb_img.to(self.device)
                ir_img = ir_img.to(self.device)

                labels = torch.cat((rgb_label, ir_label), 0).to(self.device)
                attribute = torch.cat((rgb_attribute, ir_attribute), 0).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(rgb_img, ir_img)
                loss = self.loss(outputs, labels, attribute)
                loss.backward()
                self.optimizer.step()

                self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                    epoch, self.args.epochs,
                    batch + 1, len(self.train_loader),
                    self.loss.display_loss(batch)),
                    end='' if batch + 1 != len(self.train_loader) else '\n')

            self.loss.end_log(len(self.train_loader))

        else:
            raise Exception()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        # for get sysu attribute
        # self.get_attribute(self.test_loader)

        if self.args.model == "MGN":
            self.ckpt.add_log(torch.zeros(1, 7))

            attr_acc_dict = self.eval_attribute(self.test_loader)
            attr_acc = np.mean(list(attr_acc_dict.values()))

            qf = self.extract_feature(self.query_loader).numpy()
            gf = self.extract_feature(self.test_loader).numpy()

            if self.args.re_rank:
                q_g_dist = np.dot(qf, np.transpose(gf))
                q_q_dist = np.dot(qf, np.transpose(qf))
                g_g_dist = np.dot(gf, np.transpose(gf))
                dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
            else:
                dist = cdist(qf, gf)
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            self.ckpt.log[-1, 0] = m_ap
            self.ckpt.log[-1, 1] = r[0]
            self.ckpt.log[-1, 2] = r[2]
            self.ckpt.log[-1, 3] = r[4]
            self.ckpt.log[-1, 4] = r[9]
            self.ckpt.log[-1, 5] = r[19]
            self.ckpt.log[-1, 6] = attr_acc
            best = self.ckpt.log.max(0)
            self.ckpt.write_log(
                '[INFO] mAP: {:.4f} rank(1,3,5,10,20): {:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f} attr: {:.4f} (Best: {:.4f} @epoch {})'.format(
                    m_ap,
                    r[0], r[2], r[4], r[9], r[19],
                    attr_acc,
                    best[0][0],
                    (best[1][0] + 1) * self.args.test_every
                )
            )

            if not self.args.test_only:
                self.ckpt.save(self, epoch, is_best=((best[1][0] + 1) * self.args.test_every == epoch))

        elif self.args.model == "BIMGN":
            self.ckpt.add_log(torch.zeros(1, 14))

            # rgb --> ir
            print('RGB --> IR (Trials: {})'.format(self.args.trial))
            self.loader.refresh_query(img_mode="rgb")
            self.queryset, self.query_loader = self.loader.queryset, self.loader.query_loader

            # attr_acc_dict = self.eval_attribute(self.query_loader, img_mode="rgb")
            # attr_acc = np.mean(list(attr_acc_dict.values()))
            attr_acc = 0

            qf = self.extract_feature(self.query_loader, img_mode="rgb").numpy()
            all_cmc = np.zeros(100)
            all_mAP, all_mINP = 0, 0
            for trial in range(self.args.trial):
                self.loader.refresh_gallery(img_mode="ir", trial=trial)
                self.testset, self.test_loader = self.loader.testset, self.loader.test_loader
                gf = self.extract_feature(self.test_loader, img_mode="ir").numpy()

                if self.args.re_rank:
                    q_g_dist = np.dot(qf, np.transpose(gf))
                    q_q_dist = np.dot(qf, np.transpose(qf))
                    g_g_dist = np.dot(gf, np.transpose(gf))
                    dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
                else:
                    dist = cdist(qf, gf)
                r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                        separate_camera_set=False,
                        single_gallery_shot=False,
                        first_match_break=True)
                m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

                n_cmc, n_map, n_minp =  eval_sysu(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)
                # print('{} {} {} {} {} {} {}'.format(n_map, n_minp, n_cmc[0], n_cmc[2], n_cmc[4], n_cmc[9], n_cmc[19]))

                all_cmc = all_cmc + r
                all_mAP = all_mAP + m_ap
                # all_mINP = all_mINP + 0

            self.ckpt.log[-1, 0] = m_ap / self.args.trial
            self.ckpt.log[-1, 1] = all_cmc[0] / self.args.trial
            self.ckpt.log[-1, 2] = all_cmc[2] / self.args.trial
            self.ckpt.log[-1, 3] = all_cmc[4] / self.args.trial
            self.ckpt.log[-1, 4] = all_cmc[9] / self.args.trial
            self.ckpt.log[-1, 5] = all_cmc[19] / self.args.trial
            self.ckpt.log[-1, 6] = attr_acc
            best = self.ckpt.log.max(0)

            self.ckpt.write_log(
                '[INFO] mAP: {:.4f} rank(1,3,5,10,20): {:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f} attr: {:.4f} (Best: {:.4f} @epoch {})'.format(
                    m_ap,
                    r[0], r[2], r[4], r[9], r[19],
                    attr_acc,
                    best[0][0],
                    (best[1][0] + 1) * self.args.test_every
                )
            )

            if not self.args.test_only:
                self.ckpt.save(self, epoch, is_best=((best[1][0] + 1) * self.args.test_every == epoch), comment="RGB2IR")

            # ir --> rgb
            print('IR --> RGB (Trials: {})'.format(self.args.trial))
            self.loader.refresh_query(img_mode="ir")
            self.queryset, self.query_loader = self.loader.queryset, self.loader.query_loader

            qf = self.extract_feature(self.query_loader, img_mode="ir").numpy()
            all_cmc = np.zeros(100)
            all_mAP, all_mINP = 0, 0
            for trial in range(self.args.trial):
                self.loader.refresh_gallery(img_mode="rgb", trial=trial)
                self.testset, self.test_loader = self.loader.testset, self.loader.test_loader
                gf = self.extract_feature(self.test_loader, img_mode="rgb").numpy()

                if self.args.re_rank:
                    q_g_dist = np.dot(qf, np.transpose(gf))
                    q_q_dist = np.dot(qf, np.transpose(qf))
                    g_g_dist = np.dot(gf, np.transpose(gf))
                    dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
                else:
                    dist = cdist(qf, gf)
                r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                        separate_camera_set=False,
                        single_gallery_shot=False,
                        first_match_break=True)
                m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

                n_cmc, n_map, n_minp =  eval_sysu(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)
                # print('{} {} {} {} {} {} {}'.format(n_map, n_minp, n_cmc[0], n_cmc[2], n_cmc[4], n_cmc[9], n_cmc[19]))

                all_cmc = all_cmc + r
                all_mAP = all_mAP + m_ap
                # all_mINP = all_mINP + 0

            self.ckpt.log[-1, 7] = m_ap / self.args.trial
            self.ckpt.log[-1, 8] = all_cmc[0] / self.args.trial
            self.ckpt.log[-1, 9] = all_cmc[2] / self.args.trial
            self.ckpt.log[-1, 10] = all_cmc[4] / self.args.trial
            self.ckpt.log[-1, 11] = all_cmc[9] / self.args.trial
            self.ckpt.log[-1, 12] = all_cmc[19] / self.args.trial
            self.ckpt.log[-1, 13] = attr_acc
            best = self.ckpt.log.max(0)
            self.ckpt.write_log(
                '[INFO] mAP: {:.4f} rank(1,3,5,10,20): {:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f} attr: {:.4f} (Best: {:.4f} @epoch {})'.format(
                    m_ap / self.args.trial,
                    all_cmc[0], all_cmc[2], all_cmc[4], all_cmc[9], all_cmc[19],
                    attr_acc,
                    best[0][7],
                    (best[1][7] + 1) * self.args.test_every
                )
            )

            if not self.args.test_only:
                self.ckpt.save(self, epoch, is_best=((best[1][7] + 1) * self.args.test_every == epoch), comment="IR2RGB")

        else:
            raise Exception()



    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
        return inputs.index_select(3,inv_idx)

    def extract_feature(self, loader, img_mode=""):
        features = torch.FloatTensor()
        for (inputs, labels, attribute) in loader:
            ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
            for i in range(2):
                if i==1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to(self.device)
                if img_mode == "rgb":
                    outputs = self.model(input_img, None)
                elif img_mode == "ir":
                    outputs = self.model(None, input_img)
                else:
                    outputs = self.model(input_img)
                f = outputs[0].data.cpu()

                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
        return features

    def eval_attribute(self, loader, img_mode=""):
        p1_attr, p2_attr, p3_attr = [], [], []
        gt_attr = []
        img_path = loader.dataset.imgs

        for (inputs, labels, attribute) in tqdm(loader):
            # if visualize cannot be set range(2)
            for i in range(1):
                if i==1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to(self.device)
                if img_mode == "rgb":
                    outputs = self.model(input_img, None)
                elif img_mode == "ir":
                    outputs = self.model(None, input_img)
                else:
                    outputs = self.model(input_img)
                attr_pred = outputs[12:15]
                p1, p2, p3 = attr_pred
                p1_attr.append(p1.data.cpu())
                p2_attr.append(p2.data.cpu())
                p3_attr.append(p3.data.cpu())
                gt_attr.append(attribute)

        p1_attr, p2_attr, p3_attr, gt_attr = torch.cat(p1_attr), torch.cat(p2_attr), torch.cat(p3_attr), torch.cat(gt_attr)

        # pred = p1_attr
        # if self.args.vis_attr != "":
        #     self.visualize(img_path, pred, gt_attr)

        pred = p1_attr
        acc_dict1 = {
            "gender": accuracy_score(gt_attr[:, 0], pred[:, 0:2].argmax(-1)),
            "hair": accuracy_score(gt_attr[:, 1], pred[:, 2:4].argmax(-1)),
            "up": accuracy_score(gt_attr[:, 2], pred[:, 4:6].argmax(-1)),
            "down": accuracy_score(gt_attr[:, 3], pred[:, 6:8].argmax(-1)),
            "clothes": accuracy_score(gt_attr[:, 4], pred[:, 8:10].argmax(-1)),
            "hat": accuracy_score(gt_attr[:, 5], pred[:, 10:12].argmax(-1)),
            "backpack": accuracy_score(gt_attr[:, 6], pred[:, 12:14].argmax(-1)),
            "bag": accuracy_score(gt_attr[:, 7], pred[:, 14:16].argmax(-1)),
            "handbag": accuracy_score(gt_attr[:, 8], pred[:, 16:18].argmax(-1)),
            "age": accuracy_score(gt_attr[:, 9], pred[:, 18:22].argmax(-1)),
            "upcolor": accuracy_score(gt_attr[:, 10], pred[:, 22:30].argmax(-1)),
            "downcolor": accuracy_score(gt_attr[:, 11], pred[:, 30:39].argmax(-1)),
        }

        pred = p2_attr
        acc_dict2 = {
            "gender": accuracy_score(gt_attr[:, 0], pred[:, 0:2].argmax(-1)),
            "hair": accuracy_score(gt_attr[:, 1], pred[:, 2:4].argmax(-1)),
            "up": accuracy_score(gt_attr[:, 2], pred[:, 4:6].argmax(-1)),
            "down": accuracy_score(gt_attr[:, 3], pred[:, 6:8].argmax(-1)),
            "clothes": accuracy_score(gt_attr[:, 4], pred[:, 8:10].argmax(-1)),
            "hat": accuracy_score(gt_attr[:, 5], pred[:, 10:12].argmax(-1)),
            "backpack": accuracy_score(gt_attr[:, 6], pred[:, 12:14].argmax(-1)),
            "bag": accuracy_score(gt_attr[:, 7], pred[:, 14:16].argmax(-1)),
            "handbag": accuracy_score(gt_attr[:, 8], pred[:, 16:18].argmax(-1)),
            "age": accuracy_score(gt_attr[:, 9], pred[:, 18:22].argmax(-1)),
            "upcolor": accuracy_score(gt_attr[:, 10], pred[:, 22:30].argmax(-1)),
            "downcolor": accuracy_score(gt_attr[:, 11], pred[:, 30:39].argmax(-1)),
        }

        pred = p3_attr
        acc_dict3 = {
            "gender": accuracy_score(gt_attr[:, 0], pred[:, 0:2].argmax(-1)),
            "hair": accuracy_score(gt_attr[:, 1], pred[:, 2:4].argmax(-1)),
            "up": accuracy_score(gt_attr[:, 2], pred[:, 4:6].argmax(-1)),
            "down": accuracy_score(gt_attr[:, 3], pred[:, 6:8].argmax(-1)),
            "clothes": accuracy_score(gt_attr[:, 4], pred[:, 8:10].argmax(-1)),
            "hat": accuracy_score(gt_attr[:, 5], pred[:, 10:12].argmax(-1)),
            "backpack": accuracy_score(gt_attr[:, 6], pred[:, 12:14].argmax(-1)),
            "bag": accuracy_score(gt_attr[:, 7], pred[:, 14:16].argmax(-1)),
            "handbag": accuracy_score(gt_attr[:, 8], pred[:, 16:18].argmax(-1)),
            "age": accuracy_score(gt_attr[:, 9], pred[:, 18:22].argmax(-1)),
            "upcolor": accuracy_score(gt_attr[:, 10], pred[:, 22:30].argmax(-1)),
            "downcolor": accuracy_score(gt_attr[:, 11], pred[:, 30:39].argmax(-1)),
        }

        acc_dict = {key: np.mean([acc_dict1[key], acc_dict2[key], acc_dict3[key]]) for key in acc_dict1.keys()}
        return acc_dict

    # for get sysu attribute
    def get_attribute(self, loader):
        p1_attr, p2_attr, p3_attr = [], [], []
        label = []
        img_path = loader.dataset.imgs

        for (inputs, labels, attribute) in tqdm(loader):
            input_img = inputs.to(self.device)
            outputs = self.model(input_img)
            attr_pred = outputs[12:15]
            p1, p2, p3 = attr_pred
            p1_attr.append(p1.data.cpu())
            p2_attr.append(p2.data.cpu())
            p3_attr.append(p3.data.cpu())
            label.append(labels)

        p1_attr, p2_attr, p3_attr, label = torch.cat(p1_attr), torch.cat(p2_attr), torch.cat(p3_attr), torch.cat(label)

        assert label.numpy().tolist() == loader.dataset.label
        print("SYSU unique labels: {}".format(len(set(loader.dataset.label))))

        pred = p1_attr
        self.visualize_inference(img_path, pred, loader.dataset.label)

        inference_dict = {
            "gender": pred[:, 0:2].argmax(-1),
            "hair": pred[:, 2:4].argmax(-1),
            "up": pred[:, 4:6].argmax(-1),
            "down": pred[:, 6:8].argmax(-1),
            "clothes": pred[:, 8:10].argmax(-1),
            "hat": pred[:, 10:12].argmax(-1),
            "backpack": pred[:, 12:14].argmax(-1),
            "bag": pred[:, 14:16].argmax(-1),
            "handbag": pred[:, 16:18].argmax(-1),
            "age": pred[:, 18:22].argmax(-1),
            "upcolor": pred[:, 22:30].argmax(-1),
            "downcolor": pred[:, 30:39].argmax(-1),
        }

        with open("data/sysu_attribute_raw.txt", "w")as file:
            for idx, (path, id) in enumerate(zip(img_path, label)):
                file.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                    path, id,
                    inference_dict["gender"][idx],
                    inference_dict["hair"][idx],
                    inference_dict["up"][idx],
                    inference_dict["down"][idx],
                    inference_dict["clothes"][idx],
                    inference_dict["hat"][idx],
                    inference_dict["backpack"][idx],
                    inference_dict["bag"][idx],
                    inference_dict["handbag"][idx],
                    inference_dict["age"][idx],
                    inference_dict["upcolor"][idx],
                    inference_dict["downcolor"][idx],
                ))

        return inference_dict

    def visualize(self, img_list, pred, gt):

        setFont = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)

        gender = ["M", "F"]
        hair = ["short", "long"]
        up = ["long", "short"]
        down = ["long", "short"]
        clothes = ["dress", "pants"]
        hat = ["N", "Y"]
        backpack = ["N", "Y"]
        bag = ["N", "Y"]
        handbag = ["N", "Y"]
        age = ["young", "teenager", "adult", "old"]
        upcolor = ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"]
        downcolor = ["black", "white", "pink", "purple", "yellow", "gray", "blue", "green", "brown"]

        os.makedirs(self.args.vis_attr, exist_ok=True)

        for idx in tqdm(range(len(gt))):
            if sum(gt[idx]) == -12:
                continue
            toImage = Image.new('RGB', (600, 400))
            draw = ImageDraw.Draw(toImage)

            path = img_list[idx]
            with Image.open(path, "r") as img:
                img = img.convert('RGB').resize((self.args.width, self.args.height))
            toImage.paste(img, (10, 10))
            draw.text((180,  10), "gender:    {}/{}".format(gender[gt[idx][0]] if gt[idx][0] >= 0 else "UN",
                                                            gender[pred[idx, 0:2].argmax(-1)]), font = setFont)
            draw.text((180,  30), "hair:      {}/{}".format(hair[gt[idx][1]] if gt[idx][1] >= 0 else "UN",
                                                            hair[pred[idx, 2:4].argmax(-1)]), font = setFont)
            draw.text((180,  50), "up:        {}/{}".format(up[gt[idx][2]] if gt[idx][2] >= 0 else "UN",
                                                            up[pred[idx, 4:6].argmax(-1)]), font = setFont)
            draw.text((180,  70), "down:      {}/{}".format(down[gt[idx][3]] if gt[idx][3] >= 0 else "UN",
                                                            down[pred[idx, 6:8].argmax(-1)]), font = setFont)
            draw.text((180,  90), "clothes:   {}/{}".format(clothes[gt[idx][4]] if gt[idx][4] >= 0 else "UN",
                                                            clothes[pred[idx, 8:10].argmax(-1)]), font = setFont)
            draw.text((180, 110), "hat:       {}/{}".format(hat[gt[idx][5]] if gt[idx][5] >= 0 else "UN",
                                                            hat[pred[idx, 10:12].argmax(-1)]), font = setFont)
            draw.text((180, 130), "backpack:  {}/{}".format(backpack[gt[idx][6]] if gt[idx][6] >= 0 else "UN",
                                                            backpack[pred[idx, 12:14].argmax(-1)]), font = setFont)
            draw.text((180, 150), "bag:       {}/{}".format(bag[gt[idx][7]] if gt[idx][7] >= 0 else "UN",
                                                            bag[pred[idx, 14:16].argmax(-1)]), font = setFont)
            draw.text((180, 170), "handbag:   {}/{}".format(handbag[gt[idx][8]] if gt[idx][8] >= 0 else "UN",
                                                            handbag[pred[idx, 16:18].argmax(-1)]), font = setFont)
            draw.text((180, 190), "age:       {}/{}".format(age[gt[idx][9]] if gt[idx][9] >= 0 else "UN",
                                                            age[pred[idx, 18:22].argmax(-1)]), font = setFont)
            draw.text((180, 210), "upcolor:   {}/{}".format(upcolor[gt[idx][10]] if gt[idx][10] >= 0 else "UN",
                                                            upcolor[pred[idx, 22:30].argmax(-1)]), font = setFont)
            draw.text((180, 230), "downcolor: {}/{}".format(downcolor[gt[idx][11]] if gt[idx][11] >= 0 else "UN",
                                                            downcolor[pred[idx, 30:39].argmax(-1)]), font = setFont)

            toImage.save(os.path.join(self.args.vis_attr, path.split("/")[-1]))

    def visualize_inference(self, img_list, pred, label):

        setFont = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)

        gender = ["M", "F"]
        hair = ["short", "long"]
        up = ["long", "short"]
        down = ["long", "short"]
        clothes = ["dress", "pants"]
        hat = ["N", "Y"]
        backpack = ["N", "Y"]
        bag = ["N", "Y"]
        handbag = ["N", "Y"]
        age = ["young", "teenager", "adult", "old"]
        upcolor = ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"]
        downcolor = ["black", "white", "pink", "purple", "yellow", "gray", "blue", "green", "brown"]

        os.makedirs("sysu_attr", exist_ok=True)

        for idx in tqdm(range(len(pred))):
            toImage = Image.new('RGB', (600, 400))
            draw = ImageDraw.Draw(toImage)

            path = img_list[idx]
            with Image.open(path, "r") as img:
                img = img.convert('RGB').resize((self.args.width, self.args.height))
            toImage.paste(img, (10, 10))
            draw.text((180,  10), "gender:    {}".format(gender[pred[idx, 0:2].argmax(-1)]), font=setFont)
            draw.text((180,  30), "hair:      {}".format(hair[pred[idx, 2:4].argmax(-1)]), font=setFont)
            draw.text((180,  50), "up:        {}".format(up[pred[idx, 4:6].argmax(-1)]), font=setFont)
            draw.text((180,  70), "down:      {}".format(down[pred[idx, 6:8].argmax(-1)]), font=setFont)
            draw.text((180,  90), "clothes:   {}".format(clothes[pred[idx, 8:10].argmax(-1)]), font=setFont)
            draw.text((180, 110), "hat:       {}".format(hat[pred[idx, 10:12].argmax(-1)]), font=setFont)
            draw.text((180, 130), "backpack:  {}".format(backpack[pred[idx, 12:14].argmax(-1)]), font=setFont)
            draw.text((180, 150), "bag:       {}".format(bag[pred[idx, 14:16].argmax(-1)]), font=setFont)
            draw.text((180, 170), "handbag:   {}".format(handbag[pred[idx, 16:18].argmax(-1)]), font=setFont)
            draw.text((180, 190), "age:       {}".format(age[pred[idx, 18:22].argmax(-1)]), font=setFont)
            draw.text((180, 210), "upcolor:   {}".format(upcolor[pred[idx, 22:30].argmax(-1)]), font=setFont)
            draw.text((180, 230), "downcolor: {}".format(downcolor[pred[idx, 30:39].argmax(-1)]), font=setFont)
            draw.text((180, 270), "id:        {}".format(label[idx]), font=setFont)

            toImage.save(os.path.join("sysu_attr", "{}.png".format(idx)))

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs

