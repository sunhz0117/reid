from importlib import import_module
from torchvision import transforms
from utils.random_erasing import RandomErasing
from data.sampler import RandomSampler, IdentitySampler
from torch.utils.data import dataloader
import numpy as np

from IPython import embed

class Data:
    def __init__(self, args):

        train_list = [
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if args.random_erasing:
            train_list.append(RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)
        self.train_transform = train_transform

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = test_transform

        if args.data_train in ['Market1501']:
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, 'train')
            # a = self.trainset[1]
            self.train_loader = dataloader.DataLoader(self.trainset,
                        sampler=RandomSampler(self.trainset, args.batchid, batch_image=args.batchimage),
                        # shuffle=True,
                        batch_size=args.batchid * args.batchimage,
                        num_workers=args.nThread)
        elif args.data_train in ['SYSU']:
            module_train = import_module('data.' + args.data_train.lower())
            data_path = args.datadir

            self.trainset = getattr(module_train, args.data_train)(data_path, train_transform)
            color_pos, thermal_pos = module_train.GenIdx(self.trainset.train_color_label, self.trainset.train_thermal_label)

            self.train_loader = dataloader.DataLoader(self.trainset,
                        sampler=IdentitySampler(self.trainset.train_color_label,
                            self.trainset.train_thermal_label, color_pos, thermal_pos, args.batchimage, args.batchid),
                        # shuffle=True,
                        batch_size=args.batchid * args.batchimage,
                        num_workers=args.nThread)
            self.trainset.cIndex = self.train_loader.sampler.index1  # RGB index
            self.trainset.tIndex = self.train_loader.sampler.index2  # IR index

            # a = self.trainset[10]
            # embed()

        else:
            self.train_loader = None

        # for get sysu attribute
        # args.data_test = 'SYSU'

        if args.data_test in ['Market1501']:
            module = import_module('data.' + args.data_test.lower())
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')

            print("  Dataset statistics: {}".format(args.data_test))
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print('  train    | {:5d} | {:8d}'.format(len(self.trainset.unique_ids), len(self.trainset)))
            print('  ------------------------------')
            print("  query    | {:5d} | {:8d}".format(len(self.queryset.unique_ids), len(self.queryset)))
            print("  gallery  | {:5d} | {:8d}".format(len(self.testset.unique_ids), len(self.testset)))
            print("  ------------------------------")

        elif args.data_test in ['SYSU']:
            module = import_module('data.' + args.data_test.lower())
            data_path = args.datadir

            n_class = len(np.unique(self.trainset.train_color_label))

            # rgb --> ir
            query_img, query_label, query_cam = module.process_query_sysu(data_path, mode=args.mode, img_mode="rgb")
            gall_img, gall_label, gall_cam = module.process_gallery_sysu(data_path, mode=args.mode, img_mode="ir")

            nquery_rgb2ir = len(query_label)
            ngall_rgb2ir = len(gall_label)

            self.queryset = module.TestData(query_img, query_label, transform=test_transform)
            self.testset = module.TestData(gall_img, gall_label, transform=test_transform)

            # ir --> rgb
            query_img, query_label, query_cam = module.process_query_sysu(data_path, mode=args.mode, img_mode="ir")
            gall_img, gall_label, gall_cam = module.process_gallery_sysu(data_path, mode=args.mode, img_mode="rgb")

            nquery_ir2rgb = len(query_label)
            ngall_ir2rgb = len(gall_label)

            self.queryset = module.TestData(query_img, query_label, transform=test_transform)
            self.testset = module.TestData(gall_img, gall_label, transform=test_transform)

            print("  Dataset statistics: {}".format(args.data_test))
            print("  ------------------------------")
            print("  subset       | # ids | # images")
            print("  ------------------------------")
            print('  rgb_train    | {:5d} | {:8d}'.format(n_class, len(self.trainset.train_color_label)))
            print('  ir_train     | {:5d} | {:8d}'.format(n_class, len(self.trainset.train_thermal_label)))
            print('  ------------------------------')
            print("  rgb_query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery_rgb2ir))
            print("  ir_gallery   | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall_rgb2ir))
            print("  ------------------------------")
            print("  ir_query     | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery_ir2rgb))
            print("  rgb_gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall_ir2rgb))
            print("  ------------------------------")

            # for get sysu attribute
            # data_path = "/home/zzz/pytorch/ECNU_TXD/shz/data/sysu"
            # args.mode = "all"
            # module = import_module('data.sysu')
            # self.testset = module.SYSU_INFERENCE(data_path, test_transform)
            # self.queryset = module.SYSU_INFERENCE(data_path, test_transform)

        else:
            raise Exception()

        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=args.batchtest, num_workers=args.nThread)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=args.batchtest, num_workers=args.nThread)

        self.args = args

    # only for SYSU
    def refresh_query(self, img_mode):
        if self.args.data_test in ['Market1501']:
            raise Exception()

        module = import_module('data.' + self.args.data_test.lower())
        data_path = self.args.datadir

        query_img, query_label, query_cam = module.process_query_sysu(data_path, mode=self.args.mode, img_mode=img_mode)

        self.queryset = module.TestData(query_img, query_label, transform=self.test_transform)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=self.args.batchtest, num_workers=self.args.nThread)

    # only for SYSU
    def refresh_gallery(self, img_mode, trial):
        if self.args.data_test in ['Market1501']:
            raise Exception()

        module = import_module('data.' + self.args.data_test.lower())
        data_path = self.args.datadir

        gall_img, gall_label, gall_cam = module.process_gallery_sysu(data_path, mode=self.args.mode, img_mode=img_mode, trial=trial)

        self.testset = module.TestData(gall_img, gall_label, transform=self.test_transform)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=self.args.batchtest, num_workers=self.args.nThread)

