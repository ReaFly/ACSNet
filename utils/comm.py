import models
import torch
import os


def generate_model(opt):
    model = getattr(models, opt.model)(opt.nclasses)
    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    if opt.load_ckpt is not None:
        model_dict = model.state_dict()
        load_ckpt_path = os.path.join('./checkpoints/exp'+str(opt.expID)+'/', opt.load_ckpt + '.pth')
        assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
        print('Loading checkpoint......')
        checkpoint = torch.load(load_ckpt_path)
        new_dict = {k : v for k, v in checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

        print('Done')

    return model
