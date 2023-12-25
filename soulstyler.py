import os
import random

# from style.CLIPstyler import getStyleImg
import clip
import cv2
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torchvision import utils as vutils
from torchvision.transforms.functional import adjust_contrast

import style.StyleNet as StyleNet
import style.utils as utils
import utils.config as config
from style.template import imagenet_templates

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

config_path = "./config/refcoco+/test.yaml"
model_pth = "./best_model.pth"

no_transfer_label_dict = {
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle',
    34: 'license plate'
}


def getMaskImg(seg_path, img, config_path, model_pth, sent=None, isMask=False, ):
    if not isMask:
        img_style1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_style2 = img_style1 / 255.0
        img_style3 = np.transpose(img_style2, (2, 0, 1))
        img_style4 = torch.Tensor(img_style3)
        img_style = torch.unsqueeze(img_style4, 0)

        # mask0 = getMask(img, sent, config_path, model_pth)
        mask = cv2.imread(seg_path)
        # img.shape = (1024, 2048, 3)
        # mask.shape = (1024, 2048, 3)
        # mask0.shape = (1024, 2048)

        mask = mask[:, :, 0]
        binary_mask = np.zeros_like(mask)
        for label in no_transfer_label_dict:
            binary_mask = binary_mask + (mask == label)
        binary_mask = binary_mask == 0
        mask0 = binary_mask

        mask1 = np.stack((mask0, mask0, mask0), axis=2)
        mask_img = np.array(mask1 * 255, dtype=np.uint8)
        return mask_img
    else:
        return img


def getCVImg2Torch(img):
    img_style1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_style2 = img_style1 / 255.0
    img_style3 = np.transpose(img_style2, (2, 0, 1))
    img_style4 = torch.Tensor(img_style3)
    img_style = torch.unsqueeze(img_style4, 0)
    return img_style


def load_image(img, mode="PLT"):
    if mode == "CV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(img)[:3, :, :].unsqueeze(0)
    return image.to(device)


def squeeze_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)
    image = torch.Tensor(image)
    return image


def img_normalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize(image, device):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize2(image, device):
    image = F.interpolate(image, size=224, mode='bicubic')
    return image


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


def getClipFeature(image, clip_model):
    image = F.interpolate(image, size=224, mode='bicubic')
    image = clip_model.encode_image(image.to(device))
    image = image.mean(axis=0, keepdim=True)
    image /= image.norm(dim=-1, keepdim=True)
    return image


def getVggFeature(image, device, VGG):
    return utils.get_features(img_normalize(image, device), VGG)


def getLoss(text_feature, img_feature):
    return 1 - torch.cosine_similarity(text_feature, img_feature)


def getCropImgAndFeature(img, mask, target, clip_model, size=128, batch=64, pot_part=0.9, sizePose=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img, mask, target = img.to(device), mask.to(device), target.to(device)

    back_crop, pot_crop, pot_aug, extra_pot = [], [], [], []
    cropper = transforms.RandomCrop(size)
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(400)
    ])

    max_iterations = 20000
    iteration = 0

    while len(pot_crop) < batch and iteration < max_iterations:
        iteration += 1
        if sizePose:
            (i, j, h, w) = sizePose
        else:
            (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        mask_crop = transforms.functional.crop(mask, i, j, h, w)
        img_crop = transforms.functional.crop(img, i, j, h, w)
        target_crop = transforms.functional.crop(target, i, j, h, w)

        if int(mask_crop[0].sum()) / (3 * size * size) >= 0.8:
            if len(pot_crop) < batch:
                pot_crop.append(img_crop)
                pot_aug.append(augment(target_crop))

    pot_allCrop = torch.cat(pot_crop, dim=0).to(device)
    pot_all_crop = pot_allCrop
    pot_crop_feature = clip_model.encode_image(clip_normalize2(pot_all_crop, device))

    while len(back_crop) < batch and iteration < max_iterations:
        iteration += 1
        (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        mask_crop = transforms.functional.crop(mask, i, j, h, w)
        img_crop = transforms.functional.crop(img, i, j, h, w)
        target_crop = transforms.functional.crop(target, i, j, h, w)
        if int(mask_crop[0].sum()) / (3 * size * size) < 0.1:
            img_crop_feature = clip_model.encode_image(clip_normalize2(img_crop, device))
            cos = (1 - torch.cosine_similarity(img_crop_feature, pot_crop_feature))
            if torch.numel(cos[cos > 0.12]) > 0.8 * batch:
                back_crop.append([target_crop, img_crop])

    while len(extra_pot) < 0.1 * batch and iteration < max_iterations:
        iteration += 1
        (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        mask_crop = transforms.functional.crop(mask, i, j, h, w)
        img_crop = transforms.functional.crop(img, i, j, h, w)
        target_crop = transforms.functional.crop(target, i, j, h, w)
        if int(mask_crop[0].sum()) / (3 * size * size) < 0.1:
            img_crop_feature = clip_model.encode_image(clip_normalize2(img_crop, device))
            cos = (1 - torch.cosine_similarity(img_crop_feature, pot_crop_feature))
            if torch.numel(cos[cos < 0.06]) > 0.2 * batch:
                extra_pot.append(augment(target_crop))
                pot_aug.append(augment(target_crop))

    return pot_aug, back_crop


def getTotalLoss(args, content_features, text_features, source_features, text_source, target, device, VGG, clip_model,
                 img, mask):
    target_features = utils.get_features(img_normalize(target, device), VGG)
    content_loss = 0
    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224)
    ])

    loss_patch = 0
    img_crop, back_crop = getCropImgAndFeature(img, mask, target, clip_model, size=64, batch=64, pot_part=args.pot_part,
                                               sizePose=None)
    img_crop = torch.cat(img_crop, dim=0)
    img_aug = img_crop

    image_features = clip_model.encode_image(clip_normalize(img_aug, device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    img_direction = (image_features - source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
    loss_temp[loss_temp < args.thresh] = 0
    loss_patch += loss_temp.mean()

    loss_back = 0
    lossToBack = torch.nn.MSELoss()
    for i in back_crop:
        a = i[0]
        b = i[1]
        loss_back += lossToBack(a, b)

    loss_glob = 0

    reg_tv = args.lambda_tv * get_image_prior_losses(target)
    total_loss = args.lambda_patch * loss_patch + args.lambda_c * content_loss + reg_tv + args.lambda_dir * loss_glob + args.lambda_c * loss_back

    detail_loss = {
        "loss_patch": loss_patch,
        "content_loss": content_loss,
        "reg_tv": reg_tv,
        "loss_glob": loss_glob,
        "loss_back": loss_back,
    }

    return total_loss, detail_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def img_denormalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = image * std + mean
    return image


def img_normalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize(image, device):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


def getStyleImg(config_path, content_image, source="a Photo", prompt="a Photo", seed=42, get_total_loss=getTotalLoss,
                mask=None, save_epoch=False, path=''):
    setup_seed(seed)
    args = config.load_cfg_from_cfg_file(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = content_image.to(device)
    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)
    for parameter in VGG.parameters():
        parameter.requires_grad_(False)

    content_features = utils.get_features(img_normalize(content_image, device), VGG)
    target = content_image.clone().requires_grad_(True).to(device)

    style_net = StyleNet.UNet()
    style_net.to(device)

    optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    steps = args.max_step

    output_image = content_image
    m_cont = torch.mean(content_image, dim=(2, 3), keepdim=False).squeeze(0)
    m_cont = [m_cont[0].item(), m_cont[1].item(), m_cont[2].item()]

    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224)
    ])

    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

    with torch.no_grad():
        template_text = compose_text_with_templates(prompt, imagenet_templates)
        tokens = clip.tokenize(template_text).to(device)
        text_features = clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        template_source = compose_text_with_templates(source, imagenet_templates)
        tokens_source = clip.tokenize(template_source).to(device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
        source_features = clip_model.encode_image(clip_normalize(content_image, device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

    for epoch in range(0, steps + 1):
        scheduler.step()
        target = style_net(content_image, use_sigmoid=True).to(device)
        target.requires_grad_(True)
        ###############
        total_loss, detail_loss = get_total_loss(args, content_features, text_features, source_features, text_source,
                                                 target, device, VGG, clip_model, content_image, mask)
        ###############
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('======={}/{}========'.format(epoch, steps + 1))
            print('Total loss: ', total_loss.item())
            for k, v in detail_loss.items():
                print(f"{k}:{v}")
            if save_epoch:
                output_image = target.clone()
                output_image = torch.clamp(output_image, 0, 1)
                output_image = adjust_contrast(output_image, 1.5)
                # plt.imshow(utils.im_convert2(output_image))
                # plt.show()

                out_path = f'./outputs/{prompt}_{epoch}.jpg'
                vutils.save_image(
                    output_image,
                    out_path,
                    nrow=1,
                    normalize=True)
                print(f'saved at [{out_path}]')

    output_image = target.clone()
    output_image = torch.clamp(output_image, 0, 1)
    output_image = adjust_contrast(output_image, 1.5)
    output_image = utils.im_convert2(output_image)
    return output_image


def StyleProcess(img_path, seg_path, cris_prompt, style_prompt, seed, save_epoch=False, size=128, pot_part=0.8):
    tmp_cris = cris_prompt.replace('.', '').replace(' ', '_')
    tmp_style = style_prompt.replace('.', '').replace(' ', '_')

    base_path = f'./outputs'
    img_output_image_path = os.path.join(base_path, 'ori_img.png')
    mask_output_image_path = os.path.join(base_path, 'mask_img.png')
    result_output_image_path = os.path.join(base_path, 'result_img.png')
    result_epoch_output_image_path = os.path.join(base_path, 'epoch/')

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # if save_epoch and not os.path.exists(result_epoch_output_image_path):
    #     os.makedirs(result_epoch_output_image_path)

    if os.path.exists(result_output_image_path):
        print(f"File '{result_output_image_path}' already exists. Exiting function.")
        return

    img = cv2.imread(img_path)
    mask = getMaskImg(seg_path, img, config_path, model_pth, cris_prompt, isMask=False)

    img = load_image(img, mode="CV")
    mask = load_image(mask)

    img = img.to(device)
    mask = mask.to(device)

    # show ori image
    # plt.imshow(utils.im_convert2(img))
    # plt.show()

    out_path = f'./outputs/origin.jpg'
    vutils.save_image(
        img,
        out_path,
        nrow=1,
        normalize=True)
    print(f'saved at [{out_path}]')

    # show mask image
    # plt.imshow(utils.im_convert2(mask))
    # plt.show()

    out_path = f'./outputs/mask.jpg'
    vutils.save_image(
        mask,
        out_path,
        nrow=1,
        normalize=True)
    print(f'saved at [{out_path}]')

    # save_image(img, img_output_image_path)
    # save_image(mask, mask_output_image_path)

    output_image = getStyleImg(
        config_path, img, source="a Photo",
        prompt=style_prompt,
        seed=seed,
        get_total_loss=getTotalLoss,
        mask=mask,
        save_epoch=save_epoch,
        path=result_epoch_output_image_path
    ).to(device)

    # result
    # plt.figure(figsize=(20, 20))  # 6，8分别对应宽和高
    # plt.imshow(utils.im_convert2(output_image))
    # plt.show()

    out_path = f'./outputs/output_image.jpg'
    vutils.save_image(
        output_image,
        out_path,
        nrow=1,
        normalize=True)
    print(f'saved at [{out_path}]')


style_prompt_list = [
    {'style': "white wool", 'seed': 2},
    {'style': "a sketch with crayon", 'seed': 3},
    {'style': "oil painting of flowers", 'seed': 7},
    {'style': 'pop art of night city', 'seed': 6},
    {'style': "Starry Night by Vincent van gogh", 'seed': 0},
    {'style': "neon light", 'seed': 5},
    {'style': "mosaic", 'seed': 4},
    {'style': "green crystal", 'seed': 1},
    {'style': "Underwater", 'seed': 0},
    {'style': "fire", 'seed': 0},
    {'style': 'a graffiti style painting', 'seed': 2},
    {'style': 'The great wave off kanagawa by Hokusai', 'seed': 0},
    {'style': 'Wheatfield by Vincent van gogh', 'seed': 2},
    {'style': 'a Photo of white cloud', 'seed': 3},
    {'style': 'a monet style underwater', 'seed': 3},
    {'style': 'A fauvism style painting', 'seed': 2}
]

input_data = [
    {
        'img_path': "./testimg/ship.jpg",
        'cris_prompt': "A white sailboat with three blue sails floating on the sea"
    },
    {
        'img_path': "./testimg/911.jpg",
        'cris_prompt': "a plane"
    },
    {
        'img_path': "./testimg/1.jpg",
        'cris_prompt': "a flower"
    },
    {
        'img_path': "./testimg/house.jpg",
        'cris_prompt': "a house"
    },
    {
        'img_path': "./testimg/people.jpg",
        'cris_prompt': "the face"
    },
    {
        'img_path': "./testimg/Napoleon.jpg",
        'cris_prompt': "a White War Horse"
    },
    {
        'img_path': "./testimg/apple.png",
        'cris_prompt': "a red apple"
    },
    {
        'img_path': "./testimg/bigship.png",
        'cris_prompt': "White Large Luxury Cruise Ship"
    },
    {
        'img_path': "./testimg/car.png",
        'cris_prompt': "White sports car."
    },
    {
        'img_path': "./testimg/lena.png",
        'cris_prompt': "A woman's face."
    },
    {
        'img_path': "./testimg/mountain.png",
        'cris_prompt': "mountain peak"
    },
    {
        'img_path': "./testimg/tjl.jpeg",
        'cris_prompt': "The White House at the Taj Mahal"
    },
    {
        'img_path': "./testimg/man.jpg",
        'cris_prompt': "The Men's face"
    },
]

# StyleProcess(img_path=input_data[0]['img_path'],
#              cris_prompt=input_data[0]['cris_prompt'],
#              style_prompt=style_prompt_list[2]['style'],
#              seed=style_prompt_list[2]['seed'],
#              save_epoch=True,
#              size=128,
#              pot_part=0.8)

StyleProcess(
    img_path='/nfs/s3_common_dataset/cityscapes/leftImg8bit/train/tubingen/tubingen_000061_000019_leftImg8bit.png',
    seg_path='/nfs/s3_common_dataset/cityscapes/gtFine/train/tubingen/tubingen_000061_000019_gtFine_labelIds.png',
    cris_prompt='?',
    style_prompt='snow',
    seed=12345,
    save_epoch=True,
    size=128,
    pot_part=0.8)
