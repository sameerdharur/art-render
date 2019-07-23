import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as utils
import numpy as np
import scipy
import argparse
import copy
import logging
import sys


def get_logger():

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(stdout_handler)
    return logger

logger = get_logger()


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info("Using the device : {}".format(device))


def get_style_image(style):
  
    style_image_map = {
      "1": "impressionism.jpg",
      "2": "postimpressionism.jpg",
      "3": "cubism.jpg",
      "4": "fauvism.jpg",
      "5": "expressionism.jpg",
      "6": "surrealism.jpg",
      "7": "romanticism.jpg",
      "8": "abstractexpressionism.jpg",
      "9": "renaissance.jpg",
      "10": "modern.jpg"
    }
  
    return 'images/' + style_image_map.get(style)


def image_loader(image_name):

    imsize = 512 if torch.cuda.is_available() else 128

    loader = transforms.Compose([
    transforms.Resize((imsize, imsize)), 
    transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def _imshow(tensor, title=None):
  
    image = tensor.cpu().clone()  
    image = image.squeeze(0)     
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def gram_matrix(input):
  
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers_default, style_layers_default):
  
    cnn = copy.deepcopy(cnn)
    
    content_layers=content_layers_default
    style_layers=style_layers_default
    
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0 
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
    
    
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, content_layers_default, style_layers_default, content_weight=1, num_steps=5000,
                       style_weight=1000000):
  
    logger.info('Building the style transfer model.')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img, content_layers_default, style_layers_default)
    optimizer = get_input_optimizer(input_img)

    logger.info('Optimizing the model.')
    run = [0]
    while run[0] <= num_steps:

        def closure():

            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                logger.info("run {}:".format(run))
                logger.info('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img
  
  
class ContentLoss(nn.Module):

    def __init__(self, target,):
      
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
      
      
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

      
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
      

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='style', action='store', required=True, 
                      help='Enter the preferred style of the output image : \
                      1 for Impressionism, \
                      2 for Post-Impressionism, \
                      3 for Cubism, \
                      4 for Fauvism, \
                      5 for Expressionism, \
                      6 for Surrealism, \
                      7 for Romanticism, \
                      8 for Abstract Expressionism, \
                      9 for Renaissance, \
                      10 for Modern Art.')
    parser.add_argument('-i', dest='input_img_path', action='store', required=True, help='The path to the input image, with its full name.')
    args = parser.parse_args()
  
    style = args.style
    input_img_path = args.input_img_path
    
    del args
    
    style_img_path = get_style_image(style)
    logger.info(style_img_path)
    
    style_img = image_loader(style_img_path)
    content_img = image_loader(input_img_path)
    
    logger.info(style_img.size())
    logger.info(content_img.size())

    assert style_img.size() == content_img.size(), "Style and content images need to be of the same size!"
    
    logger.info("Visualizing the style and content images.")
    _imshow(style_img, title='Style Image')
    _imshow(content_img, title='Content Image')
    
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    input_img = content_img.clone()
    
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, content_layers_default, style_layers_default)

    utils.save_image(output, 'output.jpg')
    logger.info("Saved the image successfully to output.jpg")
  

if __name__ == "__main__":
  
    main()
