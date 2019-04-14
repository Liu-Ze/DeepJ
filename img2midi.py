# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
from generate import Generation
import argparse
from midi_io import *
from util import *
from model import DeepJ


def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    # place365 related
    parser.add_argument('--arch', default='resnet18', help='the architecture to use')
    parser.add_argument('--fname', default='resnet18', help='the image path to process')
    # deepj related
    parser.add_argument('--model', help='Path to DeepJ model file')
    parser.add_argument('--length', default=5000, type=int, help='Length of generation')
    parser.add_argument('--style', default=None, type=int, nargs=4,
                        help='Specify the styles to mix together. By default will generate all possible styles.')
    parser.add_argument('--temperature', default=0.9, type=float, help='Temperature of generation')
    parser.add_argument('--beam', default=1, type=int, help='Beam size')
    parser.add_argument('--adaptive', default=False, action='store_true', help='Adaptive temperature')
    parser.add_argument('--synth', default=False, action='store_true', help='Synthesize output in MP3')
    args = parser.parse_args()

    arch = args.arch
    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # load the test image
    img_name = args.fname
    if not os.access(img_name, os.W_OK):
        img_url = 'http://places.csail.mit.edu/demo/' + img_name
        os.system('wget ' + img_url)

    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    print('{} prediction on {}'.format(arch, img_name))
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))


#########################################################################################
    if args.style:
        styles = np.array([v for v in args.style])
        styles = [styles / styles.sum()]
    else:
        # todo
        pass

    print('=== Loading Model ===')
    print('Path: {}'.format(args.model))
    print('Temperature: {}'.format(args.temperature))
    print('Beam: {}'.format(args.beam))
    print('Adaptive Temperature: {}'.format(args.adaptive))
    print('Styles: {}'.format(styles))
    settings['force_cpu'] = True

    model = DeepJ()

    if args.model:
        model.load_state_dict(torch.load(args.model))
    else:
        print('WARNING: No model loaded! Please specify model path.')

    print('=== Generating ===')

    for style in styles:
        fname = args.fname + str(list(style))
        print('File: {}'.format(fname))
        generation = Generation(model, style=style, default_temp=args.temperature, beam_size=args.beam,
                                adaptive=args.adaptive)
        generation.export(name=fname, seq_len=args.length)

        if args.synth:
            data = synthesize(os.path.join(SAMPLES_DIR, fname + '.mid'))
            with open(os.path.join(SAMPLES_DIR, fname + '.mp3'), 'wb') as f:
                f.write(data)


if __name__ == '__main__':
    main()