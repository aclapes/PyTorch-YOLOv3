from collections import OrderedDict

# def parse_model_config(path):
#     """Parses the yolo-v3 layer configuration file and returns module definitions"""
#     file = open(path, 'r')
#     lines = file.read().split('\n')
#     lines = [x for x in lines if x and not x.startswith('#')]
#     lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
#     module_defs = []
#     for line in lines:
#         if line.startswith('['): # This marks the start of a new block
#             module_defs.append({})
#             module_defs[-1]['type'] = line[1:-1].rstrip()
#             if module_defs[-1]['type'] == 'convolutional':
#                 module_defs[-1]['batch_normalize'] = 0
#         else:
#             key, value = line.split("=")
#             value = value.strip()
#             module_defs[-1][key.rstrip()] = value.strip()
#
#     return module_defs
def parse_model_cfg(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    hyperparams = {}
    module_defs = OrderedDict()
    sx = -1
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_key = line.rstrip()[1:-1].split('@')

            if module_key[0] != 'net':
                module_type = module_key[0]
                sx = module_key[1] if len(module_key) == 2 else '0'
                module_defs.setdefault(sx, []).append({})
                module_defs[sx][-1]['type'] = module_type
                if module_type == 'convolutional' or module_type == 'rconvolutional':
                    module_defs[sx][-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, value = [x.strip() for x in line.split("=")]
            if len(module_defs) > 0:
                module_defs[sx][-1][key] = value
            else:
                hyperparams[key] = value

    return hyperparams, module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def parse_normalization(line):
    if line:
        im_norm_split = line.split(':')
        img_norm = im_norm_split[0]
        if len(im_norm_split) == 0:
            return img_norm
        else:
            img_rng = [float(x) for x in im_norm_split[1].split(',')]
            return img_norm, img_rng
    else:
        return None, None


