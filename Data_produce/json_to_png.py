import argparse
import json
import os
import os.path as osp
import warnings

import PIL.Image
import yaml

from labelme import utils

json_file = 'F:/Graduate/'

list = os.listdir(json_file)#返回指定文件夹包含文件列表
for i in range(0, len(list)):
    path = os.path.join(json_file, list[i])
    filename = list[i][:-5]   #(生成.json)
    if os.path.isfile(path):
        data = json.load(open(path))
        img = utils.img_b64_to_arr(data['imageData'])
        lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

        captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
        lbl_viz = utils.draw_label(lbl, img, captions)
        # 返回最后文件名bo_json
        out_dir = osp.basename(list[i]).replace('.', '_')
        #out_dir = osp.join(osp.dirname(list[i]), out_dir)#文件夹路径和文件名合并
        out_dir = osp.join('F:/Graduate',out_dir)
        #out_dir='G:/bo/'

        if not osp.exists(out_dir):
            os.mkdir(out_dir)#自动创建目录
        '''
        PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
        PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
        '''
        PIL.Image.fromarray(img).save(osp.join(out_dir, '{}.png'.format(filename)))
        PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}_gt.png'.format(filename)))
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.png'.format(filename)))


        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in lbl_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=lbl_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

        print('Saved to: %s' % out_dir)



