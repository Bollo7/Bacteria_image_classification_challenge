from PIL import Image
from glob import glob
import os
import itertools
import numpy as np
import torchvision.transforms.functional as TF


def img_combiner(img_dir):
	dic = dict()
	for file in img_dir:
		file_c = os.path.basename(file)
		grp = dic.get(file_c[:5], [])
		grp.append(file)
		dic[file_c[:5]] = grp

	def stack(cache=dic):
		new_lst = []
		for b, r, y in cache.values():
			r, y, b = Image.open(r), Image.open(y), Image.open(b)
			r, y, b = np.asarray(r), np.asarray(y), np.asarray(b)
			stacked = np.dstack((r, y, b)).astype(np.uint8)
			new_lst.append(TF.to_tensor(Image.fromarray(stacked)))

		return new_lst
	return stack

