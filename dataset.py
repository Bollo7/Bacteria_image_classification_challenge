import torch
from PIL import Image
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

class Dataset_train(torch.utils.data.Dataset):
	def __init__(self, imgs, targets, transform = None):
		super(Dataset_train, self).__init__()

		self.imgs = imgs
		self.targets = targets
		self.transform = transform


	def __getitem__(self, index):

		image = self.imgs[index]
		target = self.targets[index]

		if self.transform:  # check for minority class
			image = TF.to_pil_image(image)
			image = self.transform(image)
			image = TF.to_tensor(image)

		return (image, target)

	def __len__(self):

		return len(self.imgs)


class Dataset_test(torch.utils.data.Dataset):
	def __init__(self, imgs):
		super(Dataset_test, self).__init__()

		self.imgs = imgs

	def __getitem__(self, index):
		image = self.imgs[index]

		return (image)

	def __len__(self):
		return len(self.imgs)