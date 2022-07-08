import numpy
import os
import sys
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision
import torchvision.transforms as transforms
from glob import glob
from tqdm import tqdm

# For recording progress and showing outputs:
from torch.utils.tensorboard import SummaryWriter

from dataset import TextDetectionDataset, TextRecognitionDataset
from model import UNet, Reshape

#wandb.init(project="drawing_to_art", entity="josephc")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ocr_text_detection"
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
EPOCHS = 100
BATCH_SIZE = 16
CHANGENOTES = "Working on the detection script again with a UNet model."


def record_run_config(filename, model, output_dir) -> int:
	"""Return the number of previous runs."""
	previous_runs = len(glob(f"{output_dir}/{filename[:-3]}*"))
	run_number = previous_runs+1  # One indexed.  Whatever.
	with open(os.path.join(output_dir, f"{filename}_{run_number}"), 'wt') as fout:
		fout.write(f"MODEL NAME: {MODEL_NAME}\n")
		fout.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
		fout.write(f"EPOCHS: {EPOCHS}\n")
		fout.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
		fout.write(f"CHANGENOTES: {CHANGENOTES}\n")
		fout.write(f"ARCH: {model}")
	return run_number


def export_model(model, input_channels, input_height, input_width, filename):
	model.eval()
	x = torch.randn(1, input_channels, input_height, input_width, requires_grad=True).to(DEVICE)
	_output = model(x)
	torch.onnx.export(
		model,
		x,
		filename,
		export_params=True,
		opset_version=10,
		do_constant_folding=True,
		input_names=['input'],
		output_names=['output'],
		dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
	)
	model.train()


def dice_loss(pred, target):
	# Compute softmax over the classes axis, normally.
	# Our output class is 1/0, though.  Mask/no mask.
	#input_soft = nn.functional.softmax(input, dim=1)

	# compute the actual dice score
	dims = (1, 2, 3)
	intersection = torch.sum(pred * target, dims)
	cardinality = torch.sum(pred + target, dims)

	dice_score = 2. * intersection / (cardinality + 1e-6)
	return torch.mean(1. - dice_score)


def train(dataset, model, optimizer, loss_fn, summary_writer=None, validation_set=None):
	for epoch_idx in range(EPOCHS):
		dataloop = tqdm(dataset)
		for batch_idx, (data, targets) in enumerate(dataloop):
			step = (epoch_idx * len(dataloop)) + batch_idx
			# For Text Detection:
			data = data.permute(0, 3, 1, 2).to(device=DEVICE)
			tgt = targets.float().unsqueeze(1).to(device=DEVICE)  # NOTE: Output is greyscale, so we unsqueeze channels at 1.
			# For Text Recognition:
			#data = data.permute(0, 3, 1, 2).to(device=DEVICE)
			#tgt = targets.float().to(device=DEVICE)
			optimizer.zero_grad()

			# Forward
			preds = model(data)

			# Backward
			loss = loss_fn(preds, tgt)
			loss.backward()
			optimizer.step()

			if summary_writer and batch_idx % 100 == 0:
				# Save sample images.
				input_grid = torchvision.utils.make_grid(data)
				#target_grid = torchvision.utils.make_grid(tgt)
				#output_grid = torchvision.utils.make_grid(preds)
				summary_writer.add_image("Input Grid", input_grid, step)
				#summary_writer.add_image("Target Grid", target_grid, step)
				#summary_writer.add_image("Output Grid", output_grid, step)
				summary_writer.add_text("Target Text", TextRecognitionDataset.sparse_array_to_text(tgt.cpu().detach().numpy()[0]))
				summary_writer.add_text("Output Text", TextRecognitionDataset.sparse_array_to_text(preds.cpu().detach().numpy()[0]))

				validation_loss = 0
				if validation_set:
					model.eval()
					preds = model(validation_set[0])
					val_loss = loss_fn(preds, validation_set[1])
					validation_loss = val_loss.item()
					validation_input_grid = torchvision.utils.make_grid(validation_set[0])
					summary_writer.add_image("Validation Input Grid", validation_input_grid, step)
					validation_target_grid = torchvision.utils.make_grid(validation_set[1])
					summary_writer.add_image("Validation Target Grid", validation_target_grid, step)
					validation_output_grid = torchvision.utils.make_grid(preds)
					summary_writer.add_image("Validation Output Grid", validation_output_grid, step)
					model.train()

				# Write all network params to the log.
				#for name, weight in model.named_parameters():
				#	summary_writer.add_histogram(name, weight, step)
				#	summary_writer.add_histogram(f'{name}.grad', weight.grad, step)

				summary_writer.add_scalars('Losses', {"Last Training Loss": loss.item(), "Validation Loss": validation_loss}, step)
				summary_writer.flush()
		torch.save(model.state_dict(), f"checkpoints/{MODEL_NAME}_{epoch_idx}")
	print("Done!  Breakpoint to save:")
	breakpoint()
	#export_model(model, 1, 128, 128, f"models/{MODEL_NAME}_{epoch_idx}.onnx")


def train_detection_model(model_start_file=None):
	model = UNet(in_channels=3, out_channels=1, feature_counts=[4, 8, 16]).to(device=DEVICE)
	#loss_fn = nn.L1Loss()
	loss_fn = dice_loss
	optimizer = opt.Adam(model.parameters(), lr=LEARNING_RATE)

	if model_start_file:
		print(f"Restarting from checkpoint {model_start_file}")
		model.load_state_dict(torch.load(model_start_file))

	dataset = TextDetectionDataset(target_width=256, target_height=256)
	training_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

	# Set up summary writer and record run stats.
	run_number = record_run_config(MODEL_NAME, model, "runs")
	os.mkdir(os.path.join("runs", str(run_number)))
	summary_writer = SummaryWriter(f"runs/{run_number}")
	print(f"Writing summary log to runs/{run_number}")

	# Log the model architecture:
	summary_writer.add_graph(model, torch.Tensor(numpy.zeros((1,3,dataset.target_height,dataset.target_width))).to(DEVICE))

	# Get a validation set.
	validation_images, validation_masks = dataset.make_validation_set()
	validation_images = validation_images.permute(0, 3, 1, 2).to(device=DEVICE)
	validation_masks = validation_masks.unsqueeze(1).to(device=DEVICE)

	# Train
	train(training_loader, model, optimizer, loss_fn, summary_writer=summary_writer, validation_set=(validation_images, validation_masks))

	# Write to file.
	export_model(model, 3, dataset.target_height, dataset.target_width, "result_model.onnx")


def train_recognition_model():
	dataset = TextRecognitionDataset(target_width=64, target_height=64)
	model = nn.Sequential(
		nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64x64x64
		nn.SiLU(inplace=True),
		nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 64x64x128
		nn.SiLU(inplace=True),
		nn.MaxPool2d(2, 2),  # 64x64x128 -> 32x32x128

		nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 32x32x128
		nn.SiLU(inplace=True),
		nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 32x32x256
		nn.SiLU(inplace=True),
		nn.MaxPool2d(2, 2),  # 32x32x256 -> 16x16x256

		nn.Flatten(),
		nn.Linear(in_features=16*16*256, out_features=1024),  # This should be 16x16x256, but it's 43264.
		nn.SiLU(inplace=True),
		nn.Linear(in_features=1024, out_features=dataset.get_output_dims()[0]*dataset.get_output_dims()[1]),
		Reshape(-1, dataset.get_output_dims()[0], dataset.get_output_dims()[1]),
		#nn.Unflatten(1, dataset.get_output_dims()),
		#nn.Softmax(dim=1),
		nn.SiLU(inplace=True),
	)
	print(f"Model: {model}")
	model.to(DEVICE)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = opt.Adam(model.parameters(), lr=LEARNING_RATE)

	training_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

	# Set up summary writer and record run stats.
	run_number = record_run_config(MODEL_NAME, model, "runs")
	os.mkdir(os.path.join("runs", str(run_number)))
	summary_writer = SummaryWriter(f"runs/{run_number}")
	print(f"Writing summary log to runs/{run_number}")

	# Log the model architecture:
	summary_writer.add_graph(model, torch.Tensor(numpy.zeros((1, 3, dataset.target_height, dataset.target_width))).to(DEVICE))

	# Train
	train(training_loader, model, optimizer, loss_fn, summary_writer=summary_writer)

	# Write to file.
	export_model(model, 3, dataset.target_height, dataset.target_width, "recognition_model.onnx")


def main(model_start_file=None):
	train_detection_model(model_start_file)
	#train_recognition_model()


if __name__=="__main__":
	starting_file = None
	if len(sys.argv) > 1:
		starting_file = sys.argv[1]
	main(starting_file)
