import glob
import numpy as np

def evaluate_fn(experiment_folder):
	"""
	A function that calculates fitness values for the individuals that were just infereced
	by the pretrained model. Note: hardcoded for now in the server code.
	:param experiment_folder: The folder where the inference data is saved in.
	"""

	#get inference data from the experiment folder
	#lawson_results = glob.glob(experiment_folder + '/yearly_lawson.npy')
	#total_area = glob.glob(experiment_folder + '/area_0.npy')
	lawson_results = np.load(experiment_folder + '/yearly_lawson.npy')
	total_area = np.load(experiment_folder + '/area_0.npy')

	unique, counts = np.unique(lawson_results, return_counts=True)

	try:
		sitting = np.sum(counts[np.where(unique<=2)[0]])
	except:
		sitting = 0

	try:
		dangerous = np.sum(counts[np.where(unique>=4)[0]])
	except:
		dangerous = 0

	sitting_percentage = (sitting / total_area.item()) * 100
	dangerous_percentage = (dangerous / total_area.item()) * 100

	#lawson_sitting.append(sitting_percentage)
	#lawson_dangerous.append(dangerous_percentage)

	return sitting_percentage, dangerous_percentage, sitting/4.2, dangerous/4.2
	#return lawson_sitting, lawson_dangerous, sitting, dangerous
