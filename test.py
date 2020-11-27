# %% codecell
import numpy as np
from tqdm import tqdm
import map_elites as ae
from plot_utils import plot_heatmap
config_path = 'config.ini'

map = ae.MapElites
qd_urban = map.from_config(config_path=config_path, overwrite=True)
qd_urban.run()

# Ideas to try:
# encoder model to learn the embedding of the behavioral space
# encoder model to learn the embedding of the elite hypervolume
# use ES apprach: alternate between selecting the most novel individual and selecting the most fit individual
# use directional mutation to evolve a population
# multi-emitter MAP-Elites
# adaptive sampling to remove noise from fitness

# %%

# %%
