# %% codecell
import numpy as np
import map_elites as ae
from plot_utils import plot_heatmap
config_path = 'config.ini'

map = ae.MapElites
qd_urban = map.from_config(config_path=config_path, overwrite=True)
qd_urban.run()

qd_urban.plot_map_of_elites(iteration=1)
# Ideas to try:
# encoder model to learn the embedding of the behavioral space
# encoder model to learn the embedding of the elite hypervolume
# use ES apprach: alternate between selecting the most novel individual and selecting the most fit individual
# use directional mutation to evolve a population
# multi-emitter MAP-Elites
# adaptive sampling to remove noise from fitness
perf = np.load("F:\PhD_Research\CaseStudies\MAP-Elites\pv_urban\my_library\logs\log_20201011010555\performances.npy")
plot_heatmap(perf*4.2)
perf.max()*4.2
