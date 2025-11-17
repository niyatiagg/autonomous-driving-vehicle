import optuna.visualization as vis
import pickle
from plotly.io import show

studyname = "intersection_grayscale_continuous"

with open(f"{studyname}.pkl", "rb") as file:
    study = pickle.load(file)

# fig = vis.plot_param_importances(study)
fig = vis.plot_rank(study)
# fig = vis.plot_slice(study)
# fig = vis.plot_intermediate_values(study)
# fig = vis.plot_timeline(study)
show(fig)