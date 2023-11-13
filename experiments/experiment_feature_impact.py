from base.dataset_loader import OxariDataManager
from base.run_utils import get_small_datamanager_configuration

import pathlib
import pickle
from postprocessors.core import DecisionExplainer, JumpRateExplainer, ResidualExplainer, ShapExplainer

if __name__ == "__main__":

    cwd = pathlib.Path(__file__).parent
    model = pickle.load((cwd.parent / 'model-data/output/T20231113_p_model_experiment_feature_impact.pkl').open('rb'))

    dataset = get_small_datamanager_configuration().run()
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    explainer0 = ShapExplainer(model.get_pipeline(1), sample_size=100).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    fig, ax = explainer0.visualize()
    fig.savefig(f'local/eval_results/importance_explainer{0}.png')
    # explainer1 = ResidualExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # explainer2 = JumpRateExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # explainer3 = DecisionExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)

    # for intervall_group, expl in enumerate([explainer1, explainer2, explainer3]):
    #     fig, ax = expl.plot_tree()
    #     fig.savefig(f'local/eval_results/tree_explainer{intervall_group+1}.png', dpi=600)
    #     fig, ax = expl.plot_importances()
    #     fig.savefig(f'local/eval_results/importance_explainer{intervall_group+1}.png')