# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import Model
from HHtobbyy.event_discrimination.models import map_model_to_Model
from HHtobbyy.workspace_utils.retrieval_utils import get_input_filepaths
from HHtobbyy.Categorization import Categorization


def main(dfdataset: DFDataset, model: Model, train_filepaths: list, test_filepaths: list):
    # Building train DFDataset
    # dfdataset.make_all_train(train_filepaths, parallel=True)

    # Training the model
    # model.train_all_folds(parallel=True)

    # Building test DFDataset
    # dfdataset.make_all_test(test_filepaths)

    # Evaluating the model
    model.predict_all_folds(regex='allData')

    # Categorizing the model
    # cat = Categorization(dfdataset, {"discriminator": "3D"})
    # cat.run()


dfdataset_config = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/DFDatasets/vManos/24_Manos_2026-04-20_12-49-04/dataset_config.json"
model, model_config = "MLP", {"output_dirpath": "/uscms/home/tsievert/nobackup/XHYbbgg/Model_Outputs/ManosMLP/2026-04-20_13-59-41", "activation_func": "ELU"}
test_eras = "/uscms/home/tsievert/nobackup/XHYbbgg/HHtobbyy/configs/DFDataset_v7_eras_1618.txt"

dfdataset = DFDataset(dfdataset_config)
model = map_model_to_Model(model)(dfdataset, model_config)

test_filepaths = get_input_filepaths(test_eras, dfdataset.class_sample_map, regex=f"*{dfdataset.filepostfix}")

main(dfdataset, model, None, test_filepaths)