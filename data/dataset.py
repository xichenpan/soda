import importlib

from data.config import dataconfig
from data.utils import MapIterator, BaseBatchGen, MixIterator, BaseDatasetWrapper


def get_dataset(process_fn, data_dir, seed):
    datasets = []
    probabilities = []
    for dataset, info in dataconfig.items():
        try:
            get_dataset = importlib.import_module(f'data.{dataset}').get_dataset
        except ImportError:
            print(f"Dataset class {dataset} is not found.")
            continue
        proportion = float(info.get('proportion', 0))
        if proportion > 0:
            datasets.append(get_dataset(data_dir, seed))
            probabilities.append(proportion)

    if len(datasets) == 0:
        raise ValueError('No dataset is enabled')
    # get float probability for each datasetcd
    probabilities = [p / sum(probabilities) for p in probabilities]
    # CheckpointableIterator wrapper for the dataset
    dataloader = MixIterator([BaseBatchGen(_iter=dataset) for dataset in datasets], probabilities)

    # similar to __getitem__ in PyTorch Dataset
    dataloader = MapIterator(
        source_iterator=dataloader,
        transform=process_fn,
    )

    return BaseDatasetWrapper(_iter=dataloader)
