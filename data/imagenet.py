from datasets import load_dataset


def get_dataset(data_dir, seed):
    dataset = load_dataset(
        "ILSVRC/imagenet-1k", trust_remote_code=True, cache_dir=data_dir,
        split="train", keep_in_memory=True
    )
    dataset = dataset.to_iterable_dataset(num_shards=1024)
    dataset = dataset.remove_columns("label")
    dataset = dataset.shuffle(seed=seed)
    return dataset
