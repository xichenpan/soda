from random import Random
from typing import Dict

from infinibatch.iterators import CheckpointableIterator, FixedBatchIterator, SelectManyIterator
from torch.utils.data import IterableDataset
from typing import Any, Callable, Optional, List

import multiprocessing


class MapIterator(CheckpointableIterator):
    """
    Applies given tranform to each data item
    """

    def __init__(self, source_iterator: CheckpointableIterator, transform: Callable[[str], Any]):
        """
        Args:
            source_iterator: checkpointable iterator
            transform: function to be applied to each data item
        """
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        self._source_iterator = source_iterator
        self._transform = transform
        self.dummy = None

    def getstate(self) -> Dict:
        return self._source_iterator.getstate()

    def setstate(self, checkpoint: Optional[Dict]):
        self._source_iterator.setstate(checkpoint)

    def __next__(self):
        item = None
        max_try = 10
        while item is None and max_try > 0:
            try:
                item = self._transform(next(self._source_iterator))
                if self.dummy is None:
                    self.dummy = item
                return item
            except:
                max_try -= 1
        return self.dummy

    def close(self):
        self._source_iterator.close()


def ParallelMapIterator(source_iterator: CheckpointableIterator, transform: Callable[[str], Any], num_processes: int,
                        num_items_per_process: int) -> CheckpointableIterator:
    """
    Applies given transform to each data item

    Behaves the same as MapIterator, but applies transform in parallel using multiple processes in a parallel map operation.

    Warning:
    The transform function has to be pickleable because it is sent across process boundaries.
    To achieve this, transform should be a top-level function.

    Args:
        source_iterator: checkpointable iterator
        transform: function to be applied to each data item, has to be pickleable, see above
        num_processes: number of processes to use for parallel map
        num_items_per_process: number of data items each process operates on
    """
    # divide stream of data items into batches
    batched_samples = FixedBatchIterator(source_iterator, num_processes * num_items_per_process)
    # create process pool and capture it in closure that performs parallel map
    p = multiprocessing.Pool(num_processes)

    def parallel_map_transform(buffer):
        return p.map(transform, buffer)

    # apply transform in parallel to data items in a batch
    batched_transformed_samples = MapIterator(batched_samples, parallel_map_transform)
    # unpack batches to go back to stream of (now transformed) data items
    transformed_samples = SelectManyIterator(batched_transformed_samples)
    return transformed_samples


class BaseBatchGen(CheckpointableIterator):
    """
    This is a base class for batch generators that use infinibatch
    """

    def __init__(self, _iter=None):
        self._iter = iter(_iter)
        self.state = None
        self.epoch = 1
        self.next_epoch_idx = 1
        self.sharded_checkpoint = False
        self.should_close_after_finished = True

    @property
    def iterator(self):
        return self._iter

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def __len__(self) -> int:
        return 819200000

    def getstate(self) -> Dict:
        return self.state

    def setstate(self, state: Dict):
        self.state = state

    def close(self):
        pass


class MixIterator(CheckpointableIterator):
    """
    Concat items from all given iterators.
    """

    def __init__(self, source_iterators, weights):
        """
        Args:
                source_iterators: list of iterators to zip, item by item
        """
        for source_iterator in source_iterators:
            if not isinstance(source_iterator, CheckpointableIterator):
                raise ValueError('all iterators in source_iterators have to be CheckpointableIterator')
        self._source_iterators = source_iterators  # type: List[CheckpointableIterator]
        assert len(weights) == len(source_iterators)
        self.weights = weights
        self.population = list(range(len(source_iterators)))

    def getstate(self):
        return {'input_states': tuple(iterator.getstate() for iterator in self._source_iterators)}

    def setstate(self, checkpoint):
        if checkpoint is None:
            for iterator in self._source_iterators:
                iterator.setstate(None)
        else:
            for iterator, state in zip(self._source_iterators, checkpoint['input_states']):
                iterator.setstate(state)

    def __next__(self):
        _random = Random()
        res = {}
        idx = _random.choices(self.population, self.weights)[0]
        res.update(next(self._source_iterators[idx]))
        return res

    def close(self):
        for it in self._source_iterators:
            it.close()


class BaseDatasetWrapper(IterableDataset):
    def __init__(self, _iter=None):
        super(BaseDatasetWrapper).__init__()
        self._iter = _iter

    def __iter__(self):
        return self._iter
