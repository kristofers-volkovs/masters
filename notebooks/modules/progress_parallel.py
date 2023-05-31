from joblib import Parallel
from tqdm.auto import tqdm 

class ProgressParallel(Parallel):
    def __init__(self, total=None, desc=None, *args, **kwargs):
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)
        
    def __call__(self, *args, **kwargs):
        with tqdm(total=self._total, desc=self._desc) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
