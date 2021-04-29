
class Comm():
  def __init__(self, distributed_training):
    if distributed_training : 
        """
          Currently we use mpi as backend.
          It requires to install pytorch with mpi (from source code).
          The backend can be easily changed to others.
        """
        import torch.distributed as dist
        from torch.multiprocessing import Process

        dist.init_process_group('mpi')
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
    else : # run sequentially 
        self.rank = 0
        self.size = 1

