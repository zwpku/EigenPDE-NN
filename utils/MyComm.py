
class Comm():
  def __init__(self, distributed_training):
    if distributed_training : 
        """
          Currently we use mpi as backend.
          It requires to install pytorch with mpi (from source code).
          The backend can be easily changed to others.
        """
        try : 
            import torch.distributed as dist
            from torch.multiprocessing import Process
            dist.init_process_group('mpi')
            self.dist = dist 
            self.rank = self.dist.get_rank()
            self.size = self.dist.get_world_size()
        except ImportError: 
            print ("Cann't import parallel libraries! Re-run with distributed_training=Flase") 
            sys.exit(1)
    else : # run sequentially 
        self.dist = None
        self.rank = 0
        self.size = 1

