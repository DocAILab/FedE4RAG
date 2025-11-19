import flgo
import flgo.algorithm.fedrag as fedrag
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

task = './num5_alpha05'
# TODO 修改数据
config = {'benchmark': {'name': 'flgo.benchmark.fedrag_classification'},
          'partitioner': {'name': 'IDPartitioner', 'para': {'num_clients': 5}}}
if not os.path.exists(task): flgo.gen_task(config, task_path=task)

fedavg_runner = flgo.init(task=task, algorithm=fedrag,
                          option={'num_rounds': 25, 'num_epochs': 1, 'gpu': 0, 'batch_size': 8,
                                  "learning_rate": 0.00001})
fedavg_runner.run()
