from flgo.benchmark.fedrag_classification.model import fedllm
import flgo.benchmark.toolkits.partition

# TODO 修改数据 客户端数量
default_partitioner = flgo.benchmark.toolkits.partition.IDPartitioner
default_partition_para = {'num_clients': 5}
default_model = fedllm
