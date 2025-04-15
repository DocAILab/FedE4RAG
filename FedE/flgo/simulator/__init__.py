r"""
This module is to simulate arbitrary system heterogeneity that may occur in practice.
We conclude four types of system heterogeneity from existing works.
System Heterogeneity Description:
    1. **Availability**: the devices will be either available or unavailable at each moment, where only the
                    available devices can be selected to participate in training.

    2. **Responsiveness**: the responsiveness describes the length of the period from the server broadcasting the
                    gloabl model to the server receiving the locally trained model from a particular client.

    3. **Completeness**: since the server cannot fully control the behavior of devices,it's possible for devices to
                    upload imcomplete model updates (i.e. only training for a few steps).

    4. **Connectivity**: the clients who promise to complete training may suffer accidients so that the server may lose
                    connections with these client who will never return the currently trained local_movielens_recommendation model.

We build up a client state machine to simulate the four types of system heterogeneity, and provide high-level
APIs to allow customized system heterogeneity simulation.

**Example**: How to customize the system heterogeneity:
```python
>>> class MySimulator(flgo.simulator.base.BasicSimulator):
...     def update_client_availability(self):
...         # update the variable 'prob_available' and 'prob_unavailable' for all the clients
...         self.set_variable(self.all_clients, 'prob_available', [0.9 for _ in self.all_clients])
...         self.set_variable(self.all_clients, 'prob_unavailable', [0.1 for _ in self.all_clients])
...
...     def update_client_connectivity(self, client_ids):
...         # update the variable 'prob_drop' for clients in client_ids
...         self.set_variable(client_ids, 'prob_drop', [0.1 for _ in client_ids])
...
...     def update_client_responsiveness(self, client_ids, *args, **kwargs):
...         # update the variable 'latency' for clients in client_ids
...         self.set_variable(client_ids, 'latency', [np.random.randint(5,100) for _ in client_ids])
...
...     def update_client_completeness(self, client_ids, *args, **kwargs):
...         # update the variable 'working_amount' for clients in client_ids
...         self.set_variable(client_ids, 'working_amount',  [max(int(self.clients[cid].num_steps*np.random.rand()), 1) for cid in client_ids])
>>> r = flgo.init(task, algorithm=fedavg, Simulator=MySimulator)
>>> # The runner r will be runned under the customized system heterogeneity, where the clients' states will be flushed by
>>> # MySimulator.update_client_xxx at each moment of the virtual clock or particular events happen (i.e. a client was selected)
```

We also provide some preset Simulator like flgo.simulator.DefaultSimulator and flgo.simulator.
"""
from flgo.simulator.default_simulator import Simulator as DefaultSimulator
from flgo.simulator.phone_simulator import Simulator as PhoneSimulator
from flgo.simulator.base import BasicSimulator
import numpy as np
import random

class ResponsivenessExampleSimulator(BasicSimulator):
    def initialize(self):
        self.client_time_response = {cid: np.random.randint(5, 1000) for cid in self.clients}
        self.set_variable(list(self.clients.keys()), 'latency', list(self.client_time_response.values()))

    def update_client_responsiveness(self, client_ids):
        latency = [self.client_time_response[cid] for cid in client_ids]
        self.set_variable(client_ids, 'latency', latency)

class CompletenessExampleSimulator(BasicSimulator):
    def update_client_completeness(self, client_ids):
        if not hasattr(self, '_my_working_amount'):
            rs = self.random_module.normal(1.0, 1.0, len(self.clients))
            rs = rs.clip(0.01, 2)
            self._my_working_amount = {cid:max(int(r*self.clients[cid].num_steps),1) for  cid,r in zip(self.clients, rs)}
        working_amount = [self._my_working_amount[cid] for cid in client_ids]
        self.set_variable(client_ids, 'working_amount', working_amount)

class AvailabilityExampleSimulator(BasicSimulator):
    def update_client_availability(self):
        if self.gv.clock.current_time==0:
            self.set_variable(self.all_clients, 'prob_available', [1.0 for _ in self.clients])
            self.set_variable(self.all_clients, 'prob_unavailable', [0.0 for _ in self.clients])
            return
        pa = [0.1 for _ in self.clients]
        pua = [0.1 for _ in self.clients]
        self.set_variable(self.all_clients, 'prob_available', pa)
        self.set_variable(self.all_clients, 'prob_unavailable', pua)

class ConnectivityExampleSimulator(BasicSimulator):
    def initialize(self):
        drop_probs = self.random_module.uniform(0.,0.05, len(self.clients)).tolist()
        self.client_drop_prob = {cid: dp for cid,dp in zip(self.clients, drop_probs)}

    def update_client_connectivity(self, client_ids):
        self.set_variable(client_ids, 'prob_drop', [self.client_drop_prob[cid] for cid in client_ids])