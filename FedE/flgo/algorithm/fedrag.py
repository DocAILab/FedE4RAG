from datetime import datetime
import os
import torch

from .fedbase import BasicServer
from .fedbase import BasicClient


class Server(BasicServer):
    def run(self):
        """
        Running the FL symtem where the global model is trained and evaluated iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')

        print(type(self.model.model))

        if not self._load_checkpoint() and self.eval_interval > 0:
            # evaluating initial model performance
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            # self.gv.logger.log_once()
            self.gv.logger.time_end('Eval Time Cost')
        # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # filename = f"x-model_{current_time}_1.bin"
        # torch.save(self.model, filename)
        # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # filename = f"x-model_{current_time}_2.bin"
        # torch.save(self.model.state_dict(), filename)
        while True:
            if self._if_exit(): break
            self.gv.clock.step()
            # iterate
            updated = self.iterate()
            # using logger to evaluate the model if the model is updated
            if updated is True or updated is None:
                self.gv.logger.info("--------------Round {}--------------".format(self.current_round))
                # check log interval
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    # self.gv.logger.log_once()
                    self.gv.logger.time_end('Eval Time Cost')
                    self._save_checkpoint()
                # check if early stopping
                if self.gv.logger.early_stop(): break

                # if self.current_round % 5 == 0:
                # # TODO 保存模型
                if self.current_round >= 0:
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"x-model_{current_time}_round{self.current_round}.bin"
                    save_path = os.path.join("/root/autodl-tmp/0401_1000", filename)
                    torch.save(self.model.model.state_dict(), save_path)
                self.current_round += 1
                # decay learning rate
                self.global_lr_scheduler(self.current_round)
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        # TODO 保存模型
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"x-model_{current_time}.bin"
        torch.save(self.model.model.state_dict(), filename)
        # self.model.save_pretrained('./logs')
        return

    def average_tensors(self, list_of_dicts):
        if not list_of_dicts:
            return None
        avg_dict = {}
        for d in list_of_dicts:
            for key, value in d.items():
                if key not in avg_dict:
                    avg_dict[key] = value.clone()
                else:
                    avg_dict[key] += value
        num_dicts = len(list_of_dicts)
        for key in avg_dict:
            avg_dict[key] /= num_dicts

        avg_dict = {'model.' + key: value for key, value in avg_dict.items()}

        return avg_dict

    def aggregate(self, model_old, models: list, *args, **kwargs):
        all_params = [model.state_dict() for model in models]
        print("server average tensor")
        ans_params = self.average_tensors(all_params)
        model_old.load_state_dict(ans_params)
        return model_old


class Client(BasicClient):
    def train(self, model, local_model):
        local_model.train()
        optimizer = self.calculator.get_optimizer(local_model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)

        model.to(self.device)
        local_model.to(self.device)

        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            local_model.zero_grad()
            # Todo baseline 上增加 server_loss
            server_loss = self.calculator.compute_server_loss(model, batch_data)
            client_loss, client_only, server_only = self.calculator.compute_client_loss(server_loss, local_model,
                                                                                        batch_data)
            # client_loss, client_only, server_only = self.calculator.compute_client_loss(local_model, batch_data)

            # print(batch_data[1])
            print(
                f"client running:{iter}/{self.num_steps}, client loss: {client_loss}, loss 1: {client_only}, loss 2: {server_only}")

            client_loss.backward()
            optimizer.step()

            if iter == self.num_steps - 1:
                print(f"server loss: {server_loss}")
        return

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        self.train(model, self.model)
        cpkg = self.pack(self.model)
        return cpkg
