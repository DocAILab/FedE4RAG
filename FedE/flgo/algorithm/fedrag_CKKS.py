# If you wish to use the CKKS encrypted version of fedrag, please replace the original fedrag.py with the content of this file.

from datetime import datetime
import os
import torch

from .fedbase import BasicServer
from .fedbase import BasicClient
import copy
import collections
import torch
import tenseal as ts  # 需要安装tenseal库: pip install tenseal
import numpy as np

class Server(BasicServer):
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        
        # 生成CKKS上下文和密钥
        self.poly_modulus_degree = 8192  # 存储多项式模次数
        self.context = self.create_ckks_context()
        self.secret_key = self.context.secret_key()
        self.context.make_context_public()  # 移除私钥信息，只保留公钥
        
        # 存储客户端上下文
        self.client_contexts = {}

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

                # # TODO 保存模型
                if self.current_round >= 0:
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"x-model_{current_time}_round{self.current_round}.bin"
                    save_path = os.path.join("path/to/output_model", filename)
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

        avg_dict = {key: value for key, value in avg_dict.items()}

        return avg_dict

    def aggregate(self, model_old, models: list, *args, **kwargs):
        all_params = [model.state_dict() for model in models]
        print("server average tensor")
        ans_params = self.average_tensors(all_params)
        model_old.load_state_dict(ans_params)
        return model_old

    def create_ckks_context(self):
        """创建CKKS加密上下文"""
        # 这些是CKKS的参数，您可能需要根据您的需求调整
        poly_modulus_degree = self.poly_modulus_degree
        coeff_mod_bit_sizes = [60, 40, 40, 60]
        scale = 2**40
        
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        context.global_scale = scale
        return context
    
    def pack(self, client_id, mtype=0, *args, **kwargs):
        """
        包装服务器发送给客户端的信息，添加加密功能
        """
        # 调用父类的pack方法
        original_package = super().pack(client_id, mtype, *args, **kwargs)
        
        # 添加服务器上下文到包中
        original_package["server_context"] = self.context.serialize(save_secret_key=False)
        
        # 如果有客户端的上下文，加密敏感数据
        if client_id in self.client_contexts:
            client_context = self.client_contexts[client_id]
            
            # 获取模型参数
            model_params = []
            if "model" in original_package:
                # 提取模型参数
                model = original_package["model"]
                for param in model.parameters():
                    # 将参数转换为numpy数组并展平
                    param_data = param.data.cpu().numpy().flatten()
                    model_params.append(param_data)
            
            # 加密模型参数
            encrypted_params = []
            for param_data in model_params:
                # 将参数分割成适合CKKS加密的大小
                # CKKS的槽数是多项式模次数的一半
                chunk_size = self.poly_modulus_degree // 2
                for i in range(0, len(param_data), chunk_size):
                    chunk = param_data[i:i+chunk_size]
                    encrypted_chunk = ts.ckks_vector(client_context, chunk)
                    encrypted_params.append(encrypted_chunk.serialize())
            
            original_package["encrypted_params"] = encrypted_params
            # 移除明文的模型数据
            if "model" in original_package:
                del original_package["model"]
        
        return original_package

    def unpack(self, packages_received_from_clients):
        """
        解包从客户端接收的信息，添加解密功能
        """
        if len(packages_received_from_clients) == 0: 
            return collections.defaultdict(list)
        
        # 存储客户端上下文并解密数据
        decrypted_models = []
        other_data = {}
        
        for cpkg in packages_received_from_clients:
            # 保存客户端上下文
            if "client_context" in cpkg:
                client_id = cpkg.get("client_id", len(self.client_contexts))
                self.client_contexts[client_id] = ts.context_from(cpkg["client_context"])
            
            # 解密加密的参数
            if "encrypted_params" in cpkg:
                decrypted_params = []
                for encrypted_param in cpkg["encrypted_params"]:
                    encrypted_vector = ts.ckks_vector_from(self.context, encrypted_param)
                    decrypted_chunk = encrypted_vector.decrypt(self.secret_key)
                    decrypted_params.extend(decrypted_chunk)
                
                # 将解密后的参数重新构造成模型
                model = self.reconstruct_model_from_params(decrypted_params)
                decrypted_models.append(model)
            
            # 收集其他非加密数据
            for key, value in cpkg.items():
                if key not in ["encrypted_params", "client_context"]:
                    if key not in other_data:
                        other_data[key] = []
                    other_data[key].append(value)
        
        # 将解密后的模型添加到结果中
        result = other_data
        if decrypted_models:
            result["model"] = decrypted_models
        
        return result
    
    def reconstruct_model_from_params(self, params):
        """
        根据参数重建模型
        这是一个示例实现，您需要根据您的模型结构进行调整
        """
        # 创建一个新的模型实例
        model = copy.deepcopy(self.model)
        
        # 获取当前模型的参数形状
        param_shapes = []
        for param in model.parameters():
            param_shapes.append(param.data.shape)
        
        # 将扁平化的参数重新构造成原始形状
        param_idx = 0
        for i, param in enumerate(model.parameters()):
            shape = param_shapes[i]
            num_elements = torch.prod(torch.tensor(shape)).item()
            
            # 提取对应形状的参数
            param_data = params[param_idx:param_idx + num_elements]
            param_idx += num_elements
            
            # 重新形状并转换为张量
            param_tensor = torch.tensor(param_data).reshape(shape)
            param.data = param_tensor.to(param.device)
        
        return model
    
class Client(BasicClient):
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        self.server_context = None
        self.poly_modulus_degree = 8192  # 与服务器相同的多项式模次数
        self.client_context = self.create_ckks_context()
        self.client_id = kwargs.get("client_id", 0)

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
    
    def create_ckks_context(self):
        """创建CKKS加密上下文"""
        # 这些是CKKS的参数，您可能需要根据您的需求调整
        poly_modulus_degree = self.poly_modulus_degree
        coeff_mod_bit_sizes = [60, 40, 40, 60]
        scale = 2**40
        
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        context.global_scale = scale
        return context
    
    def unpack(self, received_pkg):
        """
        解包从服务器接收的信息，处理加密数据
        """
        # 保存服务器上下文
        if "server_context" in received_pkg:
            self.server_context = ts.context_from(received_pkg["server_context"])
        
        # 如果有加密的参数，解密它
        if "encrypted_params" in received_pkg and self.client_context.secret_key() is not None:
            decrypted_params = []
            for encrypted_param in received_pkg["encrypted_params"]:
                encrypted_vector = ts.ckks_vector_from(self.client_context, encrypted_param)
                decrypted_chunk = encrypted_vector.decrypt()
                decrypted_params.extend(decrypted_chunk)
            
            # 将解密后的参数重新构造成模型
            model = self.reconstruct_model_from_params(decrypted_params)
            received_pkg["model"] = model
            del received_pkg["encrypted_params"]
        
        # 移除服务器上下文，避免干扰父类的处理
        if "server_context" in received_pkg:
            del received_pkg["server_context"]
        
        return super().unpack(received_pkg)
    
    def pack(self, model, *args, **kwargs):
        """
        包装客户端发送给服务器的信息，添加加密功能
        """
        # 提取模型参数并扁平化
        model_params = []
        for param in model.parameters():
            param_data = param.data.cpu().numpy().flatten()
            model_params.append(param_data)
        
        # 如果有服务器上下文，则加密参数
        if self.server_context:
            # 加密模型参数
            encrypted_params = []
            for param_data in model_params:
                # 将参数分割成适合CKKS加密的大小
                # CKKS的槽数是多项式模次数的一半
                chunk_size = self.poly_modulus_degree // 2
                for i in range(0, len(param_data), chunk_size):
                    chunk = param_data[i:i+chunk_size]
                    encrypted_chunk = ts.ckks_vector(self.server_context, chunk)
                    encrypted_params.append(encrypted_chunk.serialize())
            
            # 创建包含加密参数和客户端上下文的包
            package = {
                "encrypted_params": encrypted_params,
                "client_context": self.client_context.serialize(save_secret_key=False),
                "client_id": self.client_id
            }
            
            return package
        else:
            # 如果没有服务器上下文，使用原始client的pack方法
            original_package = super().pack(model, *args, **kwargs)
            # 添加上下文
            original_package["client_context"] = self.client_context.serialize(save_secret_key=False)
            original_package["client_id"] = self.client_id
            return original_package
    
    def reconstruct_model_from_params(self, params):
        """
        根据参数重建模型
        这是一个示例实现，您需要根据您的模型结构进行调整
        """
        # 创建一个新的模型实例
        model = copy.deepcopy(self.model)
        
        # 获取当前模型的参数形状
        param_shapes = []
        for param in model.parameters():
            param_shapes.append(param.data.shape)
        
        # 将扁平化的参数重新构造成原始形状
        param_idx = 0
        for i, param in enumerate(model.parameters()):
            shape = param_shapes[i]
            num_elements = torch.prod(torch.tensor(shape)).item()
            
            # 提取对应形状的参数
            param_data = params[param_idx:param_idx + num_elements]
            param_idx += num_elements
            
            # 重新形状并转换为张量
            param_tensor = torch.tensor(param_data).reshape(shape)
            param.data = param_tensor.to(param.device)
        
        return model
