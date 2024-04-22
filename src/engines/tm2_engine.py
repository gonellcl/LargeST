import os
import time
import torch
import numpy as np

from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics
from src.base.engine import BaseEngine


class TestEngine(BaseEngine):
    def __init__(self, device, model, dataloader, scaler, sampler, loss_fn, lrate, optimizer, \
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed):
        super(TestEngine).__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed

        self._logger.info('The number of parameters: {}'.format(self.model.param_num()))

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)

    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def load_model(self, save_path):
        filename = 'final_model_s{}.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        mc_losses = []  # To store mincut losses
        o_losses = []  # To store orthogonality losses

        for X, label in self._dataloader['train_loader'].get_iterator():
            X, label = self._to_device(self._to_tensor([X, label]))
            self._optimizer.zero_grad()

            pred, mc_loss, o_loss = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            main_loss = self._loss_fn(pred, label, mask_value)
            total_loss = main_loss + mc_loss + o_loss  # Combine losses
            print(F" Iter_count: {self._iter_cnt} main loss: {main_loss}, total_loss: {total_loss}")

            total_loss.backward()
            if self._clip_grad_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(main_loss.item())
            mc_losses.append(mc_loss.item())
            o_losses.append(o_loss.item())
            train_mape.append(masked_mape(pred, label, mask_value).item())
            train_rmse.append(masked_rmse(pred, label, mask_value).item())

            self._iter_cnt += 1
        avg_main_loss = np.mean(train_loss)
        avg_mc_loss = np.mean(mc_losses)
        avg_o_loss = np.mean(o_losses)
        avg_mape = np.mean(train_mape)
        avg_rmse = np.mean(train_rmse)

        return avg_main_loss, avg_mape, avg_rmse, avg_mc_loss, avg_o_loss

    def train(self):
        self._logger.info("Start training!")
        min_loss = float('inf')
        wait = 0

        for epoch in range(self._max_epochs):
            epoch_start = time.time()
            train_loss, train_mape, train_rmse, mc_loss, o_loss = self.train_batch()
            print(f" train loss: {train_loss}, train_mape: {train_mape}, train_rmse: {train_rmse}, mc_loss: {mc_loss}, o_loss:{o_loss}")

            valid_loss, valid_mape, valid_rmse = self.evaluate('val')
            print(F"valid loss: {valid_loss}, valid_mape: {valid_mape}, valid_rmse:{valid_rmse} ")

            epoch_time = time.time() - epoch_start
            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            self._logger.info(
                f"Epoch {epoch + 1}/{self._max_epochs}: Loss: {train_loss:.4f}, MAPE: {train_mape:.4f}, "
                f"RMSE: {train_rmse:.4f}, Mincut Loss: {mc_loss:.4f}, Ortho Loss: {o_loss:.4f}, "
                f"Val Loss: {valid_loss:.4f}, Val MAPE: {valid_mape:.4f}, Val RMSE: {valid_rmse:.4f}, "
                f"Epoch Time: {epoch_time:.2f}s", cur_lr)

            if valid_loss < min_loss:
                min_loss = valid_loss
                self.save_model(self._save_path)
                self._logger.info(f"New best model saved at {min_loss:.4f}")
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break
        self.evaluate('test')

    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred, mc_loss, o_loss = self.model(X, label)

                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
