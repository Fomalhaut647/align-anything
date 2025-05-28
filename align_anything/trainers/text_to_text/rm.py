# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trainer for reward model training."""


import argparse
import os
import sys
from typing import Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text.preference import PreferenceBatch, PreferenceDataset
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.device_utils import get_current_device, torch_gc, torch_set_device
from align_anything.utils.multi_process import (
    get_all_reduce_mean,
    get_current_device,
    is_main_process,
)
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    update_dict,
)


class RMTrainer(SupervisedTrainerBase):

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize the reward model trainer."""
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0
        self.infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}

        self.init_check()
        dist.barrier()
        self.init_models()
        if hasattr(self.model, 'infer_batch'):
            self.infer_batch = self.model.infer_batch
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

    def init_check(self) -> None:
        """Initial configuration checking."""
        super().init_check()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs is not None and self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_train_cfgs)
        self.model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            PreferenceDataset, PreferenceDataset
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines()

    def loss(
        self,
        batch: PreferenceBatch,
    ) -> dict[str, torch.Tensor]:
        """Loss function for the reward model."""
        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = batch[
            'input_ids'
        ].chunk(chunks=2, dim=0)
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        output = self.model(**self.infer_batch(batch))
        scores = output.scores
        end_scores = output.end_scores
        higher_rewards, lower_rewards = scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
        higher_end_reward, lower_end_reward = end_scores.squeeze(dim=-1).chunk(chunks=2, dim=0)

        loss = -F.logsigmoid(higher_end_reward - lower_end_reward).mean()

        if self.cfgs.train_cfgs.regularization > 0.0:
            loss = (
                loss
                + self.cfgs.train_cfgs.regularization
                * torch.stack([lower_end_reward, higher_end_reward]).square().mean()
            )

        accuracy = (higher_end_reward > lower_end_reward).float().mean()  # size = ()
        return {
            'loss': loss,  # size = ()
            'higher_end_reward': higher_end_reward,  # size = (B,)
            'lower_end_reward': lower_end_reward,  # size = (B,)
            'higher_rewards': higher_rewards,  # size = (B, L)
            'lower_rewards': lower_rewards,  # size = (B, L)
            'accuracy': accuracy,  # size = ()
        }

    def train_step(
        self,
        batch: PreferenceBatch,
    ) -> dict[str, Any]:
        """Perform a single training step."""
        loss_dict = self.loss(batch)
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        accuracy = loss_dict['accuracy']

        loss = get_all_reduce_mean(loss)
        accuracy = get_all_reduce_mean(accuracy)

        return {
            'train/loss': loss.item(),
            'train/accuracy': accuracy.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    def save_reward_scores(self, texts, scores, output_dir, file_prefix="reward_scores"):
        """Save text and corresponding reward scores to JSON file."""
        if not is_main_process():
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine texts and scores
        data = []
        for text, score in zip(texts, scores):
            data.append({
                "text": text,
                "reward_score": float(score)
            })
        
        # Save to JSON file
        output_file = os.path.join(output_dir, f"{file_prefix}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.print(f"Saved reward scores to {output_file}")
        return output_file

    def visualize_reward_distribution(self, chosen_scores, rejected_scores, output_dir):
        """Visualize the distribution of reward scores for chosen vs rejected responses."""
        if not is_main_process():
            return
            
        # Set plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convert to numpy arrays
        chosen_scores = np.array(chosen_scores)
        rejected_scores = np.array(rejected_scores)
        
        # 1. Histogram comparison
        axes[0, 0].hist(chosen_scores, bins=50, alpha=0.7, label='Chosen', color='green', density=True)
        axes[0, 0].hist(rejected_scores, bins=50, alpha=0.7, label='Rejected', color='red', density=True)
        axes[0, 0].set_xlabel('Reward Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Reward Score Distribution: Chosen vs Rejected')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot comparison
        box_data = [chosen_scores, rejected_scores]
        box_labels = ['Chosen', 'Rejected']
        box_colors = ['lightgreen', 'lightcoral']
        
        bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
        axes[0, 1].set_ylabel('Reward Score')
        axes[0, 1].set_title('Box Plot: Reward Score Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. KDE plot
        axes[1, 0].hist(chosen_scores, bins=30, alpha=0.5, label='Chosen', color='green', density=True)
        axes[1, 0].hist(rejected_scores, bins=30, alpha=0.5, label='Rejected', color='red', density=True)
        
        # Add KDE curves
        from scipy.stats import gaussian_kde
        if len(chosen_scores) > 1:
            kde_chosen = gaussian_kde(chosen_scores)
            x_range = np.linspace(min(min(chosen_scores), min(rejected_scores)), 
                                max(max(chosen_scores), max(rejected_scores)), 200)
            axes[1, 0].plot(x_range, kde_chosen(x_range), 'g-', linewidth=2, label='Chosen KDE')
        
        if len(rejected_scores) > 1:
            kde_rejected = gaussian_kde(rejected_scores)
            axes[1, 0].plot(x_range, kde_rejected(x_range), 'r-', linewidth=2, label='Rejected KDE')
        
        axes[1, 0].set_xlabel('Reward Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Kernel Density Estimation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Score difference distribution
        score_diff = chosen_scores - rejected_scores
        axes[1, 1].hist(score_diff, bins=50, alpha=0.7, color='blue', density=True)
        axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=2, label='No Difference')
        axes[1, 1].axvline(x=np.mean(score_diff), color='red', linestyle='-', linewidth=2, 
                          label=f'Mean Diff: {np.mean(score_diff):.3f}')
        axes[1, 1].set_xlabel('Score Difference (Chosen - Rejected)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution of Score Differences')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, 'reward_score_visualization.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        self.logger.print("\n" + "="*60)
        self.logger.print("REWARD SCORE STATISTICS")
        self.logger.print("="*60)
        self.logger.print(f"Chosen responses:")
        self.logger.print(f"  Mean: {np.mean(chosen_scores):.4f}")
        self.logger.print(f"  Std:  {np.std(chosen_scores):.4f}")
        self.logger.print(f"  Min:  {np.min(chosen_scores):.4f}")
        self.logger.print(f"  Max:  {np.max(chosen_scores):.4f}")
        
        self.logger.print(f"\nRejected responses:")
        self.logger.print(f"  Mean: {np.mean(rejected_scores):.4f}")
        self.logger.print(f"  Std:  {np.std(rejected_scores):.4f}")
        self.logger.print(f"  Min:  {np.min(rejected_scores):.4f}")
        self.logger.print(f"  Max:  {np.max(rejected_scores):.4f}")
        
        self.logger.print(f"\nScore Differences (Chosen - Rejected):")
        self.logger.print(f"  Mean: {np.mean(score_diff):.4f}")
        self.logger.print(f"  Std:  {np.std(score_diff):.4f}")
        self.logger.print(f"  Accuracy: {np.mean(score_diff > 0)*100:.2f}%")
        self.logger.print("="*60)
        
        self.logger.print(f"Visualization saved to {plot_file}")
        return plot_file

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        self.logger.print('\n***** Evaluating at the beginning *****')
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        if self.cfgs.train_cfgs.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
        num_correct_predictions = 0
        num_total_predictions = 0

        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
            position=1,
            leave=False,
        )

        rewards = []
        chosen_texts = []
        rejected_texts = []
        chosen_scores = []
        rejected_scores = []
        
        batch = None
        for batch in eval_dataloader:
            output = self.model(**self.infer_batch(batch))
            end_scores = output.end_scores
            higher_end_rewards, lower_end_rewards = end_scores.squeeze(dim=-1).chunk(
                chunks=2, dim=0
            )
            batch_size = higher_end_rewards.size(0)
            num_correct_predictions += (higher_end_rewards > lower_end_rewards).sum()
            num_total_predictions += batch_size

            rewards.extend([higher_end_rewards, lower_end_rewards])
            
            # Extract texts for visualization
            (
                better_input_ids,  # size = (B, L)
                worse_input_ids,  # size = (B, L)
            ) = batch['input_ids'].chunk(chunks=2, dim=0)
            
            # Decode texts
            batch_chosen_texts = self.tokenizer.batch_decode(
                better_input_ids, skip_special_tokens=True
            )
            batch_rejected_texts = self.tokenizer.batch_decode(
                worse_input_ids, skip_special_tokens=True
            )
            
            chosen_texts.extend(batch_chosen_texts)
            rejected_texts.extend(batch_rejected_texts)
            chosen_scores.extend(higher_end_rewards.cpu().tolist())
            rejected_scores.extend(lower_end_rewards.cpu().tolist())

        if batch is None:
            self.logger.print('WARNING: `eval_dataloader` is empty.')
            return {}

        accuracy = num_correct_predictions / num_total_predictions
        accuracy = get_all_reduce_mean(accuracy)

        # Gather rewards from all devices for further analysis
        rewards = torch.cat(rewards, dim=0)
        if is_main_process():
            gathered_rewards = [torch.empty_like(rewards) for _ in range(dist.get_world_size())]
        else:
            gathered_rewards = []
        dist.gather(rewards, gathered_rewards, dst=0)
        if is_main_process():
            rewards = torch.cat(gathered_rewards, dim=0)

        # Save reward scores and create visualizations (only on main process)
        if is_main_process():
            output_dir = os.path.join(self.cfgs.logger_cfgs.output_dir, 'reward_analysis')
            
            # Save chosen and rejected texts with scores
            self.save_reward_scores(chosen_texts, chosen_scores, output_dir, "chosen_responses")
            self.save_reward_scores(rejected_texts, rejected_scores, output_dir, "rejected_responses")
            
            # Create visualizations
            try:
                self.visualize_reward_distribution(chosen_scores, rejected_scores, output_dir)
            except ImportError as e:
                self.logger.print(f"Warning: Could not create visualizations due to missing dependency: {e}")
                self.logger.print("Please install matplotlib and seaborn: pip install matplotlib seaborn scipy")

        self.model.train()
        if self.cfgs.train_cfgs.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Evaluation info
        info = {
            'eval/accuracy': accuracy.item(),
            'eval/reward_mean': rewards.mean().item(),
            'eval/reward_std': rewards.std().item(),
        }

        if is_main_process():
            # Print some examples from the last batch
            max_num_rows = 5
            (
                better_input_ids,  # size = (B, L)
                worse_input_ids,  # size = (B, L)
            ) = batch[
                'input_ids'
            ].chunk(chunks=2, dim=0)
            higher_reward_texts = self.tokenizer.batch_decode(
                better_input_ids[:max_num_rows],
                skip_special_tokens=True,
            )
            lower_reward_texts = self.tokenizer.batch_decode(
                worse_input_ids[:max_num_rows],
                skip_special_tokens=True,
            )

            h_rewards = [f'{reward:.6f}' for reward in higher_end_rewards.tolist()]
            l_rewards = [f'{reward:.6f}' for reward in lower_end_rewards.tolist()]

            title = ', '.join(
                f'{key.rpartition("/")[-1]} = {value:.6f}' for key, value in info.items()
            )
            self.logger.print_table(
                title=f'Evaluation: {title}',
                columns=[
                    'higher-reward response',
                    'lower-reward response',
                    'reward',
                ],
                rows=tuple(
                    zip(
                        higher_reward_texts[:max_num_rows],
                        lower_reward_texts[:max_num_rows],
                        h_rewards[:max_num_rows],
                        l_rewards[:max_num_rows],
                    ),
                ),
                max_num_rows=max_num_rows,
            )

        return info

    def train(self) -> None:
        """Train the model."""
        # Check if we should only evaluate
        eval_only = getattr(self.cfgs.train_cfgs, 'eval_only', False) or self.cfgs.train_cfgs.epochs == 0
        
        if eval_only:
            self.logger.print('***** Running evaluation only *****')
            self.logger.log(self.eval(), step=0)
            return
        
        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.cfgs.train_cfgs.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.cfgs.train_cfgs.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )
        progress_bar.update(self.global_step)

        if self.cfgs.data_cfgs.eval_datasets:
            self.logger.log(self.eval(), step=0)

        remain_epoch = self.cfgs.train_cfgs.epochs - (
            self.global_step // len(self.train_dataloader)
        )

        start_batch_idx = self.global_step % len(self.train_dataloader)

        for epoch in range(int(remain_epoch)):
            self.model.train()
            progress_bar.set_description(
                f'Resuming from checkpoint {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
            )

            for batch_idx, batch in enumerate(self.train_dataloader):
                if epoch == 0 and batch_idx < start_batch_idx:
                    continue

                info = self.train_step(batch)
                torch_gc()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)

                save_interval = (
                    self.cfgs.train_cfgs.epochs
                    * len(self.train_dataloader)
                    // self.cfgs.logger_cfgs.save_total_limit
                )
                if self.global_step % save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.save(tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.cfgs.data_cfgs.eval_datasets
                    and self.cfgs.train_cfgs.eval_strategy == 'steps'
                    and self.global_step % self.cfgs.train_cfgs.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            self.logger.print('\n***** Evaluating...*****')
            self.logger.log(self.eval(), step=self.global_step)

            self.model.tput_timer.update_epoch_count()

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        self.save_transformers(model=model, tag=tag)


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_text', 'rm')
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = RMTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
