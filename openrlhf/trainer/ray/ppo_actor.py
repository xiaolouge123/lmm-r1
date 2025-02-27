import itertools
import math
import os
import socket
from typing import Callable, Dict, List
from tqdm import tqdm

import deepspeed
import ray
import torch
import torch.distributed
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import PPOTrainer
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker
from openrlhf.utils import blending_datasets, get_tokenizer, get_vl_processor
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray


from .launcher import BasePPORole
from .utils import get_physical_gpu_id


class ActorPPOTrainer(PPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        eval_dataloader=None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote
        self.eval_dataloader = eval_dataloader

        print("ActorPPOTrainer processor: ", self.processor)
        print("ActorPPOTrainer data_processor: ", self.data_processor)

        self.experience_maker = RemoteExperienceMaker(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.data_processor,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
        )

        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.strategy.args.colocate_all_models:
            self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)

        torch.distributed.barrier()

    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()
        status = {}

        # 2. triger remote critic model training
        if self.critic_train_remote:
            critic_status_ref = self.critic.fit.remote()
            # sync for colocate_all_models
            if self.strategy.args.colocate_all_models:
                status.update(ray.get(critic_status_ref))

        if self.strategy.args.colocate_all_models:
            torch.distributed.barrier()

        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            status.update(super().ppo_train(global_steps))
            torch.cuda.empty_cache()

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                # vLLM wakeup
                if self.strategy.args.vllm_enable_sleep:
                    torch.distributed.barrier()
                    torch.cuda.synchronize()
                    if torch.distributed.get_rank() == 0:
                        refs = []
                        for engine in self.vllm_engines:
                            refs.append(engine.wake_up.remote())
                        ray.get(refs)
                torch.distributed.barrier()
                self._broadcast_to_vllm()

        # 5. wait remote critic model training done
        if self.critic_train_remote and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        # 6. eval metrics if eval_data is provided
        if self.eval_dataloader is not None:
            status.update(self.evaluation(global_steps))

        return status

    def evaluation(self, global_steps):
        """评估当前模型性能

        使用 vllm 引擎加速文本生成
        使用远程奖励模型计算奖励

        Args:
            global_steps: 当前全局训练步数

        Returns:
            包含评估指标的字典
        """
        if self.eval_dataloader is None:
            return {}

        if self.vllm_engines is not None:
            self.actor.eval()
            torch.cuda.empty_cache()

            samples_list = []

            for batch in self.eval_dataloader:
                samples_list.extend(self._generate_vllm(batch))

            all_lengths = []
            all_response_lengths = []
            for samples in samples_list:
                all_lengths.append(samples.total_length.mean().item())
                all_response_lengths.append(samples.response_length.mean().item())

            r_refs = []
            if self.remote_rm_url:
                for samples in tqdm(samples_list, desc="Processing samples for evaluation"):
                    if not self.packing_samples:
                        queries = self.tokenizer.batch_decode(samples.sequences, skip_special_tokens=False)
                    for rm in self.remote_rm_url:
                        r = remote_rm_fn_ray.remote(rm, queries=queries, prompts=samples.prompts)
                        r_refs.append(r)
            all_rewards = ray.get(r_refs)
            all_rewards = self.reward_fn(all_rewards) if len(all_rewards) > 0 else all_rewards[0]

            # 计算平均指标
            metrics = {}
            if all_rewards:
                metrics["eval/mean_reward"] = sum(all_rewards) / len(all_rewards)
                metrics["eval/max_reward"] = max(all_rewards)
                metrics["eval/min_reward"] = min(all_rewards)

            if all_lengths:
                metrics["eval/mean_total_length"] = sum(all_lengths) / len(all_lengths)
            if all_response_lengths:
                metrics["eval/mean_response_length"] = sum(all_response_lengths) / len(all_response_lengths)

            # 记录评估步骤
            metrics["eval/step"] = global_steps

            self.actor.train()
            return metrics
        else:
            return {}

    def _generate_vllm(self, all_prompts: List[str], **kwargs):
        from vllm import SamplingParams
        from openrlhf.trainer.ray.ppo_utils.experience_maker import Samples

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
        refs = []
        if self.data_processor is None:
            # For LLM
            all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
            for i, llm in enumerate(llms):
                prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
                print(f"DEBUG: Engine {i} receives {len(prompt_token_ids)} prompt token ids in eval")  # Debug
                refs.append(
                    llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )
        else:
            # For VLM
            for i, llm in enumerate(llms):
                messages = all_prompts[i * batch_size : (i + 1) * batch_size]
                print(f"DEBUG: Engine {i} receives {len(messages)} multi-modal messages in eval")  # Debug
                if messages:
                    prompts = self.data_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    images = [self.data_processor.get_images_from_messages(m) for m in messages]
                    print(f"DEBUG: Engine {i} receives {len(images)} images in eval, {images[0]}")  # Debug
                    vllm_inputs = [
                        {
                            "prompt": p,
                            "multi_modal_data": {"image": imgs} if imgs else None,
                            "mm_processor_kwargs": {
                                "min_pixels": int(os.getenv("MIN_PIXELS", 4 * 28 * 28)),
                                "max_pixels": int(os.getenv("MAX_PIXELS", 640 * 28 * 28)),
                            },
                        }
                        for p, imgs in zip(prompts, images)
                    ]
                    refs.append(
                        llm.add_requests_vlm.remote(
                            rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs
                        )
                    )

        ray.get(refs)

        # Make sure all requests are sent.
        torch.distributed.barrier()

        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        outputs_from_engines = ray.get(all_output_refs)

        all_outputs = sum(outputs_from_engines, [])
        print(f"DEBUG: Combined all_outputs count in eval: {len(all_outputs)}")

        assert len(all_outputs) == len(
            all_prompts
        ), f"not equal error in eval:len(all_outputs) = {len(all_outputs)}, len(all_prompts) = {len(all_prompts)}"

        samples_list = []
        for i in range(0, len(all_outputs), args.eval_batch_size):
            outputs = all_outputs[i : i + args.eval_batch_size]
            prompts = all_prompts[i : i + args.eval_batch_size]
            if not self.packing_samples:
                print(f"DEBUG: no packing samples in eval")
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)
                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                visual_inputs = None
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        visual_inputs=visual_inputs,
                    )
                )

        return samples_list

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in self.vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective

                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        ray.get(refs)
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=count == num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ref = self.critic.save_checkpoint.remote(tag)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.processor or self.tokenizer,
                save_path,
            )
        # wait
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ray.get(ref)
        torch.distributed.barrier()


@ray.remote(num_gpus=1)
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(actor)
        # Support freeze some parameter
        if hasattr(strategy.args, "freeze_prefix") and strategy.args.freeze_prefix:
            frozen_count = 0
            total_params = 0
            for name, param in actor.model.named_parameters():
                total_params += 1
                if any(name.startswith(prefix) for prefix in strategy.args.freeze_prefix):
                    param.requires_grad = False
                    frozen_count += 1
            strategy.print(
                f"Froze {frozen_count}/{total_params} parameters based on prefixes: {strategy.args.freeze_prefix}"
            )

        # configure tokenizer
        if args.train_vlm:
            self.processor = get_vl_processor(
                pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = get_tokenizer(
                pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
            )

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # prepare_datasets
        self.prepare_datasets()

        # configure scheduler
        self.num_update_steps_per_episodes = (
            len(self.prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        )
        max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        actor_scheduler = get_scheduler(
            "cosine_with_min_lr",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.restore_ckpt_path:
            ckpt_path = os.path.join(args.restore_ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            if args.restore_ckpt_tag:
                _, states = strategy.load_ckpt(self.actor.model, ckpt_path, tag=args.restore_ckpt_tag)
            else:
                _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        self.prompts_dataset = PromptDataset(
            prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, args.rollout_batch_size // strategy.world_size, True, True
        )
        if args.eval_data:
            eval_data = blending_datasets(
                args.eval_data,
                args.eval_data_probs,
                strategy,
                args.seed,
                return_eval=False,
            )
            self.eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            self.eval_dataloader = strategy.setup_dataloader(
                self.eval_dataset, args.eval_batch_size // strategy.world_size, True, True
            )

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
                train_split=args.pretrain_split,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(
                    range(
                        min(
                            len(pretrain_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt
                        )
                    )
                ),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # fro GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
            eval_dataloader=self.eval_dataloader,
        )

        # broadcast checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
            torch.distributed.barrier()
            trainer._broadcast_to_vllm()

        trainer.fit(
            args,
            self.prompts_dataloader,
            self.pretrain_dataloader,
            self.consumed_samples,
            self.num_update_steps_per_episodes,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.processor or self.tokenizer,
            args.save_path,
        )
