# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig


def build_speaker_id_map(train_data, speaker_id_start):
    speaker_names = sorted({line["speaker_name"] for line in train_data})
    speaker2id = {
        speaker_name: speaker_id_start + i
        for i, speaker_name in enumerate(speaker_names)
    }
    return speaker2id


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_id_start", type=int, default=3000)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    model_path = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(model_path)

    train_data = open(args.train_jsonl, "r", encoding="utf-8").readlines()
    train_data = [json.loads(line) for line in train_data]

    for i, line in enumerate(train_data):
        if "speaker_name" not in line:
            raise ValueError(
                f"Missing `speaker_name` in sample index {i}. "
                "Please generate data via prepare_multispeaker_data.py first."
            )

    speaker2id = build_speaker_id_map(train_data, args.speaker_id_start)

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(qwen3tts.model, optimizer, train_dataloader)

    speaker_embedding_sum = {}
    speaker_embedding_count = {}

    model.train()

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]
                speaker_names = batch["speaker_names"]

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

                for idx, spk_name in enumerate(speaker_names):
                    if spk_name is None:
                        raise ValueError("speaker_name should not be None for multi-speaker training")
                    spk_id = speaker2id[spk_name]
                    input_codec_ids[idx, 6] = spk_id
                    input_codec_embedding[idx, 6, :] = speaker_embedding[idx]

                    if spk_name not in speaker_embedding_sum:
                        speaker_embedding_sum[spk_name] = speaker_embedding[idx].to(torch.float32)
                        speaker_embedding_count[spk_name] = 1
                    else:
                        speaker_embedding_sum[spk_name] += speaker_embedding[idx].to(torch.float32)
                        speaker_embedding_count[spk_name] += 1

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(model_path, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(model_path, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                speaker_name: speaker2id[speaker_name]
                for speaker_name in sorted(speaker2id.keys())
            }
            talker_config["spk_is_dialect"] = {
                speaker_name: False
                for speaker_name in sorted(speaker2id.keys())
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict["talker.model.codec_embedding.weight"]
            for speaker_name, speaker_id in speaker2id.items():
                avg_embedding = speaker_embedding_sum[speaker_name] / speaker_embedding_count[speaker_name]
                state_dict["talker.model.codec_embedding.weight"][speaker_id] = (
                    avg_embedding.to(weight.device).to(weight.dtype)
                )

            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)


if __name__ == "__main__":
    train()
