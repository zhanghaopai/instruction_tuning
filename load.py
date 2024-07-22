import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.get("base", "offline_model"),
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # 嵌入量化，会进行两轮量化，第二轮量化会为每个参数额外节省0.4bits
        ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return model


def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.get("base", "offline_model"),
        trust_remote_code=True
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def load_lora_model(model, config):
    # 添加fine-tuning层
    peft_config = LoraConfig(
        task_type=config.get("lora", "task_type"),
        r=4,  # 低秩
        lora_alpha=config.getint("lora", "alpha"),   # 缩放因子，
        lora_dropout=config.getfloat("lora", "dropout"),
        bias="none",
        target_modules=['Wqkv', 'out_proj']        # 仅训练注意力矩阵和输出权重
    )
    # 添加适配器
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    return model
