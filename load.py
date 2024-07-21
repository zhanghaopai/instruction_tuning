import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load(config):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.get("base", "offline_model"),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.get("base", "offline_model"),
        trust_remote_code=True
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return model, tokenizer


def load_fine_tuning_model(model, config):
    # 添加fine-tuning层
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=config.getint("lora", "alpha"),
        lora_dropout=config.getfloat("lora", "dropout"),
        bias="none",
        task_type=config.get("lora", "task_type"),
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense',
            'fc1',
            'fc2',
        ]
    )
    model = get_peft_model(model, peft_config)
    return model, peft_config
