# ================================
# 1. Library Installation (Run Once)
# ================================
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install torch peft bitsandbytes evaluate datasets accelerate

# ================================
# 2. Library Imports
# ================================
import torch
import pandas as pd
import numpy as np
import random
from datasets import load_dataset, DatasetDict
from peft import (
    LoraConfig, 
    PeftModel, 
    prepare_model_for_kbit_training, 
    get_peft_model
)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding, 
    EarlyStoppingCallback, 
    BitsAndBytesConfig
)
import bitsandbytes as bnb
import evaluate

# ================================
# 3. Hugging Face Login
# ================================
from huggingface_hub import notebook_login
notebook_login()
# or use: login('hf_your_token_here')

# ================================
# 4. Load IMDb Dataset
# ================================
dataset_imdb = load_dataset("imdb")

# Optional: Reduce Dataset
reduction_rate = 0.1
num_train = int(reduction_rate * dataset_imdb["train"].num_rows)
num_test = int(reduction_rate * dataset_imdb["test"].num_rows)

def select_random_indices(dataset, num_to_keep):
    indices = list(range(dataset.num_rows))
    random.shuffle(indices)
    return indices[:num_to_keep]

train_indices = select_random_indices(dataset_imdb["train"], num_train)
test_indices = select_random_indices(dataset_imdb["test"], num_test)

dataset_imdb = DatasetDict({
    "train": dataset_imdb["train"].select(train_indices),
    "test": dataset_imdb["test"].select(test_indices)
})

# ================================
# 5. Tokenization
# ================================
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = dataset_imdb.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ================================
# 6. Metrics
# ================================
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# ================================
# 7. Label Maps
# ================================
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# ================================
# 8. Quantization Configuration
# ================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ================================
# 9. Load Model
# ================================
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    quantization_config=bnb_config,
    device_map={"": 0}
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ================================
# 10. LoRA Configuration
# ================================
def find_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_modules.add(names[-1])
    lora_modules.discard("lm_head")
    return list(lora_modules)

modules = find_linear_names(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================================
# 11. Training Configuration
# ================================
training_args = TrainingArguments(
    output_dir="./epoch_weights",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
    metric_for_best_model="eval_loss"
)

early_stop = EarlyStoppingCallback(
    early_stopping_patience=1,
    early_stopping_threshold=0.0
)

# ================================
# 12. Train the Model
# ================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stop]
)

trainer.train()

# ================================
# 13. Prediction Function
# ================================
def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs).logits
    y_prob = torch.sigmoid(outputs).tolist()[0]
    return np.round(y_prob, 5)

# ================================
# 14. Evaluate on Test Set
# ================================
df_test = pd.DataFrame(dataset_imdb["test"])
df_test["prediction"] = df_test["text"].map(predict)
df_test["y_pred"] = df_test["prediction"].apply(lambda x: np.argmax(x))
accuracy = (df_test["y_pred"] == df_test["label"]).mean()
print(f"Model Accuracy on Test Data: {accuracy:.4f}")
print(df_test.head())

# ================================
# 15. Save Model
# ================================
new_model = "my_awesome_model"
trainer.model.save_pretrained(new_model)

# ================================
# 16. Load Model (Optional)
# ================================
model = AutoModelForSequenceClassification.from_pretrained(
    new_model,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    quantization_config=bnb_config,
    device_map={"": 0}
)
