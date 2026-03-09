import wandb
wandb.init(project="test-project")
wandb.config.epochs = 10

import random
offset = random.random() / 5
for epoch in range(2, wandb.config.epochs): # 保存したハイパーパラメータを使用
    accuracy = 1 - 2 ** -epoch - random.random() / epoch + offset
    loss = 2 ** -epoch + random.random() / epoch - offset
    
    # WandBへの記録
    wandb.log({"accuracy": accuracy, "loss": loss})
# WandBへ実行の終了を知らせる
wandb.finish()
