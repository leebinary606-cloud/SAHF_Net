# MiT-B1 权重

运行 `python 02_prepare_pretrained_backbone.py`。脚本优先验证本目录 `mit-b1/`，不存在时从官方 `nvidia/mit-b1` 获取并保存。离线服务器应预先复制完整 Hugging Face `mit-b1` 目录（至少配置和权重文件）。缺失时脚本会停止，绝不退回随机初始化。

第一层按 `B2←Blue, B3←Green, B4←Red, B8/B11/B12←RGB均值` 适配，来源及每个文件的 SHA256 写入 `reports/data/pretrained_backbone.json`。
