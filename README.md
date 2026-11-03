# SAHF 光伏区域分割

SAHF_Net 是唯一活跃主项目。数据采用 spatial_1km_v2：Train/Val/Test 为434/93/93景，训练使用256×256九窗口，验证使用512×512整景。每次本地代码和配置修改完成后同步服务器并核对SHA256。

## 当前入口

日常实验可直接编辑 `main.py` 顶部的模型、配置、配对模式、种子和运行编号，然后在服务器执行 `python main.py`。默认选择春季基线。每次运行按协议和运行编号归档：

```text
Output/spatial_1km_v2/<run_id>/
  checkpoints/              best.pt、last.pt
  diagnostics/              验证表格、对照报告、loss_curve.png
  train.log
  training_history.csv
  training_summary.json
  resolved_experiment.yaml
  run_identity.json
  class_statistics.json
```

训练入口使用 `configs/experiment.yaml`。新运行统一写入 `Output/spatial_1km_v2`。续训需设 `RESUME = True` 并保留原运行编号、模型、配置路径及内容和种子。改名前的运行如需续训，需使用归档的原训练入口，并恢复当时的配置及其路径，以匹配旧签名。跨目录对照可在 `COMPARE_TO` 填参考运行目录或逐景 CSV 的绝对路径。

`main.py` 支持 `spring_only`、`autumn_only`、`concat`、`aux_only` 四个模型，共享本地预训练MiT-B1及六波段输入。默认配置为 `configs/experiment.yaml`，设备明确为CUDA。`--seed`覆盖种子；`--run-id`指定独立运行；`--resume`要求代码、配置和数据签名匹配。

```powershell
python main.py --model spring_only --run-id spv2_spring_only_s42
```

以上是使用示例，不代表已经运行。当前任务先核实数据与代码，不自动启动模型比较。

训练按固定0.5阈值的Val micro IoU选择best checkpoint，结束后生成逐景、分省、目标规模、面积误差、阈值校准及可选配对比较。`--compare-to`接受参考运行或逐景CSV。比较要求同场景、同有效域、同标签、同阈值、同数值精度及同地理组；旧表缺来源字段时需使用其对应模型重新生成。阈值校准和Oracle仅用于Val诊断，不是独立Test成绩。

当前没有独立Test评价入口。模型分析中的推理耗时没有预热/重复试验，不作为正式效率基准。

## 版本与实验记录

2026-11-03审查发现当前ConvNormAct已改为包含block的模块，state_dict键与旧Sequential实现不同，因此四种架构版本标识更新为v2。历史v1权重应使用保存的历史实现读取，不隐式转换。新空间划分也要求从预训练初始化重新训练，旧模型不能直接作为新Test的独立结果。

旧README和本次整改前源码保存在 `../research/code_audit/current_before/SAHF_Net/`，历史实验结果保持原样。完整审查报告见 `../research/code_audit/CODE_AUDIT_REPORT.md`。

当前主指标为前景micro IoU，同时报告precision/recall、正景macro和分组误差。全部场景含目标，当前结果不能证明纯负地区误报水平；空间1km分组不等同未见省份验证。数据来源日期、场站归属和标注时间依据仍待补充。
