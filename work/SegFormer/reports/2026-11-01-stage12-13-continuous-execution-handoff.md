# SegFormer 第12—13阶段连续执行交接

日期：2026-11-01。**这是交给原开发 Agent 的执行指令，本文件的生成不表示两个阶段已经开始或完成。**用户已允许连续完成两个阶段，并在各阶段完成后分别提交、推送 Git，之后再进行独立验收。

> 执行方式：按任务逐项实施、自检、保存证据；支持相关技能的 Agent 可使用 `superpowers:executing-plans` 或 `superpowers:subagent-driven-development`。阶段12通过技术门禁后自动进入13，不必等待用户逐阶段回复。最终独立验收留给用户回来后进行。

**目标：**完成冻结 SegFormer-B0 基线的独立评价、单图推理、可视化、失败分析和实验卡，形成两个可追溯的阶段提交。

**架构：**先用保存的 best、结构和 processor 建立共享推理通道，完整 val 对齐后执行冻结协议的 test；阶段13消费同一批结果和训练日志生成图表、分析与复现材料。两阶段共用原图还原和指标实现，不另起训练实验。

**技术栈：**当前独立 Conda 环境内的 Python、PyTorch、Transformers、NumPy、Pillow、Matplotlib，以及 PowerShell/Git；实际版本在执行时记录，沿用环境，不安装或升级包。

**依据：**[阶段11验收](./2026-11-01-stage11-acceptance.md)、[阶段12详细交接](./2026-11-01-stage12-execution-handoff.md)、[主指南](../../../docs/research/2026-10-31-segformer-baseline-end-to-end-guide.md)步骤12—13、[环境与阶段文档规范](../../../docs/research/2026-10-31-segformer-environment-isolation-and-stage-docs.md)。

## 1. 已授权范围与协议

- 工程根：`E:\桌面\AI\AI_Try\work\SegFormer`；Git 根：`E:\桌面\AI\AI_Try`；资源根：`E:\桌面\AI\segformer_resources`。
- 本次连续实施只覆盖阶段12、13；不启动 DINOv3、融合模型、额外训练、阈值搜索、切块或新的消融实验。
- 正式 run：`runs\pcb1_b0_v2_seed42`，连续完成50轮；冻结 best **epoch39 / step10330**。
- best SHA-256：`192c827bef00607f4258e5fadf0f1d97970ca5cc727a0add749be0fba5997485`。
- 使用该 run 的 `model_config.json`、checkpoint 模型状态、保存的 `processor/`。不重新下载 MiT-B0 替代已训练模型，不 resume 正式训练。
- PCB1 v2 分组、530 train / 132 val / 442 test；test=402正常+40异常。既有清单及原数据只读。
- 512 letterbox；评价 B1、workers0、IEEE FP32，关闭 TF32/autocast；阈值 `>=0.5`，无评价增强、后处理或 TTA。
- 恢复路径固定为 `logits→512→裁padding→原H/W→softmax→阈值`，使用原始二值 GT 计分。背景0/缺陷1；padding255只存在于训练输入，不能把原掩码非零类别误当ignore。
- 第11阶段正式训练与保存推理已验收；续训状态/顺序通过，数值轨迹未通过原容差。此限制保留在实验卡中，不阻止对连续完成的 best 做评价。
- 本文件的连续执行及逐阶段 Git 授权，**覆盖**旧阶段12交接末尾“不要自动commit/push或进入13”和主指南阶段13“先不推送”的旧边界；其他技术要求继续有效。

## 2. 可以直接复制给开发 Agent 的总指令

```text
请按 reports/2026-11-01-stage12-13-continuous-execution-handoff.md，连续完成
SegFormer主指南的阶段12和阶段13。先阅读该文件及它引用的阶段12详细交接、
阶段11最终验收、根/项目AGENTS、主指南步骤12—13和阶段文档模板，再实施。

用户已授权：阶段12完成且技术门禁通过后，分别commit+push，随后自动进入13；
阶段13完成后再次commit+push，然后停止，等待用户回来后独立验收。
无需在这两个阶段之间再次询问是否继续。此前文件中的“不commit/push、不进入13”
已被本次授权覆盖。提交必须明确为“完成（待验收）”，不能声称独立验收通过。

阶段12提交名：第十二阶段完成（待验收）
阶段13提交名：第十三阶段完成（待验收）
只提交AI_Try内本任务的代码、配置、清单、中文学习文档、小报告和精选图。
数据、环境、缓存、权重、全量预测、完整概率留在segformer_resources；不上传Git。
每次push后用git ls-remote实核远端提交，不能只凭本地commit认定保存成功。

严格冻结best39/processor/清单/阈值与精度协议，不重新训练或调参。
先完整132张val逐项对齐，再对442张test做固定协议评价，不以test选择任何设置。
阶段13使用同一评价结果，只分析并提出未来实验假设，不实施新的训练或调阈值。
每阶段都交付中文设计与实现文档，更新README和阶段索引为“已完成，自检通过，待独立验收”。
完成13后给出两次提交SHA与远端核对结果、实际命令、指标、产物位置和待人工复核点。

可自主修复本范围内的普通代码、测试、绘图或路径问题，并保留修复和重跑证据。
若val不一致、权重/数据身份不符、必须更改冻结协议或环境、原文件可能被覆盖，
停止依赖该问题的工作并记录现场；不要伪造通过或跳过门禁。Git推送失败则保留本地
提交，记录准确错误，阶段12未确认远端保存前不进入13。不要force push、重写他人
提交或清理用户已有修改。成功评价后不因分数不理想而重跑选优。
```

## 3. 开始前逐项检查

- [ ] 用项目 runner 建立每个新工具进程的运行上下文；不裸跑 PATH 上的 `python/pip`。沿用现有环境及缓存位置，不修改 base、全局配置、持久环境变量或电源设置。
- [ ] 检查 Git 分支、远端和工作区；当前预期 `master`、`origin=https://github.com/lrw-666/AI_Try.git`。不覆盖已有改动，不执行 `reset --hard`、`clean` 或 force push。记录自己的改动范围；发现无关改动先保留，不混入提交。
- [ ] 核对第11阶段验收记录、实际 best SHA、结构/processor身份、CSV/split_meta SHA；原训练目录开始时建立文件大小/SHA清单，结束时确认未改动。
- [ ] 阅读实际代码和 `--help`。阶段11验收时 `evaluate.py`、`predict.py`、`summarize.py` 尚未实现；不能把文档中的示例命令当作已存在接口。
- [ ] 为本次评价和结果可视化创建新输出目录，拒绝覆盖既有目录。完整预测/中间概率/大图写资源根，不写 Git。
- [ ] 固定展示规则并保存到元数据：test清单内按image_id排序取首6张正常、首6张异常；另取异常IoU最低5张、recall最低5张、正常FP像素最多5张，并列按image_id。允许集合去重，必须保留每个入选理由。规则在读取test推理结果前固定；test图像/GT只在val门禁通过后读取。

## 4. 阶段12：评价与无GT单图预测

以[阶段12详细交接](./2026-11-01-stage12-execution-handoff.md)第2节为完整技术规格，按以下顺序完成；其中旧“完成后不自动提交/进入13”已由本文件第1节覆盖。

### 4.1 实现与自检

- [ ] 实现 `segformer_lab/evaluate.py`、`predict.py` 和必要的共享推理模块；共享已有 `data/geometry/metrics/evaluation` 的数学通道，不维护两套还原或计分公式。
- [ ] 新进程从保存结构构建模型并strict载入best，missing/unexpected均为0，实际使用保存processor；记录真实FP32/B1/eval/inference_mode及归一化证据。不做backward或optimizer.step；只读验收器为核对状态构建优化器是允许的。
- [ ] 增加有意义的测试：无GT预测、原图尺寸/裁padding、0/1导出、NA与int64聚合、非法输入/已有输出拒绝，以及test门禁与best/协议/数据绑定。运行相关测试及完整项目回归，保存真实日志。
- [ ] 完整132张val对齐保存best的逐图计数/尺寸/GT与预测面积/TPFPFNTN/各指标/NA、异常macro及有效分母、正常误报、global。固定6张预览概率比较沿用 `atol=1e-5`、XOR0；同输入evaluate/predict也应一致。任何关键差异先定位修复，不能放宽容差凑通过。
- [ ] val门禁通过后冻结评价源码/配置/权重/processor/清单身份，再对完整442张test流式推理；不把全部float概率常驻内存。默认不保存全量浮点概率，逐图累计指标并写原尺寸0/1 PNG。
- [ ] 输出完整per_image JSON和CSV、global/异常macro与有效分母、402张正常图的误报统计、NA原因、元数据、产物清单。漏图/重复为0；二值PNG为宽1404×高1070、0背景/1缺陷；可另存明确标注的0/255显示图。
- [ ] 实测无需GT的单图预测，输出热力图和叠加图，并与同输入evaluate概率/二值结果核对；不得通过读GT掩盖推理错误。

### 4.2 阶段交付与提交门禁

- [ ] 交付 `docs/stages/12-evaluation-design-and-implementation.md`，写明设计、源码接口、实际技术栈、命令、证据、局限与修订记录。
- [ ] 仓库保存 `reports/stage12-summary.json`、`reports/stage12-unit-tests.txt` 和 `reports/stage12_evidence/` 下的小型原始证据；至少保留完整132/442行逐图统计、聚合指标、val门禁及输入身份，可用无损gzip降低体积。完整预测与概率不进入仓库。
- [ ] 更新README和阶段索引，状态写“已完成，自检通过，待独立验收”；第13阶段此时仍未完成。不把test分数当作第11阶段的val分数，也不宣称工业泛化。
- [ ] 确认原训练run不变，执行第6节Git流程，提交并推送 **`第十二阶段完成（待验收）`**，核实远端SHA。成功后**直接开始阶段13**。

## 5. 阶段13：图表、失败分析与实验卡

### 5.1 结果整理

- [ ] 实现或复用 `segformer_lab/summarize.py`，消费正式run和阶段12评价输出；不调用训练或重新选best，不另算一套不同指标。解析格式与测试记录必须一致，不猜不存在的CLI参数。
- [ ] 训练曲线覆盖CE/Dice/总loss、val global IoU/Dice、异常macro、正常FP和LR，标注best39。保留绘图数值CSV；明确epoch、attempt、successful step，区分最后实际使用LR与更新后保存LR。不得把全正常batch的Dice零误作分割完美。
- [ ] 按已保存的固定/最差规则制作正常、异常和失败图。异常展示原图、GT、0—1固定色标概率、二值预测、TP/FP/FN；TP绿、FP红、FN蓝，图内有图例。正常图也展示零GT与误报，不能只放漂亮的异常图。
- [ ] 保存展示ID、排序值、入选理由、全部异常TP0/预测空ID及全部正常有FP的ID与分母。各列表口径分开，不能把“TP0”等同于“预测为空”。不另用test挑阈值或连通域面积。
- [ ] 若阶段12未保存所选图的浮点概率，可对这些图以同一冻结推理通道重建，仅用于可视化；记录理由/命令/源码与输入SHA，并核对二值结果与阶段12导出一致。不得覆盖既有指标或把多次技术复核中分数最好的一次作为最终成绩。

### 5.2 分析与学习材料

- [ ] 按“观察—证据—可能原因—下一项验证”填写失败分析。优先分析小/细缺陷漏检、正常结构误报与边界粗糙；引用实际GT、原图与512变化、概率或计数，证据不足写假设。保留val上042/051/081既有局限，不伪造唯一成因。
- [ ] 只提出后续切块、分辨率、采样或DINOv3等候选对照，**不在本次实施**。test已经被查看，后续方法调整须披露这一事实，不能继续把该test称为完全未参与开发。
- [ ] 填写 `reports/pcb1_b0_v2_seed42/experiment-card.md`，包括数据/清单/模型来源与SHA、损失、超参数、实际硬件软件版本、50轮与best39、val与test分开列出的指标、正常误报、训练耗时/显存、失败和泛化边界、恢复数值限制。
- [ ] 在同目录输出 `results.md`、`reproduction.md`、`failure-cases.csv` 与少量精选图；写清一台新电脑需要另取的数据/已训练best/结构/processor，如何核SHA、恢复环境并复算val。Git不含模型权重，下载原始MiT-B0不能替代best。
- [ ] 推理延迟不是必做；未测写“未测”，不以阶段11计时或异步调用冒充稳定延迟。本次没有多seed、基线对比或跨数据集验证，不能补写这些结论。
- [ ] 交付 `docs/stages/13-results-design-and-implementation.md`，写实现、命令、图表口径、分析、学习要点和局限；更新README与阶段索引为“已完成，自检通过，待独立验收”。
- [ ] 验证曲线/CSV/实验卡/所有报告指向同一run和best；手工打开代表正常/异常/最差图检查色标、图例、尺寸与覆盖位置。运行实际改动所需的测试及一致性检查，保存真实结果。
- [ ] 执行第6节Git流程，提交并推送 **`第十三阶段完成（待验收）`**。之后停止，不进入新模型或新实验，等待用户统一验收。

## 6. 每个阶段完成后的Git流程

本节是已授权的操作约定，开发 Agent 自行执行；不需要在每阶段重复请求许可。以下命令是示意，文件列表必须来自实际差异。

1. 在 `E:\桌面\AI\AI_Try` 查看 `git status --short --branch`、`git diff --stat`、`git diff --check`，并同时审查原有 `git diff --cached`；只选择本阶段的代码、配置、文档和小型证据。遇到无关改动保留并排除，不使用无差别 `git add -A`。若原索引已有无关暂存，普通 `git commit` 会把它们一起提交：必须保留其内容及暂存状态，使用核准的按路径提交方式隔离本阶段文件；无法安全隔离时停在Git保存门禁，不擅自撤销他人暂存或混入提交。
2. 记录待提交文件的相对路径、大小与SHA，核对全部位于仓库内。排除数据、env/cache、权重、全量预测、完整概率、凭证与临时文件；不借Git LFS上传这些内容。报告中的普通路径/文件SHA可以提交。
3. 精选图及每个归档文件尽量不超过5 MiB、本阶段小型交付总量尽量不超过20 MiB；超限先缩减展示图或无损压缩日志。大原件仍保留资源根，不能为了过检查删除原件或降低计分精度。5 MiB/20 MiB是本项目便携目标，不是GitHub平台限额。
4. `git add -- <逐项审核的文件列表>` 后检查 `git diff --cached --name-status`、文件大小、暂存字节/SHA与工作区一致及 `git diff --cached --check`；普通commit前确认整个暂存区恰好是本阶段审核清单，否则按第1项处理。历史证据不改字节以消除尾空白，不关闭全仓库检查。
5. 以对应的“完成（待验收）”名称提交，并执行 `git push origin master`。若当前仓库状态与预期分支/远端不同，记录差异并定位，不强制切换或强推。
6. 用 `git rev-parse HEAD` 与 `git ls-remote origin refs/heads/master` 比较完整SHA；再确认本地/远端计数一致及自身工作无遗漏。`commit`成功不等于`push`成功，push成功信息也应有远端SHA复核。
7. 每阶段执行时的Git HEAD、未提交源码/配置SHA写在该次元数据中；提交后SHA写入交接记录或最终回复，不为补写本次提交自己的SHA而反复修改已冻结元数据、制造自引用哈希或空提交。

可执行命令示例（在真实测试、文件清单检查通过，且索引没有无关暂存后）：

```powershell
Set-Location -LiteralPath 'E:\桌面\AI\AI_Try'
git status --short --branch
git diff --cached --check
# git add -- <本阶段实际审核通过的路径> 应在commit前完成；此处不能直接粘贴占位符。
git commit -m '第十二阶段完成（待验收）'  # 阶段13改为对应名称
if ($LASTEXITCODE -ne 0) { throw '提交失败，保留现场。' }
git push origin master
if ($LASTEXITCODE -ne 0) { throw '推送失败，不能报告云端已保存。' }
$SegLocalHead = git rev-parse HEAD
$SegRemoteHead = (git ls-remote origin refs/heads/master) -split '\s+' | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or $SegLocalHead -ne $SegRemoteHead) { throw '远端SHA核对失败。' }
git status --short --branch
git rev-list --left-right --count HEAD...origin/master
```

推送遇到暂时网络故障可以有限重试；仍失败时保留本地提交及错误，不反复空提交。阶段12未确认远端保存前，不开始13。本次授权不绕过工具本身的权限机制，若执行工具返回权限阻塞，按实际错误记录，不声称有无人值守权限保证。

## 7. 停止与交接条件

普通实现错误可自行排查修复；只有依赖门禁仍未满足、必须越出本范围、更改冻结协议、破坏性操作或工具权限阻塞等情况才停止依赖步骤。报告写“受阻”，保存命令、错误、已完成项、原文件状态及最小下一步；不能伪写“阶段完成”。如需保存未完成工作，可单独提交“阶段12/13进度保存（受阻）”，不得冒用阶段完成名称。

阶段13完成后，最后回复应包含：

1. 两阶段是否完成，分别哪些自检通过，独立验收仍待进行。
2. 两次阶段提交的完整SHA、提交名、远端master核对结果。
3. 完整val对齐结果、442张test的global/异常macro/正常误报，明确各分母与NA。
4. 两份阶段学习文档、实验卡、逐图统计、精选图与大产物的准确位置，以及实际命令。
5. 普通修复/技术重跑记录、保留局限、需要用户回来目视检查的图和未完成事项。

**本次验收不自动升级为论文研究完成，也不自动进入DINOv3。**后续由用户回来后发起独立验收，再决定新的研究任务。
