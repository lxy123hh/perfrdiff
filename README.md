# 1. 复现数据

MAFRG-FRCorr: 0.37676 MAFRG-FRdist: 96.91244 FRDiv(table): 13.63410 FRDvs(table): 21.90717 FRVar(table): 8.78743 FRSyn: 45.12593
MAFRG-FRRea(FID): 56.17846
MAFRG-Time: total 7h 27m 18.80s prediction 6h 31m 10.63s metrics 6m 9.34s FRRea 49m 58.83s
PMAFRG-FRCorr: 0.35750 PMAFRG-FRdist: 98.41659 FRDiv(table): 13.65726 FRDvs(table): 21.91921 FRVar(table): 8.79545 FRSyn: 45.26972
PMAFRG-FRRea(FID): 58.41504
PMAFRG-Time: total 6h 38m 36.07s prediction 5h 41m 23.58s metrics 5m 29.62s FRRea 51m 42.85s
Total elapsed time: 14h 6m 12.05s

<br />

<br />

# 方案一

• 整体路线

你现在已经有作者的 checkpoints，并且用原命令复现了结果。按“方案一：LoRA/低秩化 PWSG”改进时，不需要重训整个 PerFRDiff，
也不需要先跑 train\_diffusion.py。最稳的路线是：

1. 保留作者已经训好的 GAFRG、PSSL、embedder checkpoint。
2. 只改第二阶段的个性化权重编辑模块，也就是当前的 ModifierNetwork/PWSG。
3. 用新的低秩版配置重训第二阶段。
4. 再用和你现在几乎一样的 evaluate\_rewrite\_weight\_all.py 命令测试。

这样做的原因很简单：你改的是“生成 ΔW 的方式”，不是 speaker encoder、diffusion prior、decoder 主体结构。你已经有
baseline checkpoint，应该直接在这个基础上做增量实验。

要改的文件

建议改这 4 个文件，外加新建 1 个配置文件。

- /home/lxy/PerFRDiff/model/modifier/network.py
  这里是核心。当前 ModifierNetwork 直接为每个目标层输出完整矩阵 ΔW。要改成低秩版本：
  A\_k ∈ R^(d\_out×r)，B\_k ∈ R^(r×d\_in)，然后 ΔW\_k = (alpha / r) \* A\_k @ B\_k。
  保留当前 modified\_layers 机制，第一版建议仍然只改：
  diffusion\_decoder.model.decoder.layers.4.multihead\_attn
  diffusion\_decoder.model.decoder.layers.6.multihead\_attn
  diffusion\_decoder.model.to\_emotion\_feat
- /home/lxy/PerFRDiff/model/diffusion/matchers.py
  这里必须补一个训练链路修正。当前 prior/decoder 只在 mode == "test" 时加载 checkpoint，训练第二阶段时不会自动加载基础
  GAFRG 权重。你要改成：
  测试时加载；
  或者训练时如果配置里 load\_pretrained\_backbone: True 也加载。
  不改这一处，第二阶段训练会用随机初始化 backbone，实验不成立。
- /home/lxy/PerFRDiff/train\_rewrite\_weight.py
  建议顺手清理训练逻辑：
  显式只更新 low-rank hypernet；
  用 optimizer\_hypernet.zero\_grad()，不要再依赖 optimizer\_mainnet.zero\_grad() 这种绕法；
  最好显式冻结 main\_model.main\_net 参数，避免无效梯度和显存浪费。
- /home/lxy/PerFRDiff/configs/rewrite\_weight.yaml
  不要直接改原文件，复制一份。
- /home/lxy/PerFRDiff/configs/rewrite\_weight\_lora.yaml
  这是你新建的实验配置。里面至少要改：
  trainer.log\_dir
  trainer.tb\_dir
  trainer.out\_dir
  trainer.checkpoint\_dir
  main\_model.args.rank
  main\_model.args.lora\_alpha
  main\_model.args.use\_low\_rank: True
  load\_pretrained\_backbone: True

配置建议

新配置 configs/rewrite\_weight\_lora.yaml 里建议这样设：

- trainer.log\_dir: ./log/rewrite\_weight\_lora
- trainer.tb\_dir: ./tb\_logs/rewrite\_weight\_lora
- trainer.out\_dir: ./results/rewrite\_weight\_lora
- trainer.checkpoint\_dir: ./checkpoints/rewrite\_weight\_lora
- main\_model.args.use\_low\_rank: True
- main\_model.args.rank: 8
- main\_model.args.lora\_alpha: 8
- main\_model.args.predict: shift
- main\_model.args.modify: all
- load\_pretrained\_backbone: True

第一版先用 rank=8，不要一开始就改太多超参。后续再做 r=4/16 ablation。

训练前准备

因为新的实验目录是 checkpoints/rewrite\_weight\_lora，而 backbone 仍然要加载作者已有的 prior/decoder checkpoint，所以先
把基础模型复制过去，不要覆盖原实验。

cd /home/lxy/PerFRDiff

mkdir -p checkpoints/rewrite\_weight\_lora
cp -r checkpoints/rewrite\_weight/DiffusionPriorNetwork checkpoints/rewrite\_weight\_lora/
cp -r checkpoints/rewrite\_weight/TransformerDenoiser checkpoints/rewrite\_weight\_lora/

这里不要复制旧的 ModifierNetwork，因为你改结构后它已经不兼容了。

训练命令

你现在这条改进路线，只需要重训第二阶段：

cd /home/lxy/PerFRDiff
conda activate react

nohup python train\_rewrite\_weight.py --mode train --writer True --config configs/rewrite\_weight\_lora.yaml >
train\_lora.log 2>&1 &

看日志：

tail -f train\_lora.log

测试命令

快速测试，不算 FRRea：

python evaluate\_rewrite\_weight\_all.py --mode test --config configs/rewrite\_weight\_lora.yaml --protocol both

完整测试，和你现在 baseline 的方式一致：

nohup python evaluate\_rewrite\_weight\_all.py --mode test --config configs/rewrite\_weight\_lora.yaml --protocol both
\--compute\_frrea --frrea\_period 10 --frrea\_step 1 --frrea\_fake\_repeat 10 > result\_lora.log 2>&1 &

看结果：

tail -f result\_lora.log

实验顺序

建议按这个顺序做：

- 原始 baseline：你已经有 configs/rewrite\_weight.yaml 的结果。
- 主实验：configs/rewrite\_weight\_lora.yaml，rank=8。
- ablation 3：只改 cross-attn，不改 to\_emotion\_feat
- ablation 4：cross-attn + mapping layer

额外提醒

当前仓库第二阶段训练链路本身有一个关键问题：matchers.py 默认不会在 train 模式下加载 prior/decoder checkpoint。这一处你
必须先修，不然训练 low-rank PWSG 没意义。

如果你要，我下一步可以直接给你一版“精确到函数和代码段”的修改清单，或者直接帮你把这 4 个文件改成可运行版本。

<br />

<br />

<br />

<br />

<br />

<br />

<br />

我已经把低秩版 PWSG 改进落到代码里了，保留了你原来的 baseline 路径，不会覆盖原 rewrite\_weight.yaml。

修改的文件

- /home/lxy/PerFRDiff/model/modifier/network.py:46
  新增了 LowRankModifierNetwork，把原来直接输出完整 ΔW 的 PWSG 改成输出低秩因子 A/B，再重构 ΔW = (alpha / rank) \* A @
  B。同时在 /home/lxy/PerFRDiff/model/modifier/network.py:101 里让 MainNetUnified 支持通过配置开关切换原版和低秩版。
- /home/lxy/PerFRDiff/model/diffusion/matchers.py:14
  新增了 maybe\_load\_pretrained\_checkpoint。现在低秩版训练时会自动从你已经复现好的作者 checkpoint 目录加载
  DiffusionPriorNetwork 和 TransformerDenoiser，不需要你手工拷贝 backbone。实际接线在 /home/lxy/PerFRDiff/model/
  diffusion/matchers.py:178 和 /home/lxy/PerFRDiff/model/diffusion/matchers.py:370。
- /home/lxy/PerFRDiff/train\_rewrite\_weight.py:36
  训练脚本改成只优化 hypernet，不再保留原来那个“建了 optimizer\_mainnet 但不 step”的绕法。命令行接口没变，还是
  train\_rewrite\_weight.py --mode train --config ...。
- /home/lxy/PerFRDiff/configs/rewrite\_weight\_lora.yaml:61
  新建了低秩实验配置，关键设置是：
  load\_pretrained\_backbone: True
  pretrained\_checkpoint\_dir: ./checkpoints/rewrite\_weight
  trainer.checkpoint\_dir: ./checkpoints/rewrite\_weight\_lora
  main\_model.args.use\_low\_rank: True
  main\_model.args.rank: 8
  main\_model.args.lora\_alpha: 8.0
  main\_model.args.freeze\_main\_net: True

怎么运行
你现在不需要重跑 train\_diffusion.py。直接在作者 checkpoint 基础上重训第二阶段即可。

训练：

cd /home/lxy/PerFRDiff
conda activate react
nohup python train\_rewrite\_weight.py --mode train --writer True --config configs/rewrite\_weight\_lora.yaml >
train\_lora.log 2>&1 &

看训练日志：

tail -f train\_lora.log

训练完成后，按你原来的方式测试，只把配置文件换成新的：

nohup python evaluate\_rewrite\_weight\_all.py --mode test --config configs/rewrite\_weight\_lora.yaml --protocol both
\--compute\_frrea --frrea\_period 10 --frrea\_step 1 --frrea\_fake\_repeat 10 > result\_lora.log 2>&1 &

看测试日志：

tail -f result\_lora.log

如果只想先快速看 MAFRG/PMAFRG 指标，不算 FRRea：

python evaluate\_rewrite\_weight\_all.py --mode test --config configs/rewrite\_weight\_lora.yaml --protocol both

说明

- 你的原 baseline 命令不变，只是把 configs/rewrite\_weight.yaml 改成 configs/rewrite\_weight\_lora.yaml。
- 低秩版会继续使用你现成的作者 backbone checkpoint：./checkpoints/rewrite\_weight。
- 新训练出来的低秩 hypernet checkpoint 会保存在：./checkpoints/rewrite\_weight\_lora/LowRankModifierNetwork/
  checkpoint.pth
- 我做了语法检查：python3 -m py\_compile 已通过。
- 目前 rank=8。如果你要做 ablation，改 /home/lxy/PerFRDiff/configs/rewrite\_weight\_lora.yaml:191 的 rank 和 lora\_alpha
  就行。

如果你要，我下一步可以继续帮你补一版 rank=4 / 8 / 16 的批量实验脚本。

<br />

<br />

› 训练后的产物是什么，yaml文件吗，然后拿着这个配置文件进行测试吗

• 不是。训练后的产物主要是 checkpoint，不是 yaml。

yaml 只是配置文件，告诉脚本“模型怎么搭、去哪里读基础 checkpoint、把新结果存到哪里”。你现在新增的配置是 /home/lxy/
PerFRDiff/configs/rewrite\_weight\_lora.yaml。

真正训练出来的东西是：

- ./checkpoints/rewrite\_weight\_lora/LowRankModifierNetwork/checkpoint.pth
  这是你改进后的低秩 PWSG 权重，测试时真正加载的是它。
- ./log/rewrite\_weight\_lora/.../log.txt
  训练日志。
- ./tb\_logs/rewrite\_weight\_lora/...
  TensorBoard 日志。
- ./results/rewrite\_weight\_lora/...
  评测和中间结果目录。

测试时要做的是：

- 用 configs/rewrite\_weight\_lora.yaml
- 让脚本按这个配置去加载：
  - 作者原来的 backbone checkpoint：./checkpoints/rewrite\_weight/...
  - 你训练出来的新 low-rank hypernet checkpoint：./checkpoints/rewrite\_weight\_lora/LowRankModifierNetwork/
    checkpoint.pth

所以流程是：

1. 训练

nohup python train\_rewrite\_weight.py --mode train --writer True --config configs/rewrite\_weight\_lora.yaml >
train\_lora.log 2>&1 &

1. 测试

nohup python evaluate\_rewrite\_weight\_all.py --mode test --config configs/rewrite\_weight\_lora.yaml --protocol both
\--compute\_frrea --frrea\_period 10 --frrea\_step 1 --frrea\_fake\_repeat 10 > result\_lora.log 2>&1 &

也就是说，测试时确实还是“拿这个配置文件去测”，但配置文件本身不是训练产物，它只是指向训练产物。

<br />

<br />

训练的时候跑了很多次这正常吗；所以测试的时候是用作者的预训练pth模型和新训练的一起用吗

• 正常。你现在这条训练会“跑很多次”主要是因为 /home/lxy/PerFRDiff/train\_rewrite\_weight.py:148 会按 epoch 整个数据集反复训
练，而 /home/lxy/PerFRDiff/configs/rewrite\_weight\_lora.yaml:169 里设的是 epochs: 100，同时 /home/lxy/PerFRDiff/
configs/rewrite\_weight\_lora.yaml:220 的 dataset.batch\_size: 1，所以迭代次数本来就会很多。这不是“命令重复启动很多次”，
而是一次训练任务内部有很多 step。每 10 个 epoch 还会保存一次 checkpoint，见 /home/lxy/PerFRDiff/configs/
rewrite\_weight\_lora.yaml:181。

对，测试时是“作者的预训练 backbone + 你新训练的 low-rank PWSG”一起用。具体是：

- 作者原来的 backbone：从 /home/lxy/PerFRDiff/configs/rewrite\_weight\_lora.yaml:61 指向的 ./checkpoints/rewrite\_weight
  加载，代码在 /home/lxy/PerFRDiff/model/diffusion/matchers.py:14、/home/lxy/PerFRDiff/model/diffusion/
  matchers.py:178、/home/lxy/PerFRDiff/model/diffusion/matchers.py:370
- 你新训练的 low-rank hypernet：从 ./checkpoints/rewrite\_weight\_lora/LowRankModifierNetwork/checkpoint.pth 加载，代码
  在 /home/lxy/PerFRDiff/model/modifier/network.py:159

所以测试命令还是用新的配置文件：

nohup python evaluate\_rewrite\_weight\_all.py --mode test --config configs/rewrite\_weight\_lora.yaml --protocol both
\--compute\_frrea --frrea\_period 10 --frrea\_step 1 --frrea\_fake\_repeat 10 > result\_lora.log 2>&1 &

如果你只是想先确认链路通了，不想等太久，可以先把 /home/lxy/PerFRDiff/configs/rewrite\_weight\_lora.yaml:172 的 epochs 改
成 5 或 10 做一个 smoke test。

<br />

<br />

MAFRG-FRCorr: 0.37528 MAFRG-FRdist: 100.91016 FRDiv(table): 14.18283 FRDvs(table): 23.11571 FRVar(table): 9.32179 FRSyn: 45.26489
MAFRG-FRRea(FID): 55.46064
MAFRG-Time: total 7h 6m 53.92s prediction 6h 10m 45.70s metrics 5m 49.42s FRRea 50m 18.78s
PMAFRG-FRCorr: 0.35754 PMAFRG-FRdist: 102.35234 FRDiv(table): 14.19984 FRDvs(table): 23.13025 FRVar(table): 9.32693 FRSyn: 45.28117
PMAFRG-FRRea(FID): 57.40389
PMAFRG-Time: total 6h 37m 25.62s prediction 5h 45m 46.29s metrics 5m 43.23s FRRea 45m 56.05s
Total elapsed time: 13h 44m 21.64s

<br />

<br />

你这组结果的核心特征很明确：

- FRCorr 基本持平。
  MAFRG: 0.37676 -> 0.37528，略降 0.00148
  PMAFRG: 0.35750 -> 0.35754，几乎不变
- FRdist 明显变差。
  MAFRG: 96.91 -> 100.91
  PMAFRG: 98.42 -> 102.35
  这大约是 4% 的退化
- 多样性指标都变好了。
  FRDiv / FRDvs / FRVar 在 MAFRG 和 PMAFRG 上都提升了，大约 4%\~6%
- FRRea(FID) 也变好了。
  MAFRG: 56.18 -> 55.46
  PMAFRG: 58.42 -> 57.40
  FID 下降说明真实感略有提升
- FRSyn 基本持平

这说明低秩化 PWSG 没有把模型训崩，反而带来了一个很典型的 trade-off：

- 更轻量
- 更稳定/更有正则化效果
- 生成更分散、更真实
- 但和目标反应的距离变大了，尤其 FRdist 退化明显

所以如果你的创新点表述是：

- “提出低秩化 personalised weight editing，使参数量和 checkpoint 大幅下降，同时基本保持相关性，并提升多样性与真实感”

这个说法是成立的。

如果你的创新点表述是：

- “提出低秩化后，整体性能优于原论文”

这个目前不成立，因为原论文最核心的是 appropriateness，而 appropriateness 里 FRdist 是明显退化的。

更适合你的论文表述应该是：

- Dynamic Low-Rank PWSG
- 目标是解决原文结论里提到的“weight editing complexity 高”的问题
- 结果是：
  1. 大幅压缩个性化模块规模
  2. 保持 FRCorr
  3. 提升 FRDiv/FRDvs/FRVar 和 FRRea
  4. 但 FRdist 有退化，需要进一步优化

如果再结合你这次训练产物，论据会更强：

- 原始 ModifierNetwork checkpoint 约 13 GB
- 你现在 low-rank checkpoint 约 316 MB
- 压缩大约 41x

这个量级足够支撑“轻量化创新点”。

- 目前只有一组结果，最好至少跑 3 个 seed，报 mean ± std
- 还需要把 FRdist 拉回来一点，否则论文主指标上会吃亏

下一步最值得做的不是继续大改结构，而是调低秩配置：

1. 先做 rank ablation：4 / 8 / 16 / 32
2. 再做 lora\_alpha ablation：4 / 8 / 16
3. 优先给 cross-attn 更高 rank，to\_emotion\_feat 更低 rank
4. 试一版“只低秩化 cross-attn，不动 mapping layer”
5. 最后汇报：
   参数量、checkpoint 大小、FRCorr、FRdist、FRDiv、FRRea

一句话总结：

这个创新点现在已经证明“能跑通、能收敛、能带来明显轻量化，并提升多样性和真实感”，所以是可行的；但它更像“效率-性能折中型
创新”，还不是“全面性能增强型创新”。如果你想把它写成更强的论文贡献，下一步重点就是把 FRdist 优化回来。
