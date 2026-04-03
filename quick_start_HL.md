# RoboManipBaselines Tutorial for Humanoid Lab Orientation 2026

## 0: git clone および、仮想環境のセットアップ
インストールには時間がかかるため、セットアップの待ち時間と並行して、tutoralを進めていきましょう。


1.
```console
$ cd <任意のディレクトリ>
$ git clone https://github.com/isri-aist/RoboManipBaselines.git --recursive
```

2.
```console
全員へのrequirement_tutorial.txtの配布(githubのpublicなレポジトリからダウンロードしてもらうようにしよう)。配布ではなく、リンクを貼ることにする。
```

3.
並行して、ubuntu 22.04 なら、
```conssole
$ sudo apt update
$ sudo apt install python3.10-venv
````

4.
１つ目のターミナルに戻って、
```console
$ cd RoboManipBaselines
$ python3.10 -m venv .venv
$ pip install -r requirements_tutorial.txt
$ source .venv/bin/activate
$ python --version
```

5.
時間がかかる、
```console

$ pip install -e .[act]
```

Install ACT from a third party:
```console
$ cd third_party/act/detr
$ pip install -e .
```



## 1: RoboManipBaselinesとは？
JRLにも籍をおいている産総研の研究者,室岡 雅樹氏が中心となり開発した、マニピュレーション用プラットフォーム。


できること（シミュレーション、実世界の両方で）
- 遠隔操作データの収集
- 模倣学習ポリシーのトレーニング
- 模倣学習ポリシーのロールアウト
- データの可視化
- 学習アルゴリズム×タスクの迅速な比較・検証

https://isri-aist.github.io/RoboManipBaselines-ProjectPage/

![alt text](quick_start_HL_attachments/quick_start_HL-concept.png)

他のマニピュレーション学習プラットフォームとの比較

![alt text](quick_start_HL_attachments/quick_start_HL.png)




## 2: 今回、流れを確認するタスク

##### MujocoUR5eCable

最初の状態
![alt text](quick_start_HL_attachments/quick_start_HL-5.png)

最終的な成功の状態
![alt text](quick_start_HL_attachments/quick_start_HL-6.png)

ただし、公式のタスクを必ずこなす必要があるわけではありません。今回は、keyboardの遠隔操作であり、複雑な操作は難しいため、単にケーブルを持ち上げるというのシンプルなタスクを学習させてみましょう。


## 3: Data collection by teleoperation
> [!TIP]
> Instead of collecting data by teleoperation, you can download the public dataset `TeleopMujocoUR5eCable_Dataset30` from [here](https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/dataset_list.md#Demonstrations-in-MuJoCo-environments).

Operate the robot in the simulation and save the data:
```console
#Go to the top directory of this repository
$ cd robo_manip_baselines
$ # Connect a SpaceMouse to your PC
$ python ./bin/Teleop.py MujocoUR5eCable --world_idx_list 0 5 --input_device keyboard
```
if you want to use spacemouse change "keyboard" to "spacemouse"

> [!TIP]
> A teleoperation input device such as a 3D mouse can be used instead of a keyboard. See [here](../robo_manip_baselines/teleop/README.md).

In our experience, models can be trained stably with roughly 30 data sets.
The teleoperation data is saved in the `robo_manip_baselines/dataset/MujocoUR5eCable_<date_suffix>` directory (e.g., `MujocoUR5eCable_20240101_120000`).


## 4: Model training
Train the ACT:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py Act --dataset_dir ./dataset/MujocoUR5eCable_20240101_120000 --num_epochs 100
```

--num_epochsのデフォルトは1000ですが、今回はチュートリアルのためかなり短くしています。

The learned parameters are saved in the `robo_manip_baselines/checkpoint/Act/<dataset_name>_Act_<date_suffix>` directory (e.g., `MujocoUR5eCable_20240101_120000_Act_20240101_130000`).

> [!NOTE]
> The following error will occur if the chunk_size is larger than the time series length of the training data.
> In such a case, either set the `--skip` option to a small value, or set the `--chunk_size` option to a small value.
> ```console
> RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
> ```


学習済みのパラメータは、ここにあります。本格的に学習させた重みを手軽に使いたい場合はこちらからダウンロードしましょう。
https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/learned_parameters.md


## 5: Policy rollout
Rollout the ACT in the simulation:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Act MujocoUR5eCable \
--checkpoint ./checkpoint/Act/MujocoUR5eCable_20240101_120000_Act_20240101_130000/policy_last.ckpt \
--world_idx 0
```

## 6: その他、紹介しきれなかった模倣学習アルゴリズムやロボット
https://isri-aist.github.io/RoboManipBaselines-ProjectPage/

環境
- MujocoAlohaHandover
- MujocoHsrTidyup
- RealUR5eDualDemo
- RealXarm7Demo

模倣学習アルゴリズム
- MLP
- SARNN
- ACT
- Diffusion Policy
- 3D Diffusoin Policy
- Flow Policy
- ManiFlow Policy(Image)
- ManiFlow Policy(Pointcloud)
- TACT
- pi0
- GR00T



## 7: （時間が余れば）実機xArmでの２台の３Dマウスによる遠隔操作体験（２，３人程度の予定）​

#### RealXarm7DualDemo
という環境では、２台の3Dマウスによって、dual armのxArm を操作することが可能。

参考までにコマンドのみ、書いておく。あなたの環境でちゃんと実行するには、以下のコマンドの前に、xarmのセットアップや、複数のspacemouseを使えるようにするセットアップが必要。
https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/use_multiple_spacemouse.md


```console
$ python ./robo_manip_baselines/bin/Teleop.py RealXarm7DualDemo --config ./robo_manip_baselines/envs/configs/RealXarm7DualDemoEnv.yaml --input_device_config ./robo_manip_baselines/teleop/configs/SpaceMouseDual.yaml
```
