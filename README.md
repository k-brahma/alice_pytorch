# きのこの山たけのこの里判定プログラム

画像を渡すと、その画像がきのこの山のものかかたけのこの里のものかを判定してくれるプログラムです。

## 使い方

1. pip install -r requirements.txt

```bash
pip install -r requirements.txt
```

2. モデルの学習と保存、判定

```bash
python parsed_overfit_avoidance_skip_create_model_if_model_exists.py
```

`parsed_overfit_avoidance_skip_create_model_if_model_exists.py` じゃなくても良いのですが、これがいちばん出力されるログがおもしろいので。

上記により、 `model.pth` が生成されます。  
さらに、 `model.pth` を参考にして、 data/test にある画像を判定します。  
判定結果は `results/submission_avoidance.csv` に保存されます。判定結果は 0 か 1 として記載されます。

### 出力例:

```bash
(venv) PS C:\project_dir> python parsed_overfit_avoidance_skip_create_model_if_model_exists.py             
Epoch 1/15, Loss: 0.6843, Train Accuracy: 0.5500, Val Accuracy: 0.8000
Epoch 2/15, Loss: 0.2917, Train Accuracy: 0.9500, Val Accuracy: 0.7000
Epoch 3/15, Loss: 0.1601, Train Accuracy: 1.0000, Val Accuracy: 0.8000
Epoch 4/15, Loss: 0.1122, Train Accuracy: 1.0000, Val Accuracy: 0.9000
Epoch 5/15, Loss: 0.0434, Train Accuracy: 1.0000, Val Accuracy: 1.0000
Epoch 6/15, Loss: 0.0297, Train Accuracy: 1.0000, Val Accuracy: 1.0000
Epoch 7/15, Loss: 0.0327, Train Accuracy: 1.0000, Val Accuracy: 1.0000
Epoch 8/15, Loss: 0.0380, Train Accuracy: 1.0000, Val Accuracy: 1.0000
Early stopping triggered after epoch 8
Best validation accuracy: 1.0000
Model saved to models/model_avoidance.pth
ファイル名: 1.JPG
きのこ度: 38.17%, たけのこ度: 83.18%

ファイル名: 2.JPG
きのこ度: 7.43%, たけのこ度: 93.30%

ファイル名: 3.JPG
きのこ度: 33.07%, たけのこ度: 81.41%

ファイル名: 4.JPG
きのこ度: 67.76%, たけのこ度: 61.30%

ファイル名: 5.JPG
きのこ度: 96.50%, たけのこ度: 7.83%

ファイル名: 6.JPG
きのこ度: 94.74%, たけのこ度: 15.14%

ファイル名: 7.JPG
きのこ度: 98.06%, たけのこ度: 7.22%

ファイル名: 8.JPG
きのこ度: 80.11%, たけのこ度: 18.24%
```

上記の出力例では、Epoch ... というのが 1/15 から 8/15 まであります。これは、元データを使って15回くりかえし学習しようとしています。  
なのですが、Epoch 8/15 で Early stopping triggered after epoch 8 というのが出ています。これは、過学習を防ぐために、学習を打ち切ったということです。  
「過学習」については詳しく説明しませんが...ざっくりひとことで言うと「視野狭窄になってのめりこんで学習しすぎると、かえって性能が落ちることがある」といったところでしょうか。

この学習での学習結果は、 `models/model_avoidance.pth` として保存されます。  
最後に、 `models/model_avoidance.pth` を使って `data/test` にある画像をひとつひとつ判定していきます。その結果を、
`submission.csv` に保存します。

なお、すでに `models/model_avoidance.pth` がある場合は、学習をスキップして、そのモデルを使って判定を行います。  
実際に動作させると分かりますが、AIモデル生成のための学習は、かなり時間がかかる & PCに負荷をかけるので、学習をスキップすることができるのは、かなりありがたいです。

## デモ用メモ:

各ファイルについて。  
ただし、以下は、おぼろげな記憶を元に書いたものです。

| スクリプト                       | 詳細                                               |
|-----------------------------|--------------------------------------------------|
| parsed.py                   | main.py を整理したもの。過学習への配慮がないので何が何でも学習を 15 ラウンド行います |
| parsed_overfit_avoidance.py | 過学習に配慮したもの                                       |

あとは、 parsed_overfit_avoidance*.py は、 parsed_overfit_avoidance.py の派生形です。  
すでに説明したとおりモデルがすでにある場合は処理をスキップする、あるいは、別のアルゴリズムでモデルを生成する、等しています。

なお `main.py` は当初お預かりしたファイル。    
`complete_local_script.py` は、 `main.py` をとりあえず関数単位で整理したものです。
