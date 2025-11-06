# RevDEQ-PyTorch

PyTorch implementation of Reversible Deep Equilibrium Models (RevDEQ).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Note**: This is a PyTorch reimplementation inspired by the original JAX/Equinox implementation. See [Credits](#クレジットと謝辞) for details.

## 概要

Reversible Deep Equilibrium Models (RevDEQ) は、学習された関数の不動点としてモデル出力を定義する深層平衡モデル（DEQ）の一種です。RevDEQは、正確な勾配計算を可能にし、正則化を必要とせず、DEQよりも少ない関数評価で済むことが特徴です。

## 主な特徴

- **省メモリ**: 可逆的な勾配計算により、メモリ効率的な学習が可能
- **正確な勾配**: 固定点反復の可逆的な勾配計算により、正確な勾配を計算
- **少ない関数評価**: DEQよりも少ない関数評価で収束
- **PyTorch実装**: 純粋なPyTorch実装で、transformersライブラリと統合

## 環境要件

- Python 3.10+
- PyTorch 2.0+
- CUDA (GPU推奨、CPUでも動作可能)

## セットアップ

### Dockerを使用する場合（推奨）

```bash
# コンテナのビルドと起動
docker-compose up -d

# コンテナに入る
docker-compose exec revdeq bash

# 学習を実行
python train.py --config configs/default.yaml
```

### ローカル環境の場合

```bash
# uvを使用する場合（推奨）
uv sync

# pipを使用する場合
pip install -r requirements.txt

# パッケージをインストール（開発モード）
pip install -e .
```

## 使用方法

### 学習

```bash
# デフォルト設定で学習
python train.py --config configs/default.yaml

# カスタム設定で学習
python train.py --config configs/default.yaml --dataset wikitext --dataset_config wikitext-2-raw-v1

# チェックポイントから再開
python train.py --config configs/default.yaml --resume_from_checkpoint checkpoints/checkpoint-1000
```

### 推論

```bash
# モデルを使用してテキスト生成
python inference.py --model_path checkpoints/model.pt --text "The quick brown fox"

# カスタムパラメータで生成
python inference.py \
    --model_path checkpoints/model.pt \
    --text "Once upon a time" \
    --max_length 100 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9
```

## Google Colab

このリポジトリはGoogle Colabでも動作します。`notebooks/revdeq_colab.ipynb`をColabで開いて実行してください。

Colabでの使用手順:
1. Google Colabで新しいノートブックを作成
2. `notebooks/revdeq_colab.ipynb`の内容をコピー
3. セルを順番に実行

## 設定ファイル

`configs/default.yaml`でモデルと学習の設定を変更できます。主な設定項目:

- `hidden_size`: 隠れ層のサイズ
- `num_layers`: レイヤー数（固定点反復の深さ）
- `num_fixed_point_iterations`: 固定点反復の最大回数
- `fixed_point_tol`: 固定点収束の許容誤差
- `learning_rate`: 学習率
- `batch_size`: バッチサイズ

## プロジェクト構造

```
RevDEQ/
├── revdeq/              # メインモジュール
│   ├── __init__.py
│   ├── model.py        # RevDEQモデル実装
│   └── solver.py       # 固定点ソルバー
├── notebooks/           # Jupyterノートブック
│   └── revdeq_colab.ipynb
├── configs/             # 設定ファイル
│   └── default.yaml
├── train.py            # 学習スクリプト
├── inference.py        # 推論スクリプト
├── Dockerfile          # Dockerイメージ定義
├── docker-compose.yml  # Docker Compose設定
├── requirements.txt    # Python依存関係
├── pyproject.toml      # プロジェクト設定（uv用）
└── README.md           # このファイル
```

## 学習の進捗確認

学習中はTensorBoardで進捗を確認できます:

```bash
# TensorBoardを起動
tensorboard --logdir checkpoints/logs
```

ブラウザで `http://localhost:6006` にアクセスして、lossの推移を確認できます。lossが25程度まで下がることを目標にしてください。

## トラブルシューティング

### メモリ不足エラー

- `batch_size`を小さくする
- `gradient_accumulation_steps`を増やす
- `num_fixed_point_iterations`を減らす

### 学習が遅い

- GPUを使用する（`CUDA_VISIBLE_DEVICES=0 python train.py ...`）
- `fp16`を有効にする（`configs/default.yaml`で設定）
- `dataloader_num_workers`を調整する

## 参考資料

- [論文](https://arxiv.org/abs/2509.12917): "Reversible Deep Equilibrium Models"
- [Notion資料](https://www.notion.so/Reversible-Deep-Equilibrium-Models-29c9d9388fbd80f8bc8cf53c26ff1aed)

## クレジットと謝辞

この実装は以下のリソースを参考にしています：

- **論文**: "Reversible Deep Equilibrium Models" ([arXiv:2509.12917](https://arxiv.org/abs/2509.12917))
- **元の実装**: [sammccallum/reversible-deq](https://github.com/sammccallum/reversible-deq) (JAX/Equinox実装、Apache-2.0 License)
  - この実装は元のJAX/Equinox実装のアルゴリズムを参考にしていますが、PyTorchで独自に実装したものです

本実装は論文のアルゴリズムに基づいてPyTorchで実装されていますが、元のJAX/Equinox実装の設計思想を参考にしています。

## ライセンス

MIT License

Copyright (c) 2024 RevDEQ-PyTorch Contributors

本実装は以下のリソースを参考にしています：
- 論文: "Reversible Deep Equilibrium Models" ([arXiv:2509.12917](https://arxiv.org/abs/2509.12917))
- 元の実装: [sammccallum/reversible-deq](https://github.com/sammccallum/reversible-deq) (Apache-2.0 License)

## 貢献

プルリクエストやイシューを歓迎します！

