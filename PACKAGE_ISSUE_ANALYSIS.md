# OmniAvatar パッケージ化後のimportエラー分析レポート

## 概要

OmniAvatarをPyPIパッケージとしてインストール後、`scripts/inference.py`実行時に発生したエラーの詳細分析と解決方法をまとめる。

## 発生したエラー

### 主要エラー
```
TypeError: argument of type 'NoneType' is not iterable
    if 'use_audio' in args:
       ^^^^^^^^^^^^^^^^^^^
```

### エラー発生箇所
- `OmniAvatar/models/wan_video_dit.py:307`
- `OmniAvatar/models/wan_video_dit.py:567` (`from_civitai`メソッド内)

## 根本原因の分析

### 1. グローバル変数初期化タイミングの変化

**パッケージ化前（sys.path.append使用時）:**
```
scripts/inference.py:14: args = parse_args()  # argsがグローバル設定される
↓
wan_video_dit.py: from ..utils.args_config import args  # 既に初期化済みのargsを取得
```

**パッケージ化後:**
```
wan_video_dit.py: from ..utils.args_config import args  # args = None（未初期化）
↓
scripts/inference.py:14: args = parse_args()  # ローカル変数として設定（グローバルに影響なし）
```

### 2. モジュールレベルでのグローバル変数依存

`args_config.py`でグローバル変数として定義：
```python
args = None  # 初期値はNone

def parse_args():
    global args
    # ... 処理 ...
    args = parser.parse_args()  # グローバル変数に設定
    return args
```

### 3. パッケージインポート順序の影響

パッケージ化により、モジュールのインポート順序が変わり、`wan_video_dit.py`が`args`を参照する時点でまだ`None`の状態だった。

## 解決方法

### 1. WanModelクラスの修正

**修正前:**
```python
class WanModel(torch.nn.Module):
    def __init__(self, dim: int, ...):
        # ...
        if 'use_audio' in args:  # argsがNoneの場合エラー
            self.use_audio = args.use_audio
```

**修正後:**
```python
class WanModel(torch.nn.Module):
    def __init__(self, dim: int, ..., args=None):  # argsを引数として受け取る
        # ...
        if args and hasattr(args, 'use_audio'):  # 安全なチェック
            self.use_audio = args.use_audio
        else:
            self.use_audio = False
```

### 2. ModelManagerクラスの修正

**修正前:**
```python
class ModelManager:
    def __init__(self, torch_dtype=torch.float16, ...):
        # argsを保持していない
```

**修正後:**
```python
class ModelManager:
    def __init__(self, torch_dtype=torch.float16, ..., args=None):
        self.args = args  # argsを保存
```

### 3. 関数呼び出しの修正

**修正前:**
```python
model = model_class(**extra_kwargs)  # argsが渡されない
```

**修正後:**
```python
if model_class.__name__ == 'WanModel' and args is not None:
    extra_kwargs['args'] = args
model = model_class(**extra_kwargs)
```

### 4. 安全なグローバル変数取得関数の追加

```python
def get_global_args():
    """Get global args safely"""
    try:
        from ..utils.args_config import args
        return args
    except ImportError:
        return None
```

## 修正箇所詳細

### ファイル1: `OmniAvatar/models/wan_video_dit.py`

**変更箇所:**
- Line 9: グローバルargs importを削除、`get_global_args()`関数を追加
- Line 274: `WanModel.__init__()`に`args=None`パラメータを追加
- Line 308-311: argsの安全なチェックに変更
- Line 573-577: `from_civitai`メソッドでの安全なargs取得

### ファイル2: `OmniAvatar/models/model_manager.py`

**変更箇所:**
- Line 8: `load_model_from_single_file`関数に`args=None`パラメータを追加
- Line 25-26: WanModelの場合にargsを`extra_kwargs`に追加
- Line 290: `ModelManager.__init__()`に`args=None`パラメータを追加
- Line 298: `self.args = args`を追加
- Line 149, 157: `kwargs.get('args')`でargsを取得
- Line 358: model_detector.loadにargsを渡す

### ファイル3: `scripts/inference.py`

**変更箇所:**
- Line 125: `ModelManager`のインスタンス作成時に`args=args`を追加

## 検証結果

修正後、以下のコマンドが正常に動作することを確認：
```bash
uv run torchrun --standalone --nproc_per_node=1 scripts/inference.py \
  --config configs/inference_1.3B.yaml \
  --exp_path pretrained_models/OmniAvatar-1.3B \
  --input_file examples/infer_samples.txt
```

## 今後の懸念事項と予防策

### 1. 他のグローバル変数依存コード

**懸念:** 他のファイルでも同様のグローバル変数依存があるかもしれない

**予防策:**
- `args_config.py`のargsを参照している全ファイルの監査
- グローバル変数の使用を最小限に抑制
- 依存性注入パターンの採用

**要確認ファイル:**
```bash
# 調査コマンド
grep -r "from.*args_config import args" OmniAvatar/
```

### 2. 循環インポートの可能性

**懸念:** `get_global_args()`関数で動的インポートを使用しているため、循環インポートが発生する可能性

**予防策:**
- インポート構造の見直し
- 設定管理の中央集権化
- 環境変数やファイルベースの設定管理への移行検討

### 3. パッケージ配布時の互換性

**懸念:** 開発環境とパッケージ配布環境での動作差異

**予防策:**
- CI/CDでのパッケージインストールテスト追加
- 両環境での動作確認の自動化
- Docker環境での一貫性確保

### 4. args設定の遅延初期化

**懸念:** argsが必要な時点で初期化されていない場合がある

**予防策:**
- Lazy初期化パターンの実装
- argsの必須性の明確化
- デフォルト値の適切な設定

## 推奨改善案

### 1. 設定管理の改善

```python
# 推奨パターン
class Config:
    def __init__(self):
        self.use_audio = False
        self.model_config = None
    
    @classmethod
    def from_args(cls, args):
        config = cls()
        if args and hasattr(args, 'use_audio'):
            config.use_audio = args.use_audio
        return config
```

### 2. 依存性注入の採用

```python
# 推奨パターン
class WanModel(torch.nn.Module):
    def __init__(self, config: Config, ...):
        self.use_audio = config.use_audio
```

### 3. ファクトリーパターンの導入

```python
# 推奨パターン
class ModelFactory:
    @staticmethod
    def create_wan_model(args, **kwargs):
        config = Config.from_args(args)
        return WanModel(config=config, **kwargs)
```

## まとめ

今回の問題は、パッケージ化による**モジュールインポート順序の変化**が原因で発生した。グローバル変数への依存を減らし、**明示的な引数渡し**に変更することで解決した。

今後は、グローバル変数の使用を最小限に抑え、**設定管理の中央集権化**と**依存性注入パターン**の採用を検討することを推奨する。