# Golf Form Coach — APIなし版

スマホで撮ったゴルフスイング動画をアップロードし、MediaPipeの2D骨格推定とローカルのルールベース診断で、初心者向けに「次の10球で直すポイント」を最大2つ返すMVPです。

## 特徴
- OpenAI API不要
- AI利用料なし
- iPhone / Androidの動画アップロード
- MOV/HEVCを互換MP4へ変換
- 正面 / 後方の2方向に対応
- アドレス / トップ / インパクト / フィニッシュを指定して解析
- 改善点は最大2つに絞る

## 起動
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 注意
2D動画ベースの初心者向けMVPです。クラブフェース角、クラブ軌道、手首角度、3D回転量は測定しません。

Deployment target: Railway / mobile MVP
