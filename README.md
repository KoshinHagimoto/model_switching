### パラメータ変化を検出

パラメータの変化をモデル選択を用いた手法を用いて行う.

ここでは変化するパラメータはHHモデルにおける, コンダクタンス(gL)とし, gLが真の値0.3から間の2000[ms]で0.5に変化する場合のシミュレーションデータを作成し検証を行った.

モデル選択を用いた手法(main.py)：粒子フィルタにおいて, 複数のモデルを用いて一期先予測を行い, 観測データとの誤差計算をし, 誤差が小さいモデルを選択することで, パラメータ変化を検出する.