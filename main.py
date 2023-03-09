import numpy as np
import matplotlib.pyplot as plt

from HH_package.hodgkinhuxley import HodgkinHuxley
from Pf_package.particle_filter import ParticleFilter

"""
Global variables
"""
steps = 4000 #step数
sigma = 0.1

def generate_data(gL_abnormal:list):
    """
    テストデータの作成
    :returns V_n: gL一定 V_ab:g_L変化あり
    """
    HH_n = HodgkinHuxley()  # ノイズなしの正常用データ
    HH_ab = HodgkinHuxley()  # g_Lを変化させるデータ
    # 外部からの入力電流
    I_inj = np.zeros(steps)
    I_inj[:] = 20
    # 観測膜電位をそれぞれ作成
    V_n = np.zeros(steps)
    V_ab = np.zeros(steps)
    # それぞれに初期値設定
    V_n[0] = -65.0
    V_ab[0] = -65.0

    # HH.stepを実行し, データを生成
    for i in range(steps - 1):
        result_n = HH_n.step(I_inj[i])
        result_ab = HH_ab.step(I_inj[i], gL_abnormal[i])
        V_n[i+1] = result_n[3]
        V_ab[i+1] = result_ab[3]

    # 観測ノイズを付加
    noise_ab = np.random.normal(0, sigma, (steps - 1,))
    noise_ab = np.insert(noise_ab, 0, 0)
    V_n = V_n + noise_ab
    V_ab = V_ab + noise_ab
    return V_n, V_ab

def show_graph_observation_data(V_n, V_ab, gL_abnormal):
    """
    観測データの可視化
    :param V_n: 通常時
    :param V_ab: パラメータが途中で変化するデータ
    """
    t = np.arange(0, steps)
    plt.figure(figsize=(8, 5))
    plt.plot(t, V_n, label='V(t):normal')
    plt.plot(t, V_ab, c='orange', label='V(t):abnormal')
    plt.xlabel('t[ms]', fontsize=15)
    plt.ylabel('V[mV]', fontsize=15)
    plt.legend(fontsize=15, loc='lower left')
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(t, gL_abnormal, c='orange', label='gL')
    plt.ylim(0, 0.8)
    plt.xlabel('t[ms]', fontsize=15)
    plt.ylabel('gL', fontsize=15)
    plt.legend()
    plt.grid()
    plt.show()

def calculate_mse(V_ab, V_particle, V_particle_ab):
    """
    予測した粒子と観測値の二乗誤差を計算.
    誤差が小さい方を選択.
    """
    mse = np.zeros((steps))
    mse_ab = np.zeros((steps))
    for i in range(steps):
        mse[i] = (V_particle[i] - V_ab[i]) ** 2
        mse_ab[i] = (V_particle_ab[i] - V_ab[i]) ** 2
    return mse, mse_ab

def select_model(size, mse, mse_ab):
    """
    モデル選択を行う時間幅を指定し, モデル選択を行う.
    """
    models = []
    for i in range(steps // size):
        if np.sum(mse[i * size: i * size + size]) <= np.sum(mse_ab[i * size: i * size + size]):
            model = 0.3
        else:
            model = 0.5
        for _ in range(size):
            models.append(model)
    return models

def performance(gL_abnormal, models):
    """
    性能評価
    """
    count = 0
    for i in range(len(gL_abnormal)):
        if gL_abnormal[i] == models[i]:
            count += 1
    percent = (count / steps) * 100
    return percent

def show_graph_mse(percent, gL_abnormal, models):
    t = np.arange(0, steps)
    """ データを可視化 """
    plt.title(f"performance : {percent} [%]")
    plt.plot(t, gL_abnormal, label='gL:abnormal')
    plt.plot(t, models, label='gL:predict')
    plt.ylim(0, 0.8)
    plt.xlabel('t[ms]')
    plt.ylabel('gL')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    """ モデル選択を用いた変化点検出 """
    """ gL変化のndArrayを作成 """
    gL_abnormal = np.zeros(steps)
    gL_abnormal[:] = 0.3
    gL_abnormal[(steps) // 2:] = 0.5
    # 観測データを生成
    data = generate_data(gL_abnormal)
    V_n = data[0]
    V_ab = data[1]
    show_graph_observation_data(V_n, V_ab, gL_abnormal)
    # gL=0.3, 0.5で二つの場合で一期先予測を実行
    n = 100
    pf = ParticleFilter(V_ab, n_particle=n)
    pf_ab = ParticleFilter(V_ab,n_particle=n)
    pf.simulate(gL_particle=0.3)
    pf_ab.simulate(gL_particle=0.5)
    V_particle = pf.V_average
    V_particle_ab = pf_ab.V_average

    # 誤差を計算
    mse_result = calculate_mse(V_ab, V_particle, V_particle_ab)
    # モデル選択を行う時間幅
    size = 100
    # モデル選択
    models = select_model(size, mse_result[0], mse_result[1])
    # 性能評価
    percent = performance(gL_abnormal, models)
    show_graph_mse(percent, gL_abnormal, models)


if __name__ == '__main__':
    main()