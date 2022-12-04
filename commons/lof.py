import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# api参考：https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
# 算法解释参考：https://zhuanlan.zhihu.com/p/37753692

"""
返回追加了k 邻域距离和LOF 离群因子的df
"""
def localoutlierfactor(data, predict, k):
    clf = LocalOutlierFactor(n_neighbors=k, algorithm='auto', contamination=0.1, n_jobs=-1, novelty=True)
    clf.fit(data)
    # 记录 k 邻域距离
    predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
    # 记录 LOF 离群因子，做相反数处理
    predict['local outlier factor'] = -clf.decision_function(predict.iloc[:, :-1])
    return predict

"""
绘制
"""
def plot_lof(result, method):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(8, 4)).add_subplot(111)
    plt.scatter(result[result['local outlier factor'] > method].index,
                result[result['local outlier factor'] > method]['local outlier factor'], c='red', s=50,
                marker='.', alpha=None,
                label='离群点')
    plt.scatter(result[result['local outlier factor'] <= method].index,
                result[result['local outlier factor'] <= method]['local outlier factor'], c='black', s=50,
                marker='.', alpha=None, label='正常点')
    plt.hlines(method, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF局部离群点检测', fontsize=13)
    plt.ylabel('局部离群因子', fontsize=15)
    plt.legend()
    plt.show()

"""
lof接口函数
"""
def lof(data, predict=None, k=5, method=1, plot=False):
    import pandas as pd
    # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    predict = pd.DataFrame(predict)
    # 计算 LOF 离群因子
    predict = localoutlierfactor(data, predict, k)
    if plot == True:
        plot_lof(predict, method)
    # 根据阈值划分离群点与正常点
    outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <= method].sort_values(by='local outlier factor')
    return outliers, inliers


"""
将 LOF 异常值分数归一化到 [0, 1] 区间，运用统计方法进行划分下面提供使用箱型图进行界定的方法，根据异常输出情况参考选取。
"""
def box(data, legend=True):
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.style.use("ggplot")
    plt.figure()
    # 如果不是DataFrame格式，先进行转化
    if type(data) != pd.core.frame.DataFrame:
        data = pd.DataFrame(data)
    p = data.boxplot(return_type='dict')
    warming = pd.DataFrame()
    y = p['fliers'][0].get_ydata()
    y.sort()
    for i in range(len(y)):
        if legend == True:
            plt.text(1, y[i] - 1, y[i], fontsize=10, color='black', ha='right')
        if y[i] < data.mean()[0]:
            form = '低'
        else:
            form = '高'
        warming = warming.append(pd.Series([y[i], '偏' + form]).T, ignore_index=True)
    print(warming)
    plt.show()

def predict_graph_if_maclious():
    pass



