# QuantPortal 实战教程 — 从因子信号到优化投资组合

> 一步步带你理解多因子量化投资的完整流程。每个概念都有代码对应，每个公式都有直觉解释。

---

## 目录

- [Layer 0: 项目全景](#layer-0-项目全景)
- [Layer 1: 数据层 — Universe 管理](#layer-1-数据层--universe-管理)
- [Layer 2: 动量因子 — 追涨杀跌的科学](#layer-2-动量因子--追涨杀跌的科学)
- [Layer 3: 波动率因子 — 低波动异象](#layer-3-波动率因子--低波动异象)
- [Layer 4: 质量因子 — 用价格衡量"好公司"](#layer-4-质量因子--用价格衡量好公司)
- [Layer 5: 配对扫描器 — 自动发现协整对](#layer-5-配对扫描器--自动发现协整对)
- [Layer 6: 信号组合 — 从等权到机器学习](#layer-6-信号组合--从等权到机器学习)
- [Layer 7: 投资组合优化 — 该买多少](#layer-7-投资组合优化--该买多少)
- [Layer 8: 回测引擎 — 时间旅行测试](#layer-8-回测引擎--时间旅行测试)
- [Layer 9: 工程实践 — 测试与可视化](#layer-9-工程实践--测试与可视化)
- [与 MarketDNA 的关系](#与-marketdna-的关系)
- [下一步学习](#下一步学习)

---

## Layer 0: 项目全景

### 投资组合管理的三个问题

1. **买什么？** — 因子信号告诉你（动量、低波动、质量）
2. **买多少？** — 优化器告诉你（Max Sharpe、Min Variance、Risk Parity）
3. **什么时候调仓？** — 回测引擎告诉你（月度再平衡、换手率控制）

### 从 MarketDNA 到 QuantPortal

MarketDNA 是**诊断工具**——告诉你一个资产在统计上长什么样。
QuantPortal 是**决策引擎**——告诉你应该买什么、买多少。

```
MarketDNA                          QuantPortal
──────────────                     ──────────────────────
分布指纹 (厚尾检测)            →   质量因子 (尾部风险)
GARCH 波动率建模              →   波动率因子 (低波动异象)
HMM Regime 检测               →   ML 条件信号权重
协整检验                      →   配对扫描器 (自动发现)
Walk-forward 验证             →   回测引擎
统计报告                      →   投资组合优化权重
```

---

## Layer 1: 数据层 — Universe 管理

### 核心概念

**Universe（股票池）** = 你考虑投资的所有股票的集合。

代码位置: `quantportal/data/universe.py`

```python
from quantportal.data.universe import fetch_universe, SP500_TECH

# 获取10只科技股的对齐数据
universe = fetch_universe(SP500_TECH[:10], start="2018-01-01")

# universe.prices:  DataFrame (date × ticker) — 调整后收盘价
# universe.returns: DataFrame (date × ticker) — 对数收益率
```

### 关键设计决策

**为什么用对数收益率而不是百分比收益率？**

对数收益率的优势：
1. **时间可加性**: `log_ret(A→C) = log_ret(A→B) + log_ret(B→C)`
2. **对称性**: +10% 和 -10% 的对数收益率绝对值相等
3. **正态近似更好**: 对数收益率更接近正态分布

```python
# 对数收益率计算
returns = np.log(prices / prices.shift(1))
```

### 数据对齐

多只股票可能在不同日期上市或停牌。Universe 管理器自动对齐所有股票到共同交易日。

---

## Layer 2: 动量因子 — 追涨杀跌的科学

### 直觉

> 过去涨得好的股票，未来倾向于继续涨（至少短期内）。

这是金融学中最稳健的异象之一，由 Jegadeesh & Titman (1993) 记录。

代码位置: `quantportal/factors/momentum.py`

### 两种动量

**截面动量 (Cross-Sectional)**: 在所有股票中排名

```
今天的CS动量 = 过去12个月收益率的排名百分位
```

例如：如果AAPL在10只股票中排名第2，CS动量 = 0.8（前20%）

**时间序列动量 (Time-Series)**: 和自己比

```
TS动量 = sign(过去12个月收益率)
       = +1 如果过去12个月上涨
       = -1 如果过去12个月下跌
```

### 为什么跳过最近1个月？

短期反转效应：过去一周涨得多的股票，下周倾向于回调。

所以经典动量 = 过去12个月收益 - 过去1个月收益

```python
# 代码实现
cum_ret = returns.rolling(252).sum()    # 12个月累积收益
skip_ret = returns.rolling(21).sum()     # 1个月累积收益
raw_momentum = cum_ret - skip_ret        # 跳过最近1个月
```

### 排名基Z-Score

为什么不直接用收益率做排名？因为异常值：一只股票涨了300%会扭曲整个截面。

解决方案：先排名（百分位），再转成z-score

```python
ranked = row.rank(pct=True)           # [0.0, 1.0] 均匀分布
zscore = norm.ppf(ranked.clip(0.01, 0.99))  # 标准正态分位数
```

这样即使有极端值，z-score 也被限制在合理范围内。

---

## Layer 3: 波动率因子 — 低波动异象

### 直觉

> 低波动的股票反而收益更高（风险调整后）。

这违反了CAPM（高风险=高回报），但在实证中非常稳健。

代码位置: `quantportal/factors/volatility.py`

### 为什么低波动有效？

1. **彩票偏好**: 散户喜欢高波动的"彩票股"（梦想暴富），推高了这些股票的价格，降低了未来收益
2. **基准束缚**: 基金经理被考核相对于基准的跟踪误差，不愿意大量买入低波动股票
3. **杠杆厌恶**: 如果投资者不愿意加杠杆，他们会overpay高beta资产来追求高收益

### 四个信号

```python
# 1. 已实现波动率 (年化)
realized_vol = returns.rolling(21).std() * sqrt(252)

# 2. 波动率的波动率 (vol-of-vol)
vol_of_vol = realized_vol.rolling(63).std()

# 3. 低波动分数 (截面排名, 越低越好 → 分数越高)
low_vol_score = realized_vol.rank(axis=1, pct=True, ascending=False)

# 4. 波动率趋势 (正=膨胀, 负=收缩)
vol_trend = realized_vol - realized_vol.rolling(63).mean()
```

---

## Layer 4: 质量因子 — 用价格衡量"好公司"

### 直觉

> 没有P/E比率和资产负债表数据，能不能衡量"质量"？可以！用价格行为。

代码位置: `quantportal/factors/quality.py`

### 价格基质量代理

1. **滚动Sharpe**: 收益/风险比 → 一致赚钱的股票
2. **回撤恢复力**: 最大回撤越小 → 越稳定
3. **尾部风险**: 超额峰度越低 → 极端事件越少
4. **综合得分**: 三者的截面z-score平均

```python
# 综合质量分数 = (Sharpe_z + DD_z - Kurtosis_z) / 3
stability_score = (cs_zscore(sharpe) + cs_zscore(dd_score) - cs_zscore(tail_risk)) / 3
```

### 为什么减去峰度？

高峰度 = 肥尾 = 极端事件多 = 坏事（对大多数投资者）。
所以质量 = 高Sharpe + 低回撤 - 高峰度。

---

## Layer 5: 配对扫描器 — 自动发现协整对

### 问题

大多数教程手动选择配对（GLD/GDX, KO/PEP）。但真正的量化方法是系统性扫描。

代码位置: `quantportal/scanner/pair_scanner.py`

### 扫描流程

```
1. 对 Universe 中所有 C(N,2) 个配对进行协整检验
2. 按 p 值过滤（< 0.05）
3. 计算 spread 的 ADF 检验（确认平稳性）
4. 估计半衰期（越短越适合交易）
5. 综合评分 = -log10(p) + 10/HL - log10(ADF_p)
6. 应用 Bonferroni 校正（多重检验）
```

### 多重检验问题

10只股票有 C(10,2) = 45 个配对。在 5% 显著性下，预计有 ~2.25 个假阳性！

**Bonferroni 校正**: 把显著性阈值从 0.05 改为 0.05/45 ≈ 0.001

```python
threshold = max_pval / n_pairs if apply_bonferroni else max_pval
```

---

## Layer 6: 信号组合 — 从等权到机器学习

### 等权组合（基线）

最简单的方法：把所有信号的截面z-score平均

```python
combined = (momentum_z + low_vol_z + quality_z) / 3
```

代码位置: `quantportal/ml/signal_combiner.py` → `combine_signals_equal()`

### LightGBM 组合

等权假设所有信号永远一样重要。但现实中：
- 趋势市场：动量更有效
- 高波动市场：低波动和质量更有效

**LightGBM 决策树天然能处理这种条件性**:

```
如果 vol_trend > 0 (波动率膨胀):
    → 更重视 quality 和 low_vol
否则:
    → 更重视 momentum
```

代码位置: `quantportal/ml/signal_combiner.py` → `combine_signals_ml()`

### 防过拟合措施

| 措施 | 代码实现 |
|------|----------|
| 时间序列分割 | `X_train = X[:split_idx]` |
| L1/L2 正则化 | `reg_alpha=0.1, reg_lambda=1.0` |
| 限制树深度 | `max_depth=4` |
| 子采样 | `subsample=0.8, colsample_bytree=0.8` |
| 特征重要性监控 | `model.feature_importances_` |

### 过拟合诊断

```
Train R2 = 0.05, OOS R2 = 0.03 → Gap = 0.02 (Good)
Train R2 = 0.15, OOS R2 = 0.01 → Gap = 0.14 (Overfit!)
```

---

## Layer 7: 投资组合优化 — 该买多少

### 核心问题

因子信号告诉你"AAPL 比 INTC 好"。但该买 30% 的 AAPL 还是 10%？

代码位置: `quantportal/optimizer/portfolio.py`

### 三种方法

**1. 最大夏普比率 (Max Sharpe)**

```
maximize  w' μ / sqrt(w' Σ w)
subject to  sum(w) = 1, w >= 0, w <= max_weight
```

用 cvxpy 求解。因为 Sharpe 不是凸函数，所以用网格搜索风险厌恶系数 λ：

```python
for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    maximize  w'μ - λ * w'Σw
    track best Sharpe
```

**2. 最小方差 (Min Variance)**

```
minimize  w' Σ w
subject to  sum(w) = 1, w >= 0
```

不需要预期收益估计。对估计误差更稳健。

**3. 风险平价 (Risk Parity)**

```
每个资产对组合风险的贡献相等
```

简化实现：反向波动率加权

```python
w = 1 / asset_vols
w = w / sum(w)
```

### 分散化比率

```
分散化比率 = 加权平均波动率 / 组合波动率
```

> 1.0 说明分散化有效（相关性帮助降低了风险）。

---

## Layer 8: 回测引擎 — 时间旅行测试

### Walk-Forward 回测

```
Day 1 ────────────────────────── Day N
[     past data     ][ rebalance ][     hold     ][ rebalance ]...
```

每次只用**历史数据**计算信号和权重。这避免了前瞻偏差。

代码位置: `quantportal/backtest/engine.py`

### 交易成本

```python
# 每次再平衡的换手率
turnover = abs(new_weights - old_weights).sum()

# 交易成本 = 换手率 × 单边成本
transaction_cost = turnover * 0.001  # 10 bps 单边（机构水平）
```

没有交易成本的回测是童话故事。

### 关键指标

| 指标 | 公式 | 好的范围 |
|------|------|----------|
| **Sharpe Ratio** | Ann.Return / Ann.Vol | > 0.5 |
| **Max Drawdown** | (Peak - Trough) / Peak | > -30% |
| **Calmar Ratio** | Ann.Return / \|Max DD\| | > 0.5 |
| **Avg Turnover** | 平均每次换手率 | < 50% |

---

## Layer 9: 工程实践 — 测试与可视化

### 测试哲学

所有 27 个测试使用**合成数据**——不需要网络，不需要 API key。

```python
def _make_universe(n_assets=10, n_days=500, seed=42):
    """合成的多资产 Universe"""
    market = rng.normal(0.0003, 0.01, n_days)  # 共同市场因子
    for ticker in tickers:
        beta = 0.5 + rng.uniform(0, 1)
        idio = rng.normal(0, 0.005, n_days)
        returns[ticker] = market * beta + idio  # 因子模型
```

### 可视化

所有图表使用暗色主题，适合演示和论文：

```python
COLORS = {
    "bg": "#1a1a2e",      # 深蓝背景
    "primary": "#00d2ff",  # 青色主线
    "secondary": "#ff6b6b", # 红色对比
}
```

---

## 与 MarketDNA 的关系

### 技能互补

| MarketDNA 教你的 | QuantPortal 教你的 |
|------------------|-------------------|
| 什么是厚尾分布 | 怎么用厚尾做因子 (质量因子) |
| GARCH 怎么预测波动率 | 低波动异象怎么赚钱 |
| HMM 怎么检测 Regime | ML 怎么按 Regime 调权重 |
| 两只股票协不协整 | 怎么从 100 只股票中自动找到协整对 |
| 一个策略的 Sharpe 是多少 | 怎么组合 10 个策略到一个组合 |

### 学习顺序建议

```
1. 先跑 MarketDNA: scan("SPY") 理解单资产统计特性
2. 再跑 QuantPortal: quick_scan(10只股票) 理解多资产截面分析
3. 对比: 单因子 vs 多因子 vs ML组合
4. 深入: 修改因子参数，观察对 Sharpe 的影响
5. 挑战: 自己设计一个新因子，加入组合测试
```

---

## 下一步学习

### 可以自己尝试的改进

1. **加入新因子**:
   - 成交量因子: 异常成交量 → 信息不对称
   - 相关性因子: 和市场相关性低的股票 → 分散化价值
   - 技术指标因子: RSI, MACD 等

2. **改进 ML 模型**:
   - 尝试 XGBoost, CatBoost 对比
   - 加入滞后特征（过去5天的因子值变化）
   - 尝试分类模型（预测涨跌而非收益率）

3. **改进优化器**:
   - Black-Litterman 模型（结合主观观点）
   - 鲁棒优化（考虑估计误差）
   - 多目标优化（Sharpe + 换手率）

4. **改进回测**:
   - 加入滑点模型
   - 加入融资成本
   - 加入行业约束（不超配某个行业）

### 推荐阅读

| 书名 | 关键内容 |
|------|----------|
| Quantitative Equity Portfolio Management (Chincarini & Kim) | 因子模型理论 |
| Advances in Financial Machine Learning (Marcos Lopez de Prado) | ML 在金融中的正确用法 |
| Active Portfolio Management (Grinold & Kahn) | 信息比率、组合优化 |
| Efficiently Inefficient (Pedersen) | 各种策略的直觉和实证 |

---

<p align="center">
  <sub>本教程配合 QuantPortal 代码使用效果最佳。每个概念都有对应的代码文件。</sub>
</p>
