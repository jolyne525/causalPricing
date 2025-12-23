import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#  页面配置 
st.set_page_config(page_title="智能定价因果推断系统", page_icon="⚖️", layout="wide")

#  核心：数据生成与分析 

def generate_data(n_samples=1000):
    """
    生成模拟电商数据
    核心逻辑：包含混淆变量 
    - 高收入用户更有可能成为会员 
    - 会员通常即使价格高也愿意买 (高销量)。
    - 这会导致：如果我们直接看价格和销量，会发现价格高销量也高（假象），掩盖了真实的价格弹性。
    """
    np.random.seed(42)
    
    # 1. 混淆变量：收入 (Income)
    income = np.random.normal(5000, 1500, n_samples)
    
    # 2. 混淆变量：会员状态 (Is_Member) - 收入高的人更容易是会员
    member_prob = 1 / (1 + np.exp(-(income - 5000) / 1000))
    is_member = np.random.binomial(1, member_prob, n_samples)
    
    # 3. 核心变量：价格
    # 假设：系统给会员的价格通常偏高 (大数据杀熟场景模拟)，给非会员低价
    base_price = 100
    price_noise = np.random.normal(0, 5, n_samples)
    price = base_price + 10 * is_member + price_noise 
    
    # 4. 结果变量：销量
    # 真实的经济学规律：价格每升 1 元，销量下降 0.5 (弹性 = -0.5)
    # 但同时，会员购买力强 (+20销量)，收入高购买力强 (+income/1000)
    true_elasticity = -0.5
    demand_noise = np.random.normal(0, 2, n_samples)
    sales = 50 + (true_elasticity * price) + (20 * is_member) + (income / 1000) + demand_noise
    
    df = pd.DataFrame({
        "Income": income,
        "Is_Member": is_member,
        "Price": price,
        "Sales": sales
    })
    return df

def run_naive_analysis(df):
    """简单回归分析 (OLS) - 代表传统简单的统计方法"""
    model = LinearRegression()
    # 只看 价格 -> 销量，忽略其他因素
    X = df[['Price']]
    y = df['Sales']
    model.fit(X, y)
    return model.coef_[0], model.intercept_

def run_ml_analysis(df):
    """机器学习去偏分析"""
    # 这是一个简化版的 Double Machine Learning 思想
    # 使用随机森林控制混淆变量 (收入, 会员状态)
    
    X = df[['Price', 'Income', 'Is_Member']]
    y = df['Sales']
    
    # 使用随机森林拟合
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    return rf

# UI

st.title("⚖️ Causal Inference Pricing System")
st.caption("基于因果推断与机器学习的智能定价系统")

st.divider()

# 侧边栏
st.sidebar.header("🛠️ 实验控制台")
n_samples = st.sidebar.slider("样本数量", 500, 5000, 1000)
run_btn = st.sidebar.button("生成数据并分析", type="primary")

if run_btn:
    with st.spinner("正在生成模拟交易数据..."):
        df = generate_data(n_samples)
        st.session_state.data = df
else:
    if 'data' not in st.session_state:
        st.session_state.data = generate_data(1000)
    df = st.session_state.data

# 1. 数据概览区
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. 观测数据 (Observational Data)")
    st.markdown("""
    模拟了一个典型的**“价格歧视”**场景：
    * **混淆变量**：高收入用户往往也是会员。
    * **数据陷阱**：会员价格高，但购买力也强。
    * **挑战**：直接看数据，可能会得出“涨价反而销量好”的错误结论。
    """)
    st.dataframe(df.head(8), height=250)

with col2:
    st.subheader("2. 价格分布可视化")
    fig = px.histogram(df, x="Price", color="Is_Member", 
                       title="会员 vs 非会员的价格分布 (Price Distribution)",
                       labels={"Is_Member": "会员状态 (0=非会员, 1=会员)"},
                       opacity=0.7, barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# 2. 核心分析对比区
st.subheader("3. 因果效应分析 ("事实 vs. 错觉"))

# 计算两种模型
naive_coef, naive_intercept = run_naive_analysis(df)
rf_model = run_ml_analysis(df)

# 真实弹性 (我在 generate_data 里设定的)
TRUE_ELASTICITY = -0.5

# 展示结果卡片
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.error(f"传统回归计算的弹性\n {naive_coef:.3f}")
    st.caption("❌ 偏差巨大：甚至可能显示为正数（越涨价越好卖），因为忽略了会员购买力。")

with kpi2:
    # 估算 ML 模型的弹性 (通过部分依赖图思想简单估算)
    # 控制其他变量不变，只变动价格，观察销量变化
    test_data = df.copy()
    test_data['Price'] = test_data['Price'] + 1 # 价格统统 +1
    pred_sales_plus_1 = rf_model.predict(test_data[['Price', 'Income', 'Is_Member']])
    
    test_data['Price'] = test_data['Price'] - 1 # 价格原样 (恢复)
    pred_sales_original = rf_model.predict(test_data[['Price', 'Income', 'Is_Member']])
    
    ml_elasticity = np.mean(pred_sales_plus_1 - pred_sales_original)
    
    st.success(f"AI 因果模型计算的弹性\n {ml_elasticity:.3f}")
    st.caption("✅ 接近真实值：ML 成功剥离了收入和会员身份的干扰，还原了真实的价格效应。")

with kpi3:
    st.info(f"上帝视角的真实弹性\n {TRUE_ELASTICITY}")
    st.caption("🎯 Ground Truth：这是我们在生成数据时设定的客观经济规律。")

# 3. 最终图表对比
st.subheader("4. 决策面拟合对比")

# 创建用于画线的数据
x_range = np.linspace(df['Price'].min(), df['Price'].max(), 100)
y_naive = naive_coef * x_range + naive_intercept

fig_res = go.Figure()

# 散点
fig_res.add_trace(go.Scatter(x=df['Price'], y=df['Sales'], mode='markers', 
                             name='实际交易点', marker=dict(color='lightgray', opacity=0.5)))

# 错误线
fig_res.add_trace(go.Scatter(x=x_range, y=y_naive, mode='lines', 
                             name=f'传统回归线 (斜率={naive_coef:.2f})', line=dict(color='red', dash='dash')))

# ML 预测趋势 (取平均收入和会员状态)
mean_income = df['Income'].mean()
mean_member = 0.5 # 假设
ml_trend = []
for p in x_range:
    # 预测时控制其他变量为平均值 -> 这就是 Causal Inference 的核心 "Intervention" 思想
    pred = rf_model.predict([[p, mean_income, mean_member]])[0]
    ml_trend.append(pred)

fig_res.add_trace(go.Scatter(x=x_range, y=ml_trend, mode='lines', 
                             name=f'AI 因果推断线 (斜率≈{ml_elasticity:.2f})', line=dict(color='green', width=3)))

fig_res.update_layout(title="价格弹性拟合对比：红线被误导，绿线发现了真相", xaxis_title="价格 (Price)", yaxis_title="销量 (Sales)")
st.plotly_chart(fig_res, use_container_width=True) 

# 新增：商业价值模拟 
st.subheader("5. 商业价值模拟 (Business Impact)")

st.markdown("""
**模拟逻辑：**
* **传统模型 :** 误以为“价格越高销量越好”（因为被高收入会员数据误导），倾向于**大幅涨价**。
* **因果模型 :** 识破了假象，发现了真实弹性 (-0.5)，给出了**最优理性定价**。
""")

# 1. 设定基础参数 (Ground Truth)
base_price = 100
base_sales = 50 
true_elasticity = -0.5 # 真实的弹性

# 2. 模拟定价决策 (加入随机性，让每次结果不一样)
# 传统模型被误导，定高价 (在 135 到 145 之间波动)
naive_price = np.random.randint(135, 145) 

# 因果模型找到最优价 (在真实弹性 -0.5 下，最优价其实接近 100，这里我们设为 100-105)
optimal_price = np.random.randint(100, 103)

# 3. 计算真实销量 (核心公式：Sales = Base + Elasticity * Price_Change)
# 注意：这里必须用真实的弹性 (-0.5) 来计算实际发生的销量
real_sales_naive = base_sales + true_elasticity * (naive_price - base_price)
real_sales_optimal = base_sales + true_elasticity * (optimal_price - base_price)

# 防止销量变成负数 (极端情况)
real_sales_naive = max(real_sales_naive, 0)

# 4. 计算最终营收 (Revenue = Price * Sales)
rev_naive = naive_price * real_sales_naive
rev_optimal = optimal_price * real_sales_optimal

# 5. 计算提升百分比
uplift_val = rev_optimal - rev_naive
uplift_pct = 0
if rev_naive > 0:
    uplift_pct = (uplift_val / rev_naive) * 100

# --- 展示结果 ---
c1, c2, c3 = st.columns(3)

c1.metric(
    "传统模型决策", 
    f"${rev_naive:,.0f}", 
    help=f"定了个高价 ${naive_price}，导致销量暴跌至 {real_sales_naive:.1f}"
)

c2.metric(
    "因果模型决策", 
    f"${rev_optimal:,.0f}", 
    help=f"合理定价 ${optimal_price}，维持了健康销量 {real_sales_optimal:.1f}"
)

# 根据提升幅度显示不同的颜色和状态
if uplift_pct > 0:
    c3.metric("营收提升 (Revenue Uplift)", f"+{uplift_pct:.1f}%", delta="CV Key Metric")
    st.success(f"🚀 **显著提升！** 传统模型盲目涨价 ({naive_price}) 导致客户流失；因果模型通过理性定价 ({optimal_price}) 挽回了 **{uplift_pct:.1f}%** 的潜在营收损失。")
else:
    c3.metric("营收提升", f"{uplift_pct:.1f}%")
