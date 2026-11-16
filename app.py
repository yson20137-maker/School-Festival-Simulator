import hashlib
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

CATEGORIES = ['음료', '먹거리', '디저트', '굿즈·소품', '체험·서비스', '기타']
FEATURES = ['category', 'price_ratio', 'popularity', 'hot', 'rain', 'lunch']
DEFAULT_PRODUCTS = [
    {'판매': True, '상품명': '우리 반 에이드', '종류': '음료', '개당 원가(원)': 1000, '판매가(원)': 2500, '준비 수량': 60, '인기도 가정(1~5)': 3},
    {'판매': True, '상품명': '직접 만든 키링', '종류': '굿즈·소품', '개당 원가(원)': 1500, '판매가(원)': 3500, '준비 수량': 30, '인기도 가정(1~5)': 3},
]
# 아래 기준 가격은 현실 통계가 아니라 교육용 모델의 가정입니다.
REFERENCE_PRICE = np.array([2500, 4000, 3000, 4000, 3000, 3000], dtype=float)


def train_model():
    """가상의 구매 성향 점수를 학습. 실제 판매 실적을 학습한 모델이 아님."""
    rng = np.random.default_rng(42)
    n = 7000
    category = rng.integers(0, len(CATEGORIES), n)
    ratio = rng.uniform(0, 6, n)
    popularity = rng.integers(1, 6, n)
    hot = rng.integers(0, 2, n)
    rain = rng.integers(0, 2, n)
    lunch = rng.integers(0, 2, n)
    x = pd.DataFrame(dict(category=category, price_ratio=ratio,
                          popularity=popularity, hot=hot, rain=rain, lunch=lunch))
    score = (0.4 + 0.48 * (popularity - 3) - 1.35 * (ratio - 1)
             + 0.7 * hot * (category == 0)
             + 0.55 * lunch * (category == 1)
             - 0.3 * rain * np.isin(category, [0, 4])
             + rng.normal(0, 0.10, n))
    y = 1 / (1 + np.exp(-score))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=80, max_depth=12,
                                  min_samples_leaf=4, random_state=42, n_jobs=-1)
    model.fit(x_train[FEATURES], y_train)
    mae = mean_absolute_error(y_test, model.predict(x_test[FEATURES]))
    return model, float(mae)


@st.cache_resource
def get_model():
    return train_model()


def validate_products(frame):
    if frame is None or frame.empty:
        return [], ['상품을 한 개 이상 추가해 주세요.']
    rows, errors = [], []
    names = set()
    for index, raw in enumerate(frame.to_dict('records'), 1):
        selected = raw.get('판매')
        if selected is None or pd.isna(selected) or not bool(selected):
            continue
        name = str(raw.get('상품명') or '').strip()
        if name in ('nan', 'None'):
            name = ''
        if not name:
            errors.append(f'{index}행: 상품명을 입력해 주세요.')
            continue
        if name in names:
            errors.append(f'{index}행: 상품명 “{name}”이 중복됩니다. 다른 이름을 붙여 주세요.')
            continue
        names.add(name)
        category = raw.get('종류')
        if category not in CATEGORIES:
            errors.append(f'{name}: 종류를 선택해 주세요.')
            continue
        nums = {}
        valid = True
        for key in ['개당 원가(원)', '판매가(원)', '준비 수량', '인기도 가정(1~5)']:
            try:
                value = float(raw.get(key))
                if not math.isfinite(value) or value != int(value):
                    raise ValueError
                nums[key] = int(value)
            except (TypeError, ValueError, OverflowError):
                errors.append(f'{name}: {key}에 정수를 입력해 주세요.')
                valid = False
        if not valid:
            continue
        if not (0 <= nums['개당 원가(원)'] <= 100000 and 0 <= nums['판매가(원)'] <= 100000):
            errors.append(f'{name}: 원가와 판매가는 0~100,000원 범위여야 합니다.')
        elif not 0 <= nums['준비 수량'] <= 10000:
            errors.append(f'{name}: 준비 수량은 0~10,000개여야 합니다.')
        elif not 1 <= nums['인기도 가정(1~5)'] <= 5:
            errors.append(f'{name}: 인기도 가정은 1~5여야 합니다.')
        else:
            rows.append({'상품명': name, '종류': category, **nums})
    if not rows and not errors:
        errors.append('판매할 상품에 체크하고 상품을 입력해 주세요.')
    if len(rows) > 20:
        errors.append('원활한 비교를 위해 한 번에 최대 20종까지 판매할 수 있습니다.')
    return rows, errors


def simulate(products, visitors, customer_budget, weather, period, advertising, fixed_cost, model):
    """기대값 기반 배분: 동일 조건에는 같은 결과, 손님은 최대 1개만 구매."""
    data = pd.DataFrame(products)
    category = data['종류'].map({v: i for i, v in enumerate(CATEGORIES)}).to_numpy()
    price = data['판매가(원)'].to_numpy(dtype=float)
    qty = data['준비 수량'].to_numpy(dtype=int)
    cost = data['개당 원가(원)'].to_numpy(dtype=int)
    x = pd.DataFrame({
        'category': category,
        'price_ratio': np.minimum(price / REFERENCE_PRICE[category], 6),
        'popularity': data['인기도 가정(1~5)'].to_numpy(),
        'hot': int(weather == '더움'), 'rain': int(weather == '비'),
        'lunch': int(period == '점심 시간'),
    })
    p = np.clip(model.predict(x[FEATURES]), 0.01, 0.99)
    # 1만 원당 초반 홍보효과가 크고 이후 둔화되는 교육용 가정.
    ad_factor = 1 + 0.25 * np.log1p(advertising / 10000)
    weights = p / (1 - p) * ad_factor
    # 학습 범위 이상 고가에 대해 고정 예측이 되는 트리 모델의 한계를 보정.
    weights *= np.exp(-np.maximum(price / REFERENCE_PRICE[category] - 6, 0))
    weights[(price > customer_budget) | (qty == 0)] = 0
    # '우리 부스에서 사지 않음'도 하나의 선택지로 넣어 무조건 구매를 방지.
    probabilities = np.append(weights, 1.5)
    probabilities = probabilities / probabilities.sum()
    expected = visitors * probabilities
    allocated = np.floor(expected).astype(int)
    left = int(visitors - allocated.sum())
    if left:
        order = np.argsort(-(expected - allocated), kind='stable')
        allocated[order[:left]] += 1
    demand = allocated[:-1]
    sold = np.minimum(demand, qty)
    remaining = qty - sold
    revenue = sold * price.astype(int)
    prep = qty * cost
    result = data[['상품명', '종류', '판매가(원)', '준비 수량']].copy()
    result['배분 수요(개)'] = demand
    result['예상 판매(개)'] = sold
    result['남은 수량(개)'] = remaining
    result['기회손실 수량(개)'] = np.maximum(demand - qty, 0)
    result['매출(원)'] = revenue
    result['준비비(원)'] = prep
    result['상품 손익(원)'] = revenue - prep
    prep_total = int(prep.sum())
    revenue_total = int(revenue.sum())
    total_cost = prep_total + int(advertising) + int(fixed_cost)
    fingerprint = hashlib.sha256(json.dumps(
        dict(products=products, visitors=visitors, budget=customer_budget, weather=weather,
             period=period, advertising=advertising, fixed_cost=fixed_cost),
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
    return dict(table=result, revenue=revenue_total, cost=total_cost,
                profit=revenue_total-total_cost, sold=int(sold.sum()),
                remaining=int(remaining.sum()), missed=int(np.maximum(demand-qty, 0).sum()),
                non_buyers=int(visitors-sold.sum()), prep=prep_total, fingerprint=fingerprint)


def main():
    st.set_page_config(page_title='내가 축제 사장이라면?', page_icon='🏪', layout='wide')
    st.title('🏪 내가 축제 사장이라면?')
    st.caption('나만의 상품을 만들고 가격과 수량을 정해 보는 학교 축제 경영 시뮬레이터')
    st.info('교육용 가상 실험입니다. 실제 학교의 판매 기록을 학습하지 않았으며, 실제 매출을 보장하거나 판매를 권장하는 도구가 아닙니다.')
    model, mae = get_model()

    with st.sidebar:
        st.header('🎪 공통 축제 조건')
        visitors = st.number_input('우리 부스 앞 잠재 손님 수(명)', 1, 10000, 300, 10)
        customer_budget = st.number_input('손님 1명의 구매 예산(원)', 0, 100000, 7000, 500)
        weather = st.selectbox('날씨', ['보통', '더움', '비'])
        period = st.selectbox('운영 시간대', ['점심 시간', '그 외 시간'])
        st.caption('손님 1명은 우리 부스에서 최대 상품 1개를 삽니다. 모든 손님의 예산은 같다고 가정합니다.')
        st.header('💰 우리 가게 비용')
        capital = st.number_input('사용 가능한 준비 예산(원)', 0, 10000000, 300000, 10000)
        advertising = st.number_input('광고비(원)', 0, 1000000, 10000, 1000)
        fixed_cost = st.number_input('부스·장비 등 고정비(원)', 0, 1000000, 20000, 1000)

    st.subheader('1. 팔고 싶은 상품을 자유롭게 입력해요')
    st.write('아래 칸을 클릭해 이름·종류·원가·가격·수량을 바꿔 보세요. 맨 아래 빈 행에서 상품을 추가할 수 있어요. 최대 20종입니다.')
    st.caption('예: 타코야끼, 과일컵, 마카롱, 키링, 스티커, 사진 촬영, 미니 게임 체험권 등. 학교에서 허용되는 품목과 운영 방식인지 먼저 확인하세요.')
    edited = st.data_editor(
        pd.DataFrame(DEFAULT_PRODUCTS), num_rows='dynamic', hide_index=True,
        width='stretch', key='products_editor',
        column_config={
            '판매': st.column_config.CheckboxColumn('판매', default=True),
            '상품명': st.column_config.TextColumn('상품명', required=True, max_chars=60),
            '종류': st.column_config.SelectboxColumn('종류', options=CATEGORIES, required=True, default='기타'),
            '개당 원가(원)': st.column_config.NumberColumn(min_value=0, max_value=100000, step=100, default=1000, required=True),
            '판매가(원)': st.column_config.NumberColumn(min_value=0, max_value=100000, step=100, default=3000, required=True),
            '준비 수량': st.column_config.NumberColumn(min_value=0, max_value=10000, step=1, default=30, required=True),
            '인기도 가정(1~5)': st.column_config.NumberColumn(min_value=1, max_value=5, step=1, default=3, required=True),
        })
    st.caption('인기도는 실제 조사 결과가 아닌 직접 정하는 가정입니다. 친구끼리 겨룰 때는 모두 3으로 고정하세요. 체험·서비스의 원가는 1회당 재료비, 수량은 운영 가능한 횟수로 입력하세요.')
    products, errors = validate_products(edited)
    for error in errors:
        st.error(error)
    preparation = sum(p['개당 원가(원)'] * p['준비 수량'] for p in products)
    needed = preparation + advertising + fixed_cost
    st.write(f'**필요한 준비비: {needed:,}원** = 상품 준비비 {preparation:,}원 + 광고비 {advertising:,}원 + 고정비 {fixed_cost:,}원')
    over_budget = needed > capital
    if over_budget:
        st.warning(f'예산보다 {needed-capital:,}원 초과했어요. 준비 수량이나 비용을 줄이거나 예산을 조정해 주세요.')
    if not errors and any(p['판매가(원)'] < p['개당 원가(원)'] for p in products):
        st.warning('원가보다 싸게 판매하는 상품이 있어요. 많이 팔아도 손해가 날 수 있습니다.')

    st.subheader('2. 장사 결과 확인하기')
    st.caption('아래 결과는 마지막으로 누른 실행 조건을 기준으로 합니다. 설정을 바꾼 뒤 다시 실행해 주세요.')
    if st.button('🚀 이 전략으로 장사 시작!', type='primary', disabled=bool(errors) or over_budget):
        outcome = simulate(products, visitors, customer_budget, weather, period, advertising, fixed_cost, model)
        st.session_state['outcome'] = outcome
        st.session_state['last_conditions'] = dict(잠재손님=visitors, 손님예산=customer_budget,
                                                   날씨=weather, 시간대=period, 광고비=advertising,
                                                   고정비=fixed_cost, 준비예산=capital)
    outcome = st.session_state.get('outcome')
    if outcome:
        st.write('**실행 당시 조건:**', st.session_state['last_conditions'])
        a, b, c, d = st.columns(4)
        a.metric('예상 매출', f"{outcome['revenue']:,}원")
        b.metric('전체 비용', f"{outcome['cost']:,}원")
        c.metric('예상 순이익', f"{outcome['profit']:,}원")
        d.metric('예상 판매량', f"{outcome['sold']:,}개")
        st.dataframe(outcome['table'], hide_index=True, width='stretch')
        st.caption('상품 손익에는 광고비·고정비가 빠져 있습니다. 전체 순이익에서는 두 비용을 한 번만 추가로 차감합니다.')
        st.write(f"📦 남은 수량 **{outcome['remaining']:,}개** · 품절로 놓친 수요 **{outcome['missed']:,}개** · 구매하지 못했거나 선택하지 않은 손님 **{outcome['non_buyers']:,}명**")
        st.markdown('**전략 점검**')
        if outcome['remaining']:
            st.write('• 남은 상품이 있어요. 가격을 낮추거나 준비 수량을 줄인 전략과 비교해 보세요.')
        if outcome['missed']:
            st.write('• 일부 상품이 품절됐어요. 준비 예산 안에서 수량을 늘린 결과를 비교해 보세요.')
        if outcome['profit'] < 0:
            st.write('• 적자예요. 매출뿐 아니라 전체 준비비와 광고비도 함께 살펴보세요.')
        st.caption('위 문장은 계산 결과에 따른 규칙 기반 안내이며, 생성형 AI의 조언이 아닙니다.')
        st.download_button('📥 상품별 결과 CSV 다운로드',
                           outcome['table'].to_csv(index=False).encode('utf-8-sig'),
                           'festival_result.csv', 'text/csv')
        strategy_name = st.text_input('비교표에 남길 전략 이름', value='내 전략', max_chars=40)
        if st.button('📌 현재 실행 결과를 비교표에 저장'):
            history = st.session_state.setdefault('history', [])
            history.append({'전략': strategy_name.strip() or '내 전략',
                            **st.session_state['last_conditions'],
                            '매출(원)': outcome['revenue'], '전체비용(원)': outcome['cost'],
                            '순이익(원)': outcome['profit'], '판매량': outcome['sold'],
                            '상품설정': outcome['table'][['상품명', '판매가(원)', '준비 수량']].to_json(orient='records', force_ascii=False),
                            '실행ID': outcome['fingerprint']})
            st.success('이 브라우저 세션의 비교표에 저장했어요.')
    if st.session_state.get('history'):
        st.subheader('3. 전략 비교')
        st.dataframe(pd.DataFrame(st.session_state['history']), hide_index=True, width='stretch')
        st.download_button('📥 비교표 CSV 다운로드',
                           pd.DataFrame(st.session_state['history']).to_csv(index=False).encode('utf-8-sig'),
                           'festival_strategies.csv', 'text/csv')
        st.caption('서버나 계정에 영구 저장하지 않습니다. 연결 종료·새로고침 등으로 기록이 사라질 수 있으니 CSV를 받아 두세요. 공정한 비교를 위해 손님 수·날씨·예산·인기도를 같게 설정하세요.')

    with st.expander('🤖 AI의 역할과 계산 가정 — 발표 전에 꼭 읽기'):
        st.markdown('''
- **진짜 학습 모델:** 가상 데이터 7,000건을 만들고, 그중 80%로 랜덤 포레스트 회귀 모델을 학습하고 20%로 평가합니다.
- **학습하지 않은 것:** 실제 학교 매출·학생 선호·상품명 의미는 학습하지 않았습니다. 상품명을 바꿔도 다른 조건이 같으면 예측은 같습니다.
- **수요 배분:** AI 점수를 선택 가중치로 바꿔, 여러 상품과 '구매 안 함' 사이에 한정된 손님 수를 나눕니다. 상품마다 손님 수를 따로 더하지 않습니다.
- **실제 판매량:** 배분 수요와 준비 수량 중 작은 값입니다. 품절 손님이 다른 상품으로 갈아타는 행동은 반영하지 않습니다.
- **재고 처리:** 모든 준비비를 당일 비용으로 처리하고, 남은 음식·굿즈 모두 잔존가치를 0원으로 봅니다. 굿즈 재판매나 식품 폐기비는 반영하지 않습니다.
- **손님 가정:** 손님마다 예산이 같고 최대 1개만 구매합니다. 다른 축제 부스는 별도 모델링하지 않고 '구매 안 함'에 간접 포함합니다.
- **홍보 가정:** 광고비의 효과는 줄어드는 비율로 증가하는 수식입니다. AI가 실제 광고 성과를 배운 것이 아닙니다.
- **인기도 가정:** 사용자가 입력한 가정입니다. 높은 값이 실제 인기를 뜻하지 않습니다. 똑같은 상품을 여러 이름으로 복제하면 수요가 왜곡될 수 있습니다.
- **가격 범위:** 종류별 기준 가격의 최대 6배까지 학습하고, 그 이상의 가격은 가중치를 낮추는 추가 수식으로 처리합니다.
- **전체 순이익 = 매출 − (개당 원가 × 준비 수량의 합) − 광고비 − 고정비.** 세금·인건비 등은 원가나 고정비에 직접 포함해야 합니다.
- 같은 조건에는 같은 결과가 나옵니다. 다만 정수 배분·품절·모델 특성 때문에 설정을 조금 바꿔도 결과가 같을 수 있습니다.
''')
        st.write(f'가상 평가 데이터의 구매 성향 점수 MAE: {mae:.4f} (0~1 점수 단위)')
        st.caption('이 값은 가상 규칙을 얼마나 잘 따라 했는지를 나타냅니다. 실제 판매 예측 정확도가 아니며, 정확도 %로 표현하면 안 됩니다.')


if __name__ == '__main__':
    main()
