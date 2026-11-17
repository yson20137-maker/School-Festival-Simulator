"""학교 축제 시뮬레이터 — Streamlit Community Cloud용 단일 파일.
교육용 합성 데이터 수요 모델 + 제약 조건을 반영한 제한 탐색 추천.
"""
from __future__ import annotations
import copy
import hashlib
import html
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

CATS = ['따뜻한 먹거리', '차가운 먹거리', '음료', '디저트', '굿즈·소품', '체험·게임', '사진·서비스', '기타']
REF = [3500, 3500, 2500, 3000, 4000, 2500, 3500, 3000]
BLUE, ORANGE, INK, GRID = '#1751D0', '#E2661E', '#1E2433', '#E2E6F0'
PRESETS = {
    '직접 입력': ('기타', 1000, 3000),
    '붕어빵': ('따뜻한 먹거리', 450, 1200),
    '떡볶이': ('따뜻한 먹거리', 1300, 3500),
    '타코야끼': ('따뜻한 먹거리', 1400, 3500),
    '닭강정': ('따뜻한 먹거리', 1700, 4000),
    '소떡소떡': ('따뜻한 먹거리', 1000, 2500),
    '과일컵': ('차가운 먹거리', 1500, 3500),
    '샌드위치': ('차가운 먹거리', 1700, 4000),
    '에이드': ('음료', 800, 2500),
    '아이스티': ('음료', 500, 1800),
    '핫초코': ('음료', 900, 2500),
    '와플': ('디저트', 1000, 3000),
    '쿠키': ('디저트', 700, 2000),
    '츄러스': ('디저트', 800, 2500),
    '키링': ('굿즈·소품', 1500, 4000),
    '스티커': ('굿즈·소품', 300, 1500),
    '미니 게임': ('체험·게임', 300, 2000),
    '사진 촬영': ('사진·서비스', 500, 2500),
}


@st.cache_resource
def demand_model():
    rng = np.random.default_rng(241)
    n = 4500
    cat = rng.integers(0, len(CATS), n)
    pop = rng.uniform(1, 5, n)
    temp = rng.uniform(0, 35, n)
    meal = rng.integers(0, 2, n)
    rain = rng.integers(0, 2, n)
    x = np.column_stack([cat, pop, temp, meal, rain])
    z = (0.1 + .45 * (pop - 3) + .055 * (temp - 20) * (cat == 2)
         - .035 * (temp - 20) * (cat == 0)
         + .45 * meal * np.isin(cat, [0, 1]) - .22 * rain * np.isin(cat, [2, 5, 6]))
    y = np.exp(z + rng.normal(0, .045, n))
    model = RandomForestRegressor(n_estimators=45, max_depth=10, min_samples_leaf=5,
                                  random_state=241, n_jobs=1)
    model.fit(x, y)
    return model


def number(value, default=0.):
    try:
        val = float(value)
        return val if math.isfinite(val) else default
    except (ValueError, TypeError):
        return default


def text(value):
    return '' if value is None or str(value) in ('nan', 'None', '<NA>') else str(value).strip()


def allocate(expected, total):
    """최대 나머지 방식으로 비구매자를 포함한 전체 방문객 수를 보존."""
    arr = np.floor(expected).astype(int)
    left = int(total - arr.sum())
    if left > 0:
        arr[np.argsort(-(expected-arr), kind='stable')[:left]] += 1
    return arr


def prepare(lines, materials, config, model=None):
    if not lines:
        raise ValueError('상품을 한 개 이상 입력해 주세요.')
    keys = set()
    for row in lines:
        if not row['label'] or row['id'] in keys:
            raise ValueError('상품·옵션 이름을 서로 다르게 입력해 주세요.')
        keys.add(row['id'])
    for mat in materials:
        if mat['stock'] < 0 or mat['unit_cost'] < 0:
            raise ValueError('재료 수량과 단가는 0 이상이어야 합니다.')
    for row in lines:
        ci = CATS.index(row['category'])
        row['reference'] = REF[ci]
        row['cost'] = (row['extra_cost'] + sum(materials[i]['unit_cost'] * use for i, use in row['recipe'].items())
                       if row['has_recipe'] else row['unit_cost'])
        row['cost'] = round(row['cost'], 6)
    mdl = model if model is not None else demand_model()
    features = [[CATS.index(r['category']), r['pop'], config['temp'], int(config['meal']), int(config['rain'])] for r in lines]
    base = np.maximum(mdl.predict(np.asarray(features)), .01)
    groups = defaultdict(list)
    for i, r in enumerate(lines):
        groups[r['group']].append(i)
    # 같은 상품 옵션을 늘린 것만으로 전체 인기도가 부풀지 않도록 그룹 내 비중 합을 1로 제한.
    for indices in groups.values():
        shares = np.array([lines[i]['preference'] for i in indices], float)
        shares = shares / shares.sum()
        for i, share in zip(indices, shares):
            lines[i]['base'] = float(base[i] * share)
    return {'lines': lines, 'materials': materials, 'config': config}


def demand(ctx, prices):
    rows, cfg = ctx['lines'], ctx['config']
    p = np.asarray(prices, float)
    ref = np.array([r['reference'] for r in rows])
    base = np.array([r['base'] for r in rows])
    # 가격은 단조 감소 보정: 트리 모델의 학습 범위 밖 가격에서도 수요가 평평해지지 않게 처리.
    weights = base * np.exp(np.clip(-cfg['elasticity'] * (p/ref-1), -35, 10))
    weights[p > cfg['customer_budget']] = 0
    weights /= 1 + .22 * cfg['competition']
    # 옵션을 포함한 상품 1개 또는 구매하지 않음을 선택. 준비 수량 자체로 수요를 부풀리지 않음.
    scores = np.append(weights, cfg['outside'])
    expected = cfg['visitors'] * scores / scores.sum()
    return allocate(expected, cfg['visitors'])[:-1]


def needs(ctx, quantities):
    result = np.zeros(len(ctx['materials']), float)
    for row, qty in zip(ctx['lines'], quantities):
        for m, use in row['recipe'].items():
            result[m] += use * qty
    return result


def feasible(ctx, quantities):
    q = np.maximum(np.asarray(quantities, int), 0)
    rows, cfg, mats = ctx['lines'], ctx['config'], ctx['materials']
    if np.any(q > np.array([r['limit'] for r in rows])):
        return False
    if np.dot(q, [r['minutes'] for r in rows]) > cfg['capacity_minutes'] + 1e-7:
        return False
    if cfg['fixed'] + np.dot(q, [r['cost'] for r in rows]) > cfg['capital'] + 1e-7:
        return False
    required = needs(ctx, q)
    return all(m['can_buy'] or required[i] <= m['stock']+1e-7 for i, m in enumerate(mats))


def fit_quantities(ctx, desired, prices):
    """예산·시간·품목 상한·추가구매 불가 재료를 동시에 만족하는 수량 구성.
    분할 추가 시 손익 증가량/자원 비율이 큰 품목 우선의 휴리스틱이며 전역 최적화 아님.
    """
    rows, cfg, mats = ctx['lines'], ctx['config'], ctx['materials']
    desired = np.minimum(np.maximum(np.asarray(desired, int), 0), [r['limit'] for r in rows])
    costs = np.array([r['cost'] for r in rows], float)
    times = np.array([r['minutes'] for r in rows], float)
    margins = np.asarray(prices)-costs
    desired = np.where(margins > 0, desired, 0)
    if feasible(ctx, desired):
        return desired.astype(int)
    money = max(0., cfg['capital']-cfg['fixed'])
    duration = cfg['capacity_minutes']
    remaining = np.array([m['stock'] for m in mats], float)
    result = np.zeros(len(rows), int)
    scale = costs/max(money, 1) + times/max(duration, 1)
    for i, row in enumerate(rows):
        for j, use in row['recipe'].items():
            if not mats[j]['can_buy']:
                scale[i] += use/max(remaining[j], 1e-8)
    order = sorted(range(len(rows)), key=lambda i: -margins[i]/max(scale[i], 1e-8))
    for i in order:
        row = rows[i]
        count = int(desired[i])
        if costs[i] > 0:
            count = min(count, max(0, int((money+1e-8)//costs[i])))
        if times[i] > 0:
            count = min(count, max(0, int((duration+1e-8)//times[i])))
        for j, use in row['recipe'].items():
            if use > 0 and not mats[j]['can_buy']:
                count = min(count, max(0, int((remaining[j]+1e-8)//use)))
        result[i] = count
        money -= count*costs[i]
        duration -= count*times[i]
        for j, use in row['recipe'].items():
            remaining[j] -= count*use
    return result


def evaluate(ctx, prices, quantities):
    q = np.maximum(np.asarray(quantities, int), 0)
    prices = np.asarray(prices, int)
    d = demand(ctx, prices)
    sold = np.minimum(d, q)
    cost = float(np.dot(q, [r['cost'] for r in ctx['lines']]) + ctx['config']['fixed'])
    revenue = int(np.dot(sold, prices))
    return {'prices': prices, 'q': q, 'demand': d, 'sold': sold,
            'left': q-sold, 'missed': np.maximum(d-q, 0),
            'cost': cost, 'revenue': revenue, 'profit': revenue-cost,
            'feasible': feasible(ctx, q)}


def optimize(ctx):
    rows = ctx['lines']
    current_prices = np.array([r['price'] for r in rows], int)
    original = evaluate(ctx, current_prices, [r['quantity'] for r in rows])
    q = fit_quantities(ctx, demand(ctx, current_prices), current_prices)
    best = evaluate(ctx, current_prices, q)
    # 이미 좋은 사용자의 계획은 수량을 이유 없이 늘리지 않음. 동률이면 원안 보존.
    if original['feasible'] and original['profit'] >= best['profit']-1e-7:
        best = original
    if ctx['config']['adjust_price']:
        grids = []
        for row in rows:
            # 사용자 가격 ±30%, 100원 단위 후보. 원래 가격도 항상 포함.
            low = max(100, int(math.ceil(row['price']*.7/100)*100))
            high = min(100000, max(low, int(math.floor(row['price']*1.3/100)*100)))
            grid = ([row['price']] if row['price'] < 100 else
                    sorted(set([row['price']] + [int(round(p/100)*100) for p in np.linspace(low, high, 7)])))
            grids.append(grid)
        for _ in range(2):
            improved = False
            for i in range(len(rows)):
                for price in grids[i]:
                    candidate_prices = best['prices'].copy()
                    candidate_prices[i] = price
                    candidate_q = fit_quantities(ctx, demand(ctx, candidate_prices), candidate_prices)
                    candidate = evaluate(ctx, candidate_prices, candidate_q)
                    if candidate['feasible'] and candidate['profit'] > best['profit']+1e-7:
                        best, improved = candidate, True
            if not improved:
                break
    return original, best


def material_table(ctx, before, after):
    b, a = needs(ctx, before['q']), needs(ctx, after['q'])
    return pd.DataFrame([{
        '재료': m['name'], '단위': m['unit'], '보유량': m['stock'],
        '현재 계획 필요량': round(float(b[i]), 2), '추천 필요량': round(float(a[i]), 2),
        '준비량 조정': round(float(a[i]-b[i]), 2),
        '추가 구매량': round(max(0., float(a[i]-m['stock'])), 2),
        '추가 구매비(원)': round(max(0., float(a[i]-m['stock']))*m['unit_cost']),
    } for i, m in enumerate(ctx['materials'])])


def feedback(ctx, before, after):
    messages = []
    if not before['feasible']:
        messages.append(('계획 조정 필요', '현재 계획은 예산·운영 시간·수량 상한 또는 추가 구매 불가 재료 조건을 초과합니다. 추천안은 입력한 제한을 반영했습니다.'))
    for i, r in enumerate(ctx['lines']):
        delta = int(after['q'][i]-before['q'][i])
        if delta:
            direction = '늘리기' if delta > 0 else '줄이기'
            reason = '예상 수요 대비 부족한 준비량을 보완합니다.' if delta > 0 else '남는 수량 또는 자원 초과를 줄입니다.'
            if r['cost'] >= after['prices'][i] and after['q'][i] == 0:
                reason = '원가 대비 판매가가 낮아 판매를 보류하는 편이 유리합니다.'
            messages.append((f"{r['label']} · {abs(delta):,}개 {direction}",
                             f"{before['q'][i]:,}개 → {after['q'][i]:,}개. {reason}"))
        if before['prices'][i] != after['prices'][i]:
            messages.append((f"{r['label']} · 가격 조정", f"{before['prices'][i]:,}원 → {after['prices'][i]:,}원. 현재 가격 주변의 후보를 비교한 추천입니다."))
        if before['prices'][i] > ctx['config']['customer_budget']:
            messages.append((f"{r['label']} · 구매 예산 초과", '손님의 1인 예산보다 비쌉니다. 가격을 낮추거나 상품 구성을 바꿔 보세요.'))
    mats = material_table(ctx, before, after)
    for row in mats.to_dict('records'):
        if row['추가 구매량'] > 0:
            messages.append((f"{row['재료']} · 추가 준비", f"추천안 기준 {row['추가 구매량']:g}{row['단위']} 추가 구매가 필요합니다. 보유량을 뺀 수량입니다."))
    if after['profit'] <= 0:
        messages.append(('수익 구조 점검', '추천안도 예상 순이익이 0원 이하입니다. 고정비·원가를 줄이거나 상품 구성을 바꿔 다시 분석하세요.'))
    elif int(after['missed'].sum()) > 0:
        messages.append(('운영 여력 확인', f"추천안에도 미충족 수요 {int(after['missed'].sum()):,}개가 남습니다. 준비 예산·운영 인원·제작 시간·재료 제한을 확인하세요."))
    return messages


def signature(lines, mats, cfg):
    return hashlib.sha256(json.dumps([lines, mats, cfg], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def product_seed(name='붕어빵'):
    cat, cost, price = PRESETS.get(name, PRESETS['직접 입력'])
    return {'name': '' if name == '직접 입력' else name, 'category': cat, 'unit_cost': cost, 'price': price}


def style():
    st.markdown('''<style>
    .stApp { background:#FDFDFF; color:#1E2433; }
    .block-container { max-width:1390px; padding-top:2.3rem; padding-bottom:2rem; }
    [data-testid="stSidebar"] { background:#F6F8FC; min-width:340px; }
    h1 { font-size:2.15rem!important; letter-spacing:-.07rem; font-weight:800!important; }
    h2 { font-size:1.3rem!important; letter-spacing:-.035rem; }
    h3 { font-size:1.05rem!important; }
    p, label, .stCaption { word-break:keep-all; }
    [data-testid="stMetric"] { border:1px solid #E2E6F0; padding:18px 15px; border-radius:12px; background:#fff; }
    [data-testid="stMetricValue"] { font-size:1.65rem; font-variant-numeric:tabular-nums; }
    div.stButton > button[kind="primary"] { background:#1751D0; border:0; border-radius:9px; min-height:46px; font-weight:700; }
    .intro { font-size:1rem; color:#505A74; margin-top:-8px; margin-bottom:24px; }
    .empty { border:1px solid #E2E6F0; border-radius:16px; padding:44px 30px; margin-top:24px; }
    .empty b { font-size:1.35rem; display:block; margin-bottom:12px; }
    .empty p { color:#505A74; margin:0; }
    .note { font-size:.76rem; color:#737D96; margin-top:24px; }
    .rec { border:1px solid #E2E6F0; border-left:4px solid #1751D0; padding:14px 18px; border-radius:8px; margin:10px 0; }
    .rec b { display:block; margin-bottom:4px; }
    .rec p { color:#505A74; margin:0; font-size:.92rem; }
    @media (max-width:1100px) {
      [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:180px; flex:1 1 180px; }
      [data-testid="stHorizontalBlock"]:has([data-testid="stPlotlyChart"]) > [data-testid="stColumn"] { min-width:280px; }
    }
    </style>''', unsafe_allow_html=True)


def chart_group(names, first, second, labels, unit):
    fig = go.Figure()
    fig.add_trace(go.Bar(name=labels[0], y=names, x=list(first), orientation='h', marker_color=BLUE))
    fig.add_trace(go.Bar(name=labels[1], y=names, x=list(second), orientation='h', marker_color=ORANGE))
    fig.update_layout(barmode='group', height=max(240, min(850, 100+60*len(names))),
                      margin=dict(l=0,r=15,t=40,b=0), paper_bgcolor='#FDFDFF', plot_bgcolor='#FDFDFF',
                      font=dict(color=INK, family='sans-serif'),
                      legend=dict(orientation='h', y=1.18, x=0),
                      xaxis=dict(title=unit, gridcolor=GRID, zerolinecolor='#CDD3E0', rangemode='tozero'),
                      yaxis=dict(autorange='reversed', automargin=True))
    fig.update_traces(hovertemplate='%{y}<br>%{x:,.0f}'+unit+'<extra>%{fullData.name}</extra>')
    return fig


def render_results(ctx, before, after):
    gain = after['profit']-before['profit']
    st.subheader('시뮬레이션 결과')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('현재 예상 순이익', f"{before['profit']:,.0f}원")
    c2.metric('추천안 예상 순이익', f"{after['profit']:,.0f}원", f'{gain:+,.0f}원' if abs(gain)>.5 else '변동 없음')
    c3.metric('추천안 예상 판매', f"{int(after['sold'].sum()):,}개")
    c4.metric('추천안 남는 수량', f"{int(after['left'].sum()):,}개", f"{int(after['left'].sum()-before['left'].sum()):+,}개", delta_color='inverse')
    if not before['feasible']:
        st.warning('현재 계획은 입력한 운영 제한을 초과합니다. 현재 수치는 제한을 초과한 계획의 계산값이며, 추천안과 동일한 실행 조건이 아닙니다.')
    t1,t2,t3 = st.tabs(['계획 비교', '상품별 상세', '재료 준비'])
    names = [r['label'] for r in ctx['lines']]
    with t1:
        left,right=st.columns(2)
        with left:
            st.markdown('### 준비 수량 조정')
            st.plotly_chart(chart_group(names,before['q'],after['q'],['현재 계획','추천안'],'개'),width='stretch',key='quantities')
        with right:
            st.markdown('### 예상 손익 비교')
            st.plotly_chart(chart_group(['매출','전체 비용','순이익'],
                [before['revenue'],before['cost'],before['profit']],
                [after['revenue'],after['cost'],after['profit']],['현재 계획','추천안'],'원'),
                width='stretch',key='financial')
        overview=pd.DataFrame({'항목':['예상 매출','전체 준비 비용','예상 순이익'],
                               '현재 계획(원)':[before['revenue'],before['cost'],before['profit']],
                               '추천안(원)':[after['revenue'],after['cost'],after['profit']]})
        st.dataframe(overview,hide_index=True,width='stretch')
    details=pd.DataFrame({'상품·옵션':names,'개당 원가(원)':[round(r['cost'],2) for r in ctx['lines']],
                          '현재 가격(원)':before['prices'],'추천 가격(원)':after['prices'],
                          '현재 수량':before['q'],'추천 수량':after['q'],'예상 수요':after['demand'],
                          '예상 판매':after['sold'],'남는 수량':after['left'],'미충족 수요':after['missed']})
    with t2:
        st.markdown('### 추천안 수요와 준비량')
        st.plotly_chart(chart_group(names,after['demand'],after['q'],['예상 수요','추천 준비량'],'개'),width='stretch',key='demand')
        st.dataframe(details,hide_index=True,width='stretch')
        st.download_button('상품별 결과 저장',details.to_csv(index=False).encode('utf-8-sig'),'축제_판매계획.csv','text/csv')
    with t3:
        table=material_table(ctx,before,after)
        if table.empty:
            st.info('왼쪽 상품 설정에서 재료 상세 입력을 켜면 재료별 준비량이 표시됩니다.')
        else:
            st.dataframe(table,hide_index=True,width='stretch')
            st.caption('추가 구매비는 보유 재료를 뺀 구매금액입니다. 전체 준비 비용에는 사용할 재료 원가가 이미 포함되어 있어 중복 합산하지 않습니다.')
    st.subheader('계획 피드백')
    messages=feedback(ctx,before,after)
    if not messages:
        st.success('계획 점검 통과! 현재 탐색 범위에서는 더 유리한 조정안이 없습니다.')
    else:
        if after['profit'] > 0 and after['feasible']:
            st.caption('입력한 제한 안에서 비교한 조정안입니다.')
        for title,body in messages:
            st.markdown(f'<div class="rec"><b>{html.escape(title)}</b><p>{html.escape(body)}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="note">가상 데이터 기반 추정 · 추천안은 제한된 후보 탐색 결과이며 실제 매출이나 최적 결과를 보장하지 않습니다.</div>',unsafe_allow_html=True)


def sidebar_inputs():
    lines, materials, errors = [], [], []
    with st.sidebar:
        st.header('판매 계획')
        with st.expander('축제 운영 조건', expanded=True):
            visitors=st.number_input('부스 앞 예상 방문객(명)',0,10000,300,10)
            capital=st.number_input('총 준비 예산(원)',0,10000000,300000,10000)
            duration=st.number_input('운영 시간(분)',1,1440,180,30)
            staff=st.number_input('운영 인원(명)',1,100,3,1)
            fixed=st.number_input('장비·부스 등 고정비(원)',0,10000000,20000,1000)
        with st.expander('수요 조건'):
            customer_budget=st.number_input('손님 1인 구매 예산(원)',0,100000,5000,500)
            temp=st.slider('예상 기온(℃)',0,35,18)
            meal=st.toggle('식사 시간대',value=True)
            rain=st.toggle('비 오는 날')
            competition=st.number_input('비슷한 경쟁 부스 수',0,50,2)
            intent=st.select_slider('기본 구매 관심도',options=['낮음','보통','높음'],value='보통')
            sensitivity=st.select_slider('가격 민감도',options=['낮음','보통','높음'],value='보통')
        st.divider()
        count=st.number_input('판매할 상품 종류',1,12,1,1)
        all_recipe_records=[]
        for i in range(count):
            with st.expander(f'상품 {i+1}',expanded=(i==0)):
                preset=st.selectbox('상품 선택',list(PRESETS),index=list(PRESETS).index('붕어빵') if i==0 else 0,key=f'preset_{i}')
                seed=product_seed(preset)
                prefix=f'product_{i}_{preset}'
                name=st.text_input('상품명',value=seed['name'],max_chars=40,key=prefix+'name')
                category=st.selectbox('상품 분류',CATS,index=CATS.index(seed['category']),key=prefix+'category')
                price=st.number_input('판매 가격(원)',0,100000,seed['price'],100,key=prefix+'price')
                cost=st.number_input('개당 원가(원)',0,100000,seed['unit_cost'],50,key=prefix+'cost')
                qty=st.number_input('준비 수량',0,10000,80,10,key=prefix+'qty')
                pop=st.select_slider('예상 인기도',options=[1,2,3,4,5],value=3,key=prefix+'pop')
                minutes=st.number_input('1개당 작업 시간(분)',0.,120.,.5,.1,key=prefix+'minutes')
                limit=st.number_input('상품·옵션별 최대 준비 수량',0,10000,500,10,key=prefix+'limit')
                variants=st.checkbox('맛·종류별 옵션 설정',key=prefix+'variants')
                if variants:
                    defaults=([{'옵션명':'팥붕','판매가(원)':price,'개당 원가(원)':cost,'준비 수량':qty//2,'선호도':3},
                               {'옵션명':'슈붕','판매가(원)':price,'개당 원가(원)':cost+50,'준비 수량':qty-qty//2,'선호도':3}]
                              if preset=='붕어빵' else [{'옵션명':'기본','판매가(원)':price,'개당 원가(원)':cost,'준비 수량':qty,'선호도':3}])
                    st.caption('옵션을 켜면 아래 표의 가격·원가·수량이 적용됩니다.')
                    edited=st.data_editor(pd.DataFrame(defaults),num_rows='dynamic',hide_index=True,key=prefix+'options',
                        column_config={
                            '옵션명':st.column_config.TextColumn(required=True,max_chars=30),
                            '판매가(원)':st.column_config.NumberColumn(min_value=0,max_value=100000,step=100,required=True),
                            '개당 원가(원)':st.column_config.NumberColumn(min_value=0,max_value=100000,step=50,required=True),
                            '준비 수량':st.column_config.NumberColumn(min_value=0,max_value=10000,step=1,required=True),
                            '선호도':st.column_config.NumberColumn(min_value=1,max_value=5,step=1,required=True)})
                    if edited.empty or len(edited)>8:
                        errors.append(f'상품 {i+1}: 옵션은 1~8개로 입력해 주세요.')
                    variant_rows=edited.to_dict('records')[:8]
                else:
                    variant_rows=[{'옵션명':'','판매가(원)':price,'개당 원가(원)':cost,'준비 수량':qty,'선호도':3}]
                product_lines=[]
                for j,v in enumerate(variant_rows):
                    option=text(v.get('옵션명'))
                    if variants and not option:
                        errors.append(f'상품 {i+1}: 옵션 이름을 입력해 주세요.')
                    label=f'{name} · {option}' if variants else name
                    numeric={k:number(v.get(k),-1) for k in ['판매가(원)','개당 원가(원)','준비 수량','선호도']}
                    if any(x<0 for x in numeric.values()) or any(x!=int(x) for x in numeric.values()) or not 1<=numeric['선호도']<=5:
                        errors.append(f'{label}: 옵션의 가격·원가·수량은 정수, 선호도는 1~5로 입력해 주세요.')
                    row={'id':f'{i}_{j}','group':str(i),'label':label,'category':category,
                         'price':max(0,int(numeric['판매가(원)'])),'unit_cost':max(0,int(numeric['개당 원가(원)'])),
                         'quantity':max(0,int(numeric['준비 수량'])),'pop':pop,'preference':max(1,numeric['선호도']),
                         'minutes':minutes,'limit':limit,'recipe':{},'has_recipe':False,'extra_cost':0.}
                    product_lines.append(row)
                recipe_on=st.checkbox('재료 상세 입력',key=prefix+'recipe')
                if recipe_on:
                    st.caption('옵션별 1개 사용량을 입력하세요. 같은 재료명은 공용 재고로 합산합니다. 재료 원가와 부대비가 위 개당 원가를 대체합니다.')
                    extra=st.number_input('개당 포장·기타 비용(원)',0,100000,100,50,key=prefix+'extra')
                    if preset=='붕어빵' and variants:
                        default_recipe=[{'옵션명':'팥붕','재료명':'반죽','단위':'g','1개 사용량':40.,'보유량':3000.,'단위당 원가(원)':3.,'추가 구매 가능':True},
                                        {'옵션명':'팥붕','재료명':'팥','단위':'g','1개 사용량':20.,'보유량':800.,'단위당 원가(원)':8.,'추가 구매 가능':True},
                                        {'옵션명':'슈붕','재료명':'반죽','단위':'g','1개 사용량':40.,'보유량':3000.,'단위당 원가(원)':3.,'추가 구매 가능':True},
                                        {'옵션명':'슈붕','재료명':'슈크림','단위':'g','1개 사용량':20.,'보유량':800.,'단위당 원가(원)':10.,'추가 구매 가능':True}]
                    else:
                        default_recipe=[{'옵션명':text(variant_rows[0].get('옵션명')) if variant_rows else '',
                                         '재료명':'주재료','단위':'g','1개 사용량':20.,'보유량':2000.,'단위당 원가(원)':10.,'추가 구매 가능':True}]
                    recipes=st.data_editor(pd.DataFrame(default_recipe),num_rows='dynamic',hide_index=True,key=prefix+'materials',
                        column_config={
                            '옵션명':st.column_config.SelectboxColumn(options=[text(v.get('옵션명')) for v in variant_rows],required=True),
                            '재료명':st.column_config.TextColumn(required=True,max_chars=35),
                            '단위':st.column_config.SelectboxColumn(options=['g','ml','개','장','회'],required=True),
                            '1개 사용량':st.column_config.NumberColumn(min_value=.001,max_value=100000.,required=True),
                            '보유량':st.column_config.NumberColumn(min_value=0.,max_value=100000000.,required=True),
                            '단위당 원가(원)':st.column_config.NumberColumn(min_value=0.,max_value=1000000.,required=True),
                            '추가 구매 가능':st.column_config.CheckboxColumn(default=True)})
                    st.caption('같은 재료의 단위·보유량·단가는 모든 행에서 같게 입력하세요. g와 kg를 섞지 마세요.')
                    seen=set()
                    for rr in recipes.to_dict('records'):
                        opt=text(rr.get('옵션명'))
                        targets=[idx for idx,v in enumerate(variant_rows) if text(v.get('옵션명'))==opt]
                        if not targets:
                            errors.append(f'{name}: 재료 표의 옵션명을 실제 옵션명과 맞춰 주세요.')
                            continue
                        j=targets[0]
                        mat_name=text(rr.get('재료명'))
                        unit=text(rr.get('단위'))
                        use=number(rr.get('1개 사용량'),-1)
                        stock=number(rr.get('보유량'),-1)
                        unit_cost=number(rr.get('단위당 원가(원)'),-1)
                        if not mat_name or unit not in ['g','ml','개','장','회'] or use<=0 or stock<0 or unit_cost<0:
                            errors.append(f'{name}: 재료명·단위·사용량·보유량·단가를 확인해 주세요.')
                            continue
                        dup=(j,mat_name)
                        if dup in seen:
                            errors.append(f'{name}: 같은 옵션의 재료가 중복되었습니다.')
                        seen.add(dup)
                        all_recipe_records.append((product_lines[j],dict(name=mat_name,unit=unit,stock=stock,unit_cost=unit_cost,can_buy=bool(rr.get('추가 구매 가능',False))),use))
                    for j,row in enumerate(product_lines):
                        if not any(record[0] is row for record in all_recipe_records):
                            errors.append(f"{row['label']}: 재료를 한 개 이상 입력해 주세요.")
                        row['has_recipe']=True
                        row['extra_cost']=extra
                if not name.strip():
                    errors.append(f'상품 {i+1}: 상품명을 입력해 주세요.')
                lines.extend(product_lines)
        material_index={}
        for row,mat,use in all_recipe_records:
            key=mat['name']
            if key not in material_index:
                material_index[key]=len(materials)
                materials.append(mat)
            idx=material_index[key]
            if materials[idx]!=mat:
                errors.append(f'{key}: 같은 재료의 단위·보유량·단가·추가 구매 여부를 통일해 주세요.')
            row['recipe'][idx]=use
        labels=[r['label'] for r in lines]
        if len(labels)!=len(set(labels)):
            errors.append('상품·옵션 이름이 중복됩니다. 이름을 구분해 주세요.')
        st.divider()
        adjust_price=st.toggle('판매 가격 조정도 제안',value=True)
        st.caption('현재 가격 ±30% 범위의 후보를 비교합니다.' if adjust_price else '입력한 가격을 유지하고 준비 수량만 조정합니다.')
        cfg=dict(visitors=int(visitors),capital=int(capital),fixed=int(fixed),
                 capacity_minutes=float(duration*staff),customer_budget=int(customer_budget),temp=int(temp),
                 meal=bool(meal),rain=bool(rain),competition=int(competition),
                 outside={'낮음':2.4,'보통':1.5,'높음':.8}[intent],
                 elasticity={'낮음':.9,'보통':1.5,'높음':2.1}[sensitivity],adjust_price=bool(adjust_price))
        if fixed>capital:
            errors.append('고정비가 총 준비 예산보다 큽니다. 예산 또는 고정비를 조정해 주세요.')
        for err in list(dict.fromkeys(errors)):
            st.error(err)
        start=st.button('시뮬레이터 시작!',type='primary',width='stretch',disabled=bool(errors))
    return lines,materials,cfg,errors,start


def main():
    st.set_page_config(page_title='학교 축제 시뮬레이터',page_icon='🎪',layout='wide',initial_sidebar_state='expanded')
    style()
    st.title('학교 축제 시뮬레이터')
    st.markdown('<div class="intro">AI 수요 분석으로 설계하는 축제 판매 전략</div>',unsafe_allow_html=True)
    lines,mats,cfg,errors,start=sidebar_inputs()
    sig=signature(lines,mats,cfg)
    if start and not errors:
        with st.spinner('판매 계획과 조정안을 분석하고 있어요.'):
            ctx=prepare(copy.deepcopy(lines),copy.deepcopy(mats),cfg)
            before,after=optimize(ctx)
            st.session_state['report_v2']=(ctx,before,after,sig)
    if 'report_v2' not in st.session_state:
        st.markdown('<div class="empty"><b>판매 계획을 수익 전략으로.</b><p>왼쪽에서 상품과 운영 조건을 입력한 뒤<br>시뮬레이터를 시작하세요.</p></div>',unsafe_allow_html=True)
        st.markdown('<div class="note">가상 데이터 기반 추정 · 실제 매출 보장 아님</div>',unsafe_allow_html=True)
    else:
        ctx,before,after,last_sig=st.session_state['report_v2']
        if last_sig!=sig:
            st.warning('입력이 변경되었습니다. 다시 실행하면 결과에 반영됩니다.')
        render_results(ctx,before,after)


if __name__=='__main__':
    main()
