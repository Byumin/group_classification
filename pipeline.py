import importlib
import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go # kde 추정
from scipy.stats import gaussian_kde # kde 추정
import numpy as np # kde 추정

import altair as alt

st.set_page_config(page_title="집단 분류 파이프라인", layout="wide")
# 사이드바 메뉴
st.sidebar.title("메타 설정")
st.sidebar.header("1. 파일 업로드")
uploaded_file = st.sidebar.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])

# 파일 업로드 시
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state['raw_df'] = df
    st.session_state['cols'] = df.columns.tolist()
    st.sidebar.success("파일이 성공적으로 업로드되었습니다.")
else:
    st.sidebar.warning("엑셀 파일을 업로드해주세요.")

# 변수 선택
continuous_variable = st.sidebar.multiselect(
    "연속형 변수를 선택하세요",
    options=st.session_state.get('cols', []),
    help="시험 점수와 같은 연속형 변수를 선택하세요."
)
if continuous_variable:
    st.session_state['continuous_variable'] = continuous_variable
    st.sidebar.success("변수 선택이 완료되었습니다.")
else:
    st.sidebar.warning("변수를 선택해주세요.")
discrete_variable = st.sidebar.multiselect(
    "범주형 변수를 선택하세요",
    options=st.session_state.get('cols', []),
    help="성별과 같은 범주형 변수를 선택하세요."
)
if discrete_variable:
    st.session_state['discrete_variable'] = discrete_variable
    st.sidebar.success("변수 선택이 완료되었습니다.")
else:
    st.sidebar.warning("변수를 선택해주세요.")

# =============== 본문 영역 ===============
st.title("🔧 집단 분류 파이프라인")

# 본문 탭 구성
tabs = st.tabs(["🧪 변수 생성", "⚙️ 분류 알고리즘", "📊 분류 전 분포 확인", "🧠 집단 분류", "📊 분류 후 분포 확인"])

# [1] 변수 생성 탭
with tabs[0]:

    # 계산 목록 정의
    available_calculations = {
        '합계': 'sum',
        '평균': 'mean',
        '중앙값': 'median',
        '최대값': 'max',
        '최소값': 'min',
        '표준편차': 'std',
        '분산': 'var',
        'z-점수': 'z_score',
        '백분위수': 'percentile'
    }

    st.header("변수 생성")
    st.write("메타 설정에서 선택한 변수를 활용해 변수를 생성할 수 있습니다.")
    # 생성할 변수 갯수
    num_variables = st.number_input("생성할 변수의 개수를 입력하세요", min_value=1, max_value=10, value=1)
    # 변수 생성 입력 필드
    for i in range(num_variables):
        st.subheader(f"{i+1}번째 변수 생성")
        variable_name = st.text_input(f"생성할 변수 이름", key=f"var_name_{i+1}")
        selected_continuous_variable = st.multiselect(
            f"사용할 변수 선택",
            options=st.session_state.get('continuous_variable', []),key=f"var_select_{i+1}",
            help="사용할 변수를 선택하세요."
        )
        variable_formula = st.selectbox(f"변수 계산식", options=list(available_calculations.keys()), key=f"var_formula_{i+1}")
        if variable_name and variable_formula:
            st.session_state[f'var_{i+1}'] = {
                'name': variable_name,
                'variables': selected_continuous_variable,
                'formula': variable_formula
            }
        else:
            st.warning("모든 필드를 입력해주세요.")
    # 변수 생성 버튼
    if st.button("변수 생성"):
        if 'raw_df' in st.session_state:
            df = st.session_state['raw_df']
            for i in range(num_variables):
                var_info = st.session_state.get(f'var_{i+1}', {})
                var_name = var_info['name']
                variables = var_info['variables']
                formula = available_calculations.get(var_info['formula'], None)
                try:
                    if formula == 'sum':
                        df[var_name] = df[variables].sum(axis=1)
                    elif formula == 'mean':
                        df[var_name] = df[variables].mean(axis=1)
                    elif formula == 'median':
                        df[var_name] = df[variables].median(axis=1)
                    elif formula == 'max':
                        df[var_name] = df[variables].max(axis=1)
                    elif formula == 'min':
                        df[var_name] = df[variables].min(axis=1)
                    elif formula == 'std':
                        df[var_name] = df[variables].std(axis=1)
                    elif formula == 'var':
                        df[var_name] = df[variables].var(axis=1)
                    elif len(variables) == 1 and formula == 'z_score':
                        df[var_name] = (df[variables] - df[variables].mean()) / df[variables].std()
                    # ! 백분위는 후에 별도로 처리
                    else:
                        st.warning("변수 처리에 오류가 발생했습니다.")
                except Exception as e:
                    st.warning(f"변수 처리에 오류가 발생했습니다: {e}")
        else:
            st.error("업로드된 데이터프레임이 없습니다. 파일을 업로드해주세요.")
        # 데이터프레임 업데이트
        st.session_state['df'] = df
        # 연속형 변수 업데이트
        available_continuous_variables = st.session_state['continuous_variable'] + [st.session_state[f'var_{i+1}']['name'] for i in range(num_variables)]
        st.session_state['available_continuous_variables'] = available_continuous_variables
        # 범주형 변수 업데이트
        available_discrete_variables = st.session_state['discrete_variable']
        st.session_state['available_discrete_variables'] = available_discrete_variables
        # 데이터프레임 표시
        st.dataframe(df.head(10), use_container_width=True)
    else:
        pass

# [2] 분류 알고리즘
with tabs[1]:
    st.header("⚙️ 분류 알고리즘")
    st.write("집단을 분류하고자 할때 사용할 알고리즘을 선택할 수 있습니다.")
    try:
        available_continuous_variables = st.session_state['available_continuous_variables']
        available_discrete_variables = st.session_state['available_discrete_variables']
        df = st.session_state['df']

        # 알고리즘 목록
        algorithms = {
            '정렬 기반 그룹화': 'sort_based',
            'K-평균 군집화': 'kmeans',
            'DBSCAN 군집화': 'dbscan',
            '계층적 군집화': 'hierarchical',
            '랜덤 포레스트': 'random_forest',
            'XGBoost': 'xgboost',
            'LightGBM': 'lightgbm'
            }
        # 알고리즘 선택
        selected_algorithm = st.selectbox(
            "사용할 알고리즘을 선택하세요",
            options=list(algorithms.keys()),
            help="집단 분류에 사용할 알고리즘을 선택하세요."
        )
        st.session_state['selected_algorithm'] = selected_algorithm
        if selected_algorithm == '정렬 기반 그룹화':
            st.write("정렬 기반 그룹화는 데이터를 정렬하여 그룹을 형성하는 방법입니다.")

            # 정렬할 연속형 변수 선택
            selected_sort_variable = {}
            sortable_variable_number = st.number_input(
                "정렬하고자 하는 변수의 개수를 입력하세요",
                min_value=1, max_value=10, value=1,
                help="정렬하고자 하는 변수의 개수를 입력하세요."
            )
            for n in range(sortable_variable_number):
                st.subheader(f"{n+1}번째 정렬 변수")
                # 정렬 변수 선택
                sort_variable = st.selectbox(
                    f"정렬 변수 선택",
                    options=st.session_state.get('available_continuous_variables', []),
                    key=f'sort_var_{n+1}',
                    help="정렬할 변수를 선택하세요."
                )
                # 오름차순 정렬 여부 선택
                is_ascending = st.checkbox(
                    f"오름차순 정렬 (체크: 오름차순 / 해제: 내림차순)",
                    value=True,
                    key=f'sort_asc_{n+1}',
                    help="정렬 방향을 선택하세요."
                )

                if sort_variable:
                    # 선택된 정렬 변수를 딕셔너리에 저장
                    selected_sort_variable[sort_variable] = is_ascending
                else:
                    st.warning(f"{n+1}번째 정렬 변수를 선택해주세요.")
            print(f"Selected sort variable: {selected_sort_variable}")
            st.session_state['selected_sort_variable_dict'] = selected_sort_variable

            # 그룹별 균형을 맞춰야하는 범주형 변수 파라미터 설정
            st.subheader("그룹별 균형을 맞춰야하는 범주형 변수")
            selected_discrete_variable = st.multiselect(
                "범주형 변수를 선택하세요",
                options=available_discrete_variables,
                help="그룹별 균형을 맞추고자 하는 범주형 변수를 선택하세요."
                )
            # 범주형 변수 선택이 없을 수 있음.
            st.session_state['selected_discrete_variable'] = selected_discrete_variable
            print(f"Selected discrete variable: {selected_discrete_variable}")

        else :
            st.warning("정렬 기반 그룹화 외의 알고리즘은 아직 구현되지 않았습니다.")

    except Exception as e:
        st.warning("변수를 선택하고 데이터프레임을 생성한 후 다시 시도해주세요.")

# [3] 분류 전 분포 확인
with tabs[2]:
    st.header("📊 분류 전 분포 확인")
    st.write("선택한 변수의 분포를 확인할 수 있습니다.")
    try:
        # 세션 상태에서 데이터프레임과 변수 가져오기
        df = st.session_state['df']
        selected_sort_variable_dict = st.session_state['selected_sort_variable_dict'] # 정렬 연속형 변수 딕셔너리
        discrete_variable = st.session_state['selected_discrete_variable'] # 범주형 변수
        print(df)
        print(f"Selected sort variable dict: {selected_sort_variable_dict}")
        print(f"Selected discrete variable: {discrete_variable}")

        if all(k in st.session_state for k in ['df', 'selected_sort_variable_dict', 'selected_discrete_variable']):
            # ============================================================
            # 연속형 변수와 범주형 변수의 분포를 시각화
            if selected_sort_variable_dict:
                st.subheader("연속형 변수 분포")
                # 연속형 변수 설정 기준으로 df 정렬
                df_sorted = df.sort_values(by=list(selected_sort_variable_dict.keys()), ascending=list(selected_sort_variable_dict.values()))

                # 연속형 변수의 시각화 블럭
                for var in selected_sort_variable_dict.keys():
                    st.write(f"🔹 `{var}` 의 분포 (히스토그램 + 밀도곡선)")
                    fig = px.histogram(
                        df_sorted, x=var,
                        marginal="box",  # box, violin, rug 가능
                        opacity=0.7, # 투명도 설정
                        histnorm=None
                    )
                    # 밀도곡선 추가
                    data = df_sorted[var].dropna()
                    kde = gaussian_kde(data)
                    x_vals = np.linspace(data.min(), data.max(), 200)
                    y_vals = kde(x_vals)
                    # 실제 bin 개수 추정
                    counts, bins = np.histogram(data, bins='auto')
                    bin_width = bins[1] - bins[0]
                    # KDE를 count 스케일로 보정
                    y_scaled = y_vals * len(data) * bin_width
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_scaled,
                            mode="lines",
                            line=dict(color="lightblue", width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0,0,1,0.2)',
                            showlegend=False
                        )
                    )
                    fig.update_layout(
                        bargap=0,
                        title=f"{var}의 분포 (Count + KDE)",
                        xaxis_title=var,
                        yaxis_title="빈도 (count)",
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)
            # ============================================================
            # 범주형 변수의 시각화 블럭
            if discrete_variable:
                st.subheader("범주형 변수 분포")
                color_sequence = px.colors.qualitative.Set2 # 범주형 변수 색상 목록
                for var in discrete_variable:
                    st.write(f"🔹 `{var}` 의 분포 (막대그래프)")
                    freq_df = df[var].value_counts().reset_index()
                    freq_df.columns = [var, 'count']
                    colors = color_sequence[:len(freq_df)]
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=freq_df[var],
                            y=freq_df['count'],
                            marker_color=colors,
                            width=[0.4] * len(freq_df), # 막대 너비 설정
                        )
                    )
                    fig.update_layout(
                        title=f"{var}의 분포",
                        xaxis_title=var,
                        yaxis_title="빈도 (count)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("데이터프레임과 변수가 설정되지 않았습니다. 먼저 변수를 선택하고 데이터프레임을 생성해주세요.")
    except Exception as e:
        st.warning(f"분포 시각화 중 오류가 발생했습니다: {e}")

# [4] 집단 분류 규칙
with tabs[3]:
    st.header("🧠 집단 분류")
    st.write("집단을 분류하기 위해 필요한 규칙을 설정할 수 있습니다.")
    # 집단 수 설정
    group_count = st.number_input(
        "분류할 집단의 개수를 입력하세요",
        min_value=2, max_value=10, value=2,
        help="분류할 집단의 개수를 입력하세요."
    )
    st.session_state['group_count'] = group_count
    # 정렬기반인 경우 round-robin 방식인지 serpentine 방식인지 선택
    sortable_method = st.selectbox(
        "분배 방식을 선택해 주세요.",
        options=["round-robin", "serpentine"],
        help="round-robin : 1→2→3→4→1→2→3→4 순으로 분류되며, serpentine : 1→2→3→4→4→3→2→1 순으로 분류됩니다."
    )
    st.session_state['sortable_method'] = sortable_method
    # 분류 한 후 집단명 설정
    group_names = []
    for i in range(group_count):
        group_name = st.text_input(f"집단 {i+1}의 이름을 입력하세요", value=f"Group {i+1}")
        group_names.append(group_name)
    st.session_state['group_names'] = group_names

    # 알고리즘에 따라 파라미터가 다양해지기 때문에 context에 다 넣어서 처리
    context = {
        'df': st.session_state.get('df', None),
        'selected_sort_variable_dict': st.session_state.get('selected_sort_variable_dict', {}),
        'selected_discrete_variable': st.session_state.get('selected_discrete_variable', []),
        'selected_algorithm': st.session_state.get('selected_algorithm', ''),
        'group_count': st.session_state.get('group_count', 0),
        'sortable_method': st.session_state.get('sortable_method', ''),
        'group_names': st.session_state.get('group_names', [])
    }

    # 집단 분류 버튼
    if st.button("집단 분류 시작"):
        try:
            if all(k in st.session_state for k in ['df', 'selected_sort_variable_dict', 'selected_discrete_variable', 'selected_algorithm', 'group_count', 'sortable_method', 'group_names']):
                df = st.session_state['df']
                selected_sort_variable_dict = st.session_state['selected_sort_variable_dict']
                selected_discrete_variable = st.session_state['selected_discrete_variable']
                selected_algorithm = st.session_state['selected_algorithm']
                group_count = st.session_state['group_count']
                sortable_method = st.session_state['sortable_method']
                group_names = st.session_state['group_names']

                module_path = algorithms[selected_algorithm]

                module = importlib.import_module(module_path)
                result_grouping_df = module.run(context)
                st.session_state['result_grouping_df'] = result_grouping_df
                print(f"Result grouping df: {result_grouping_df}")

            else:
                st.error("집단 분류를 위한 모든 파라미터가 설정되지 않았습니다. 다시 확인해주세요.")
        
        except Exception as e:
            st.error(f"집단 분류 중 오류가 발생했습니다: {e}")

with tabs[4]:
    st.header("📊 분류 후 분포 확인")
    st.write("집단 분류 후 각 집단의 분포를 확인할 수 있습니다.")
    result_grouping_df = st.session_state.get('result_grouping_df', None)
    selected_sort_variable_dict = st.session_state.get('selected_sort_variable_dict', {})
    try:
        # 연속형 변수의 시각화 블럭 (그룹별)
        for var in selected_sort_variable_dict.keys():
            st.write(f"🔹 `{var}` 의 분포 (히스토그램 + 밀도곡선)")
            fig = px.histogram(
                result_grouping_df, x=var,
                color='group',  # 그룹별 색상 구분
                barmode='overlay',  # 겹쳐서 표시
                marginal="box",  # box, violin, rug 가능
                opacity=0.7, # 투명도 설정
                histnorm=None
            )

            group_list = result_grouping_df['group'].unique()
            colors = px.colors.qualitative.Plotly  # 그룹별 KDE 곡선 색상 설정

            for i, group in enumerate(group_list):
                group_data = result_grouping_df[result_grouping_df['group'] == group][var].dropna()

                if len(group_data) < 2:
                    continue  # KDE 계산 불가능한 경우 스킵

                kde = gaussian_kde(group_data)
                x_vals = np.linspace(group_data.min(), group_data.max(), 200)
                y_vals = kde(x_vals)

                # bin-width 기반 스케일 보정
                counts, bins = np.histogram(group_data, bins='auto')
                bin_width = bins[1] - bins[0]
                y_scaled = y_vals * len(group_data) * bin_width

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_scaled,
                        mode='lines',
                        name=f'{group} KDE',
                        line=dict(color=colors[i % len(colors)], width=2),
                        opacity=0.7,
                        showlegend=False  # KDE는 범례에서 제외하고 싶을 경우
                    )
                )

            fig.update_layout(
                bargap=0,
                title=f"{var}의 그룹별 분포 (Count + KDE)",
                xaxis_title=var,
                yaxis_title="빈도 (count)",
                height=800
            )

            st.plotly_chart(fig, use_container_width=True)
        
        # 범주형 변수가 있는 경우
        # 범주형 변수의 시각화 블럭 (그룹별)
        discrete_variable = st.session_state.get('selected_discrete_variable', [])
        if discrete_variable:
            for var in discrete_variable:
                st.write(f"🔹 `{var}` 의 분포 (막대그래프)")
                freq_df = result_grouping_df.groupby(['group', var]).size().reset_index(name='count')
                fig = px.bar(
                    freq_df, x=var, y='count',
                    color='group', barmode='group',
                    height=500,
                    title=f"{var}의 그룹별 분포"
                )
                fig.update_layout(
                    xaxis_title=var,
                    yaxis_title="빈도 (count)"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            pass

    except Exception as e:
        st.error(f"분포 시각화 중 오류가 발생했습니다: {e}")