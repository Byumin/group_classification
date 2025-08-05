import streamlit as st
import pandas as pd

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
st.set_page_config(page_title="집단 분류 파이프라인", layout="wide")
st.title("🔧 집단 분류 파이프라인")

# 본문 탭 구성
tabs = st.tabs(["🧪 변수 생성", "⚙️ 분류 알고리즘", "📊 분류 전 분포 확인", "🧩 집단 분류 규칙", "🧠 집단 분류", "📊 분류 후 분포 확인"])

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
        if selected_algorithm == '정렬 기반 그룹화':
            st.write("정렬 기반 그룹화는 데이터를 정렬하여 그룹을 형성하는 방법입니다.")
            # 정렬 딕셔너리 초기화
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
        else :
            st.warning("정렬 기반 그룹화 외의 알고리즘은 아직 구현되지 않았습니다.")

    except Exception as e:
        st.warning("변수를 선택하고 데이터프레임을 생성한 후 다시 시도해주세요.")
    
    if st.button("알고리즘 적용"):
        if selected_algorithm == '정렬 기반 그룹화' and len(selected_sort_variable) == sortable_variable_number:
            st.write("알고리즘을 적용합니다...")
            variables_list = [key for key in selected_sort_variable.keys()]
            is_ascending_list = [value for value in selected_sort_variable.values()]
            sorted_df = df.sort_values(by=variables_list, ascending=is_ascending_list)
            st.session_state['sorted_df'] = sorted_df
            st.dataframe(sorted_df.head(10), use_container_width=True)
        elif selected_algorithm != '정렬 기반 그룹화':
            st.warning("정렬 기반 그룹화 외의 알고리즘은 아직 구현되지 않았습니다.")
        else:
            st.warning("변수를 선택해주세요.")
    else:
        pass
        
with tabs[2]:
    st.header("📊 분류 전 분포 확인")
    st.write("여기에 다른 내용을 작성할 수 있습니다.")

with tabs[3]:
    st.header("🧩 집단 분류 규칙")
    st.write("여기에 다른 내용을 작성할 수 있습니다.")

with tabs[4]:
    st.header("🧩 집단 분류 규칙")
    st.write("여기에 다른 내용을 작성할 수 있습니다.")

with tabs[5]:
    st.header("🧠 집단 분류")
    st.write("여기에 다른 내용을 작성할 수 있습니다.")