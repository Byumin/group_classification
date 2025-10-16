import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="그룹 분류 파이프라인", layout="wide")
# 사이드바 메뉴
st.sidebar.title("메타 설정")
st.sidebar.header("1. 파일 업로드")
student_file = st.sidebar.file_uploader("학생 명렬표를 업로드하세요", type=["xlsx"])
uploaded_file = st.sidebar.file_uploader("심리검사 결과 파일을 업로드하세요", type=["xlsx"])

# 파일 업로드 시
if student_file and uploaded_file:
    student_df = pd.read_excel(student_file)
    st.session_state['student_df'] = student_df
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
st.title("🔧 그룹 분류 파이프라인")

# 본문 탭 구성
tabs = st.tabs(["🔍 명렬표 & 검사결과 비교", "🧪 변수 생성", "⚙️ 분류 알고리즘", "🧠 그룹 분류", "📊 분류 후 분포 확인"])

# 학생 명렬표와 검사 결과 데이터프레임 병합 비교 검토 필요
# 병합했을 때 서로 겹치는 프레임과
# 겹치지 않는 프레임이 있을 수 있음 (학생 명렬표에 있는데 검사 결과에 없는 경우 / 학생 명렬표에 없는데 검사 결과에 있는 경우)
# 모두 시각화해서 사용자가 식별할 수 있도록
# [1] 명렬표 & 검사결과 비교
with tabs[0]:
    #st.header("명렬표 & 검사결과 비교")
    if 'student_df' in st.session_state and 'raw_df' in st.session_state:
        student_df = st.session_state['student_df']
        raw_df = st.session_state['raw_df']
        st.subheader("학생 명렬표")
        st.dataframe(student_df.head(10), use_container_width=True)
        st.subheader("검사 결과 데이터프레임")
        st.dataframe(raw_df.head(10), use_container_width=True)

        # 학생 명렬표 프레임에서 병합할 열 만들기
        # 학년(1자리) + 반(2자리) + 번호(2자리) + 성별(1자리) + 이름
        student_df['학년'] = student_df['학년'].astype(str)
        student_df['임시반'] = student_df['임시반'].astype(str).str.zfill(2)
        student_df['임시번호'] = student_df['임시번호'].astype(str).str.zfill(2)
        student_df['성별'] = student_df['성별'].map({'남': '1', '여': '2'}).astype(str)
        student_df['이름'] = student_df['이름'].astype(str)
        student_df['merge_key'] = student_df['학년'] + student_df['임시반'] + student_df['임시번호'] + student_df['성별'] + student_df['이름']

        # 검사 결과 프레임에서 병합할 열 만들기
        # 학년반번호(5자리) + 성별(1자리) + 이름
        raw_df['학년반번호'] = raw_df['학년반번호'].astype(str)
        if raw_df['성별'].dtype == 'O':  # object 타입(문자열)이면 변환
            raw_df['성별'] = raw_df['성별'].map({'남': '1', '여': '2'}).astype(str)
        else:
            raw_df['성별'] = raw_df['성별'].astype(str)
        raw_df['이름'] = raw_df['이름'].astype(str)
        raw_df['merge_key'] = raw_df['학년반번호'] + raw_df['성별'] + raw_df['이름']

        # merge_key 열을 기준으로 병합 후
        # 병합된 데이터프레임 표시
        st.subheader("병합 결과 예상")
        st.dataframe(pd.merge(student_df, raw_df, on='merge_key', how='outer', indicator=True, suffixes=('_명렬표', '_검사결과')).head(10), use_container_width=True)
        # 명렬표에만 있는 행 표시
        st.subheader("명렬표에만 있는 행")
        st.dataframe(student_df[~student_df['merge_key'].isin(raw_df['merge_key'])], use_container_width=True)
        # 검사 결과에만 있는 행 표시
        st.subheader("검사 결과에만 있는 행")
        st.dataframe(raw_df[~raw_df['merge_key'].isin(student_df['merge_key'])], use_container_width=True)

        st.write("병합 예상 결과를 확인 후, 병합을 진행하세요.")
        # 병합 버튼
        if st.button("병합 진행"):
            # 무조건 merge_key로 병합
            merged_df = pd.merge(student_df, raw_df, on='merge_key', how='outer', indicator=True, suffixes=('_명렬표', '_검사결과'))
            st.session_state['merged_df'] = merged_df
        else :
            pass
        # 병합된 데이터프레임 기반으로 결시생, 동명이인(성+이름 동일) 처리
        # 결시생 수, 표시 / 동명이인 수(성+이름 동일), 표시
        if 'merged_df' in st.session_state:
            merged_df = st.session_state['merged_df']
            st.subheader("병합된 데이터프레임")
            st.dataframe(merged_df.head(10), use_container_width=True)
            # 결시생
            num_absent = merged_df[merged_df['_merge'] == 'left_only'].shape[0]
            st.write(f"결시생 수: {num_absent}명")
            st.dataframe(merged_df[merged_df['_merge'] == 'left_only'], use_container_width=True)
            # 동명이인 수(이름 동일)
            dup_names = merged_df[merged_df.duplicated('이름_명렬표', keep=False)]
            st.write(f"동명이인 수 : {dup_names.shape[0]}명")
            st.dataframe(dup_names, use_container_width=True)
            # 확인한 결시생과 동명이인이 맞다면 클릭
            if st.button("결시생, 동명이인 라벨링"):
                st.session_state['raw_df'] = merged_df
                merged_df['결시생'] = merged_df['_merge'].apply(lambda x: 1 if x == 'left_only' else 0)
                merged_df['동명이인'] = merged_df.duplicated('이름_명렬표', keep=False).astype(int)
                merged_df['동명이인_ID'] = (
                    merged_df.groupby('이름_명렬표', sort=False).ngroup()
                )
                merged_df.loc[merged_df['동명이인'] == 0, '동명이인_ID'] = np.nan
                st.session_state['merged_df'] = merged_df
                st.success("결시생, 동명이인 라벨링이 완료되었습니다. 변수 생성을 진행해주세요.")
                st.dataframe(merged_df, use_container_width=True)

        else:
            st.warning("병합을 진행해주세요.")

# [1] 변수 생성 탭
with tabs[1]:

    # 계산 목록 정의
    available_calculations = {
        '합계': 'sum',
        '평균': 'mean',
        '중앙값': 'median',
        '표준편차': 'std',
        '분산': 'var',
        'z-점수': 'z_score',
        '백분위수': 'percentile'
    }

    #st.header("변수 생성")
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
with tabs[2]:
    #st.header("분류 방법 선택")
    st.write("집단을 분류하고자 할때 사용할 방법을 선택할 수 있습니다.")
    try:
        available_continuous_variables = st.session_state['available_continuous_variables']
        available_discrete_variables = st.session_state['available_discrete_variables']
        df = st.session_state['df']

        # 알고리즘 목록
        algorithms = {
            '규칙 기반 그룹화': 'init_group_assign',
            '신경망 그룹화(추후 개발 진행)': 'neural_network_grouping',
            }
        # 알고리즘 선택
        selected_algorithm = st.selectbox(
            "사용할 알고리즘을 선택하세요",
            options=list(algorithms.keys()),
            help="집단 분류에 사용할 알고리즘을 선택하세요."
        )
        st.session_state['selected_algorithm'] = selected_algorithm

        if selected_algorithm == '규칙 기반 그룹화':
            st.write("규칙 기반 그룹화는 데이터를 정렬하여 그룹을 형성하는 방법입니다.")

            # 정렬할 연속형 변수 선택
            selected_sort_variable = {}
            sortable_variable_number = st.number_input(
                "정렬하고자 하는 변수의 개수를 입력하세요",
                min_value=1, max_value=len(available_continuous_variables), value=1,
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

# [3] 집단 분류
with tabs[3]:
    #st.header("집단 분류")
    st.write("집단을 분류하기 전에 관련한 옵션(합반, 분반 등)을 선택해주세요.")
    try:
        # 성별 분류 선택
        sex_classification = st.selectbox(
            "남여 합반/분반을 선택해 주세요.",
            options=["합반", "분반", "남학교", "여학교"],
            help="업로드 파일에 '성별' 컬럼이 있는지 꼭 확인해 주세요."
        )
        st.session_state['sex_classification'] = sex_classification
        try:
            if sex_classification == '분반' and df['성별'].nunique() == 2:
                # 남자 집단 갯수
                male_class_count = st.number_input(
                    "남자 집단의 개수를 입력하세요",
                    min_value=1, max_value=10, value=1,
                    help="남자 집단의 개수를 입력하세요."
                )
                # 여자 집단 갯수
                female_class_count = st.number_input(
                    "여자 집단의 개수를 입력하세요",
                    min_value=1, max_value=10, value=1,
                    help="여자 집단의 개수를 입력하세요."
                )
                st.session_state['male_class_count'] = male_class_count
                st.session_state['female_class_count'] = female_class_count
                st.session_state['group_count'] = male_class_count + female_class_count
            elif sex_classification == '합반' and df['성별'].nunique() == 2:
                group_count = st.number_input(
                    "분류할 집단의 개수를 입력하세요",
                    min_value=2, max_value=10, value=2,
                    help="분류할 집단의 개수를 입력하세요."
                )
                st.session_state['group_count'] = group_count
            elif sex_classification == '남학교' or sex_classification == '여학교':
                group_count = st.number_input(
                    "분류할 집단의 개수를 입력하세요",
                    min_value=2, max_value=10, value=2,
                    help="분류할 집단의 개수를 입력하세요."
                )
                st.session_state['group_count'] = group_count
            else:
                st.error("업로드 된 파일에 성별 컬럼이 없거나, 분반 또는 합반을 선택했지만 성별이 하나만 존재합니다.")
        except Exception as e:
            st.warning(f"성별 분류 설정 중 오류가 발생했습니다: {e}")
    except Exception as e:
        st.warning(f"파일을 업로드 하세요. {e}")

    # 과목기반
    st.subheader("과목 기반 분류 여부")
    subject_based_classification = st.radio(
        "과목 기반 분류를 선택하세요",
        options=["예", "아니오"],
        help="학생 명렬표에 선택 과목에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['subject_based_classification'] = subject_based_classification

    # 결시 학생 처리
    st.subheader("결시생 처리")
    special_student_handling = st.radio(
        "결시생을 그룹별로 균형있게 배정하시겠습니까?",
        options=["예", "아니오"],
        help="학생 명렬표에 결시생에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['special_student_handling'] = special_student_handling

    # 특수 학생 처리
    st.subheader("특수 학생 처리")
    special_student_handling = st.radio(
        "특수 학생을 그룹별로 균형있게 배정하시겠습니까?",
        options=["예", "아니오"],
        help="학생 명렬표에 특수 학생에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['special_student_handling'] = special_student_handling

    # 출신 학교 기반 분류
    st.subheader("출신 학교 기반 분류 여부")
    school_based_classification = st.radio(
        "출신 학교을 고려해 그룹별로 균형있게 배정하시겠습니까?",
        options=["예", "아니오"],
        help="학생 명렬표에 출신 학교에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['school_based_classification'] = school_based_classification

    if st.session_state.get('group_count', 0) > 0:
        full_group_names = []
        for i in range(st.session_state['group_count']):
            group_name = st.text_input(f"집단 {i+1}의 이름을 입력하세요", value=f"Group {i+1}")
            full_group_names.append(group_name)
        st.session_state['full_group_names'] = full_group_names
    else:
        st.warning(f"집단 이름 설정 중 오류가 발생했습니다.")
    
    #! 동명이인은 무조건 다른 그룹으로 배정


# streamlit run c:/Users/USER/group_classification/pipeline_v2.0.py
# streamlit run /Users/mac/insight_/group_classification/pipeline_v2.0.py