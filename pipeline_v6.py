import streamlit as st
import pandas as pd
import numpy as np # xlsxwriter 설치 필요 (다른 환경에서)
import traceback
import io

st.set_page_config(page_title="그룹 분류 파이프라인", layout="wide")
# 사이드바 메뉴
st.sidebar.title("업로드 및 설정")
st.sidebar.header("1. 기존 반편성 검사 선택")
options = ['A Type-능력', 'B Type-인성', 'C Type-학습', 'A+B Type', 'B+C Type', 'A+C Type', 'Compact Type', 'Custom Type']
existing_type = st.sidebar.selectbox("기존 반편성 타입 검사를 선택하세요", options=options, disabled=False, help="기존 타입 검사를 선택하면 연속형 변수와 범주형 변수가 자동으로 선택됩니다.", key="existing_type_selectbox")
st.session_state['existing_type'] = existing_type

# 다중 컬럼 구조를 지닌 검사가 존재
# 업로드된 검사 결과가 B타입 또는 Compact 타입인 경우 다중 컬럼 구조 해제
def flatten_multiindex_columns(cols): # 다중 컬럼 구조 해제 함수
    flatten_cols = []
    for col_tuple in cols:
        temp_col = []
        for element in col_tuple:
            if 'Unnamed' not in str(element):
                temp_col.append(str(element))
            else:
                pass
        if len(temp_col) >= 2:
            temp_col = "_".join(map(str, temp_col))
            flatten_cols.append(temp_col)
        else:
            flatten_cols.append(temp_col[0])
    dup_count = {}
    final_cols = []
    for col in flatten_cols:
        if col not in dup_count:
            dup_count[col] = 0
            final_cols.append(col)
        else:
            dup_count[col] += 1
            new_name = f"{col}_{dup_count[col]}"
            final_cols.append(new_name)
    return final_cols
## 기존 반편성 검사 선택 시와 커스텀 선택 시를 분기처리하여 통제
if existing_type in ['A Type-능력', 'B Type-인성', 'C Type-학습', 'A+B Type', 'B+C Type', 'A+C Type', 'Compact Type']:
    st.sidebar.info("기존 반편성 검사 타입이 선택되어 있습니다. 연속형 변수와 범주형 변수가 자동으로 선택됩니다.")
    # 명렬표, 검사 결과 업로드
    st.sidebar.header("2. 파일 업로드")
    student_file = st.sidebar.file_uploader("학생 명렬표를 업로드하세요", type=["xlsx"], key="student_file_uploader")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        uploaded_file_1 = st.file_uploader("검사 결과 파일 업로드", type=["xlsx"], key="uploaded_file_1_uploader")
    with col2:
        uploaded_file_2 = st.file_uploader("", type=["xlsx"], key="uploaded_file_2_uploader")
    # B타입과 Compact 타입의 경우 다중 컬럼 구조를 지니고 있어 전처리가 필요함
    # 각 타입에만 있는 컬럼명 정의
    b_type_essential_cols = ['대인관계 힘듬', '매우 불행', '만성적 학업부진', '자긍심 낮음', '사회적 고립', '심리적 충격', '폭력 경향성', '반사회적 성향', '매우 심한 우울증', 'ADHD 가능성']
    compact_type_essential_cols = ['기초학습역량', '타당도 지표', '외향성', '정서적 민감성']
    # 두 검사 결과가 모두 업로드된 경우
    if student_file and uploaded_file_1 and uploaded_file_2:
        student_df = pd.read_excel(student_file)
        st.session_state['student_df'] = student_df
        raw_df_1 = pd.read_excel(uploaded_file_1)
        raw_df_2 = pd.read_excel(uploaded_file_2)
        if set(b_type_essential_cols) & set(raw_df_1.columns): # raw_df_1가 B타입인 경우
            raw_df_1 = pd.read_excel(uploaded_file_1, header=[0,1])
            raw_df_1.columns = flatten_multiindex_columns(raw_df_1.columns)
            rename_map = {}
            for col in raw_df_1.columns:
                if '성별' in col and ('남' in col or '여' in col):
                    rename_map[col] = '성별'
            if rename_map:
                raw_df_1.rename(columns=rename_map, inplace=True)
        elif set(b_type_essential_cols) & set(raw_df_2.columns): # raw_df_2가 B타입인 경우
            raw_df_2 = pd.read_excel(uploaded_file_2, header=[0,1])
            raw_df_2.columns = flatten_multiindex_columns(raw_df_2.columns)
            rename_map = {}
            for col in raw_df_2.columns:
                if '성별' in col and ('남' in col or '여' in col):
                    rename_map[col] = '성별'
            if rename_map:
                raw_df_2.rename(columns=rename_map, inplace=True)
        elif set(compact_type_essential_cols) & set(raw_df_1.columns): # raw_df_1가 Compact 타입인 경우
            raw_df_1 = pd.read_excel(uploaded_file_1, header=[0,1,2])
            raw_df_1.columns = flatten_multiindex_columns(raw_df_1.columns)
            rename_map = {}
            for col in raw_df_1.columns:
                if '성별' in col and ('남' in col or '여' in col):
                    rename_map[col] = '성별'
                if '학년' in col and '반' in col and '번호' in col:
                    rename_map[col] = '학년반번호'
            if rename_map:
                raw_df_1.rename(columns=rename_map, inplace=True)
        elif set(compact_type_essential_cols) & set(raw_df_2.columns): # raw_df_2가 Compact 타입인 경우
            raw_df_2 = pd.read_excel(uploaded_file_2, header=[0,1,2])
            raw_df_2.columns = flatten_multiindex_columns(raw_df_2.columns)
            rename_map = {}
            for col in raw_df_2.columns:
                if '성별' in col and ('남' in col or '여' in col):
                    rename_map[col] = '성별'
                if '학년' in col and '반' in col and '번호' in col:
                    rename_map[col] = '학년반번호'
            if rename_map:
                raw_df_2.rename(columns=rename_map, inplace=True)
        else:
            pass
        # b_type와 compact_type 이외의 타입인 경우 컬럼명 통일 작업
        rename_map = {}
        for col in raw_df_1.columns:
            if '성별' in col and ('남' in col or '여' in col):
                rename_map[col] = '성별'
            if '학년' in col and '반' in col and '번호' in col:
                rename_map[col] = '학년반번호'
        if rename_map:
            raw_df_1.rename(columns=rename_map, inplace=True)
        rename_map = {}
        for col in raw_df_2.columns:
            if '성별' in col and ('남' in col or '여' in col):
                rename_map[col] = '성별'
            if '학년' in col and '반' in col and '번호' in col:
                rename_map[col] = '학년반번호'
        if rename_map:
            raw_df_2.rename(columns=rename_map, inplace=True)
        st.session_state['raw_df_1'] = raw_df_1
        st.session_state['raw_df_2'] = raw_df_2
        print(raw_df_2.columns)
        raw_df_1.to_excel("raw_df_1_test.xlsx", index=False)
        raw_df_2.to_excel("raw_df_2_test.xlsx", index=False)
        # 두 검사 결과 데이터프레임 병합
        raw_df_1['merge_key'] = raw_df_1['학년반번호'].astype(str) + raw_df_1['성별'].astype(str) + raw_df_1['이름'].astype(str)
        raw_df_2['merge_key'] = raw_df_2['학년반번호'].astype(str) + raw_df_2['성별'].astype(str) + raw_df_2['이름'].astype(str)
        raw_df = pd.merge(raw_df_1, raw_df_2, on='merge_key', how='outer', indicator=True, suffixes=('_검사1', '_검사2'))
        raw_df.to_excel("raw_df_merged_test.xlsx", index=False)
        if raw_df[raw_df['_merge']=='left_only'].shape[0] > 0:
            st.sidebar.warning(f"검사1에만 있는 학생이 {raw_df[raw_df['_merge']=='left_only'].shape[0]}명 있습니다. 확인해주세요.")
        elif raw_df[raw_df['_merge']=='right_only'].shape[0] > 0:
            st.sidebar.warning(f"검사2에만 있는 학생이 {raw_df[raw_df['_merge']=='right_only'].shape[0]}명 있습니다. 확인해주세요.")
        else :
            # 학교명, 반학년번호, 성별, 이름 컬럼이 중복으로 존재하기 때문에 하나로 통일
            st.sidebar.success("두 검사 결과가 모두 일치합니다.")
            base_cols = ['학교명', '학년반번호', '성별', '이름']
            for col in base_cols:
                col1 = f"{col}_검사1"
                col2 = f"{col}_검사2"
                if col1 in raw_df.columns and col2 in raw_df.columns:
                    raw_df[col] = raw_df[col1].combine_first(raw_df[col2])
                    raw_df.drop(columns=[col1, col2], inplace=True)
            # 통합한 열 앞으로 순서 재배치
            cols = raw_df.columns.tolist()
            front_cols = [col for col in base_cols if col in cols]
            remaining_cols = [col for col in cols if col not in front_cols]
            raw_df = raw_df[front_cols + remaining_cols]
        raw_df.drop(columns=['_merge'], inplace=True)
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
        st.sidebar.success("파일이 성공적으로 업로드되었습니다.")
    elif student_file and uploaded_file_1 and not uploaded_file_2: # 검사 결과 파일 1만 업로드된 경우
        student_df = pd.read_excel(student_file)
        st.session_state['student_df'] = student_df
        raw_df = pd.read_excel(uploaded_file_1)
        if set(b_type_essential_cols) & set(raw_df.columns): # raw_df가 B타입인 경우
            raw_df = pd.read_excel(uploaded_file_1, header=[0,1])
            raw_df.columns = flatten_multiindex_columns(raw_df.columns)
            rename_map = {}
            for col in raw_df.columns:
                if '성별' in col and ('남' in col or '여' in col):
                    rename_map[col] = '성별'
            if rename_map:
                raw_df.rename(columns=rename_map, inplace=True)
        elif set(compact_type_essential_cols) & set(raw_df.columns): # raw_df가 Compact 타입인 경우
            raw_df = pd.read_excel(uploaded_file_1, header=[0,1,2])
            raw_df.columns = flatten_multiindex_columns(raw_df.columns)
            rename_map = {}
            for col in raw_df.columns:
                if '성별' in col and ('남' in col or '여' in col):
                    rename_map[col] = '성별'
                if '학년' in col and '반' in col and '번호' in col:
                    rename_map[col] = '학년반번호'
            if rename_map:
                raw_df.rename(columns=rename_map, inplace=True)
        else:
            pass
        rename_map = {}
        for col in raw_df.columns:
            if '성별' in col and ('남' in col or '여' in col):
                rename_map[col] = '성별'
            if '학년' in col and '반' in col and '번호' in col:
                rename_map[col] = '학년반번호'
        if rename_map:
            raw_df.rename(columns=rename_map, inplace=True)
        raw_df['merge_key'] = raw_df['학년반번호'].astype(str) + raw_df['성별'].astype(str) + raw_df['이름'].astype(str)
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
        st.sidebar.success("파일이 성공적으로 업로드되었습니다.")
    elif student_file and not uploaded_file_1 and uploaded_file_2: # 검사 결과 파일 2만 업로드된 경우
        student_df = pd.read_excel(student_file)
        st.session_state['student_df'] = student_df
        raw_df = pd.read_excel(uploaded_file_2)
        if set(b_type_essential_cols) & set(raw_df.columns): # raw_df가 B타입인 경우
            raw_df = pd.read_excel(uploaded_file_2, header=[0,1])
            raw_df.columns = flatten_multiindex_columns(raw_df.columns)
            rename_map = {}
            for col in raw_df.columns:
                if '성별' in col and ('남' in col or '여' in col):
                    rename_map[col] = '성별'
            if rename_map:
                raw_df.rename(columns=rename_map, inplace=True)
        elif set(compact_type_essential_cols) & set(raw_df.columns): # raw_df가 Compact 타입인 경우
            raw_df = pd.read_excel(uploaded_file_2, header=[0,1,2])
            raw_df.columns = flatten_multiindex_columns(raw_df.columns)
            rename_map = {}
            for col in raw_df.columns:
                if '성별' in col and ('남' in col or '여' in col):
                    rename_map[col] = '성별'
                if '학년' in col and '반' in col and '번호' in col:
                    rename_map[col] = '학년반번호'
            if rename_map:
                raw_df.rename(columns=rename_map, inplace=True)
        rename_map = {}
        for col in raw_df.columns:
            if '성별' in col and ('남' in col or '여' in col):
                rename_map[col] = '성별'
            if '학년' in col and '반' in col and '번호' in col:
                rename_map[col] = '학년반번호'
        if rename_map:
            raw_df.rename(columns=rename_map, inplace=True)
        raw_df['merge_key'] = raw_df['학년반번호'].astype(str) + raw_df['성별'].astype(str) + raw_df['이름'].astype(str)
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
        st.sidebar.success("파일이 성공적으로 업로드되었습니다.")
    else:
        st.sidebar.warning("엑셀 파일을 업로드해주세요.")
elif existing_type == 'Custom Type': # 커스텀 타입 선택 시
    st.sidebar.info("커스텀 타입이 선택되어 있습니다. 연속형 변수와 범주형 변수를 직접 선택해주세요.")
    st.sidebar.header("2. 파일 업로드")
    # 명렬표, 검사 결과 업로드
    student_file = st.sidebar.file_uploader("학생 명렬표를 업로드하세요", type=["xlsx"], key="student_file_uploader")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        uploaded_file_1 = st.file_uploader("검사 결과 파일 업로드", type=["xlsx"], key="uploaded_file_1_uploader")
    with col2:
        uploaded_file_2 = st.file_uploader("", type=["xlsx"], key="uploaded_file_2_uploader")
    # 두 검사 결과가 모두 업로드된 경우
    if student_file and uploaded_file_1 and uploaded_file_2:
        student_df = pd.read_excel(student_file)
        st.session_state['student_df'] = student_df
        raw_df_1 = pd.read_excel(uploaded_file_1)
        raw_df_2 = pd.read_excel(uploaded_file_2)
        raw_df_1['merge_key'] = raw_df_1['학년반번호'].astype(str) + raw_df_1['성별'].astype(str) + raw_df_1['이름'].astype(str)
        raw_df_2['merge_key'] = raw_df_2['학년반번호'].astype(str) + raw_df_2['성별'].astype(str) + raw_df_2['이름'].astype(str)
        raw_df_1.to_excel("raw_df_1_test.xlsx", index=False)
        raw_df_2.to_excel("raw_df_2_test.xlsx", index=False)
        st.session_state['raw_df_1'] = raw_df_1
        st.session_state['raw_df_2'] = raw_df_2
        # 두 검사 결과 데이터프레임 병합
        raw_df = pd.merge(raw_df_1, raw_df_2, on='merge_key', how='outer', indicator=True, suffixes=('_검사1', '_검사2'))
        if raw_df[raw_df['_merge']=='left_only'].shape[0] > 0:
            st.sidebar.warning(f"검사 간 학생 명단이 일치하지 않습니다. 확인해주세요.")
        elif raw_df[raw_df['_merge']=='right_only'].shape[0] > 0:
            st.sidebar.warning(f"검사 간 학생 명단이 일치하지 않습니다. 확인해주세요.")
        else :
            st.sidebar.success("두 검사 결과가 모두 일치합니다.")
            base_cols = ['학교명', '학년반번호', '성별', '이름']
            for col in base_cols:
                col1 = f"{col}_검사1"
                col2 = f"{col}_검사2"
                if col1 in raw_df.columns and col2 in raw_df.columns:
                    raw_df[col] = raw_df[col1].combine_first(raw_df[col2])
                    raw_df.drop(columns=[col1, col2], inplace=True)
            # 통합한 열 앞으로 순서 재배치
            cols = raw_df.columns.tolist()
            front_cols = [col for col in base_cols if col in cols]
            remaining_cols = [col for col in cols if col not in front_cols]
            raw_df = raw_df[front_cols + remaining_cols]
        raw_df.drop(columns=['_merge'], inplace=True)
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
        st.sidebar.success("파일이 성공적으로 업로드되었습니다.")
    elif student_file and uploaded_file_1 and not uploaded_file_2: # 검사 결과 파일 1만 업로드된 경우
        student_df = pd.read_excel(student_file)
        st.session_state['student_df'] = student_df
        raw_df = pd.read_excel(uploaded_file_1)
        raw_df['merge_key'] = raw_df['학년반번호'].astype(str) + raw_df['성별'].astype(str) + raw_df['이름'].astype(str)
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
        st.sidebar.success("파일이 성공적으로 업로드되었습니다.")
    elif student_file and not uploaded_file_1 and uploaded_file_2: # 검사 결과 파일 2만 업로드된 경우
        student_df = pd.read_excel(student_file)
        st.session_state['student_df'] = student_df
        raw_df = pd.read_excel(uploaded_file_2)
        raw_df['merge_key'] = raw_df['학년반번호'].astype(str) + raw_df['성별'].astype(str) + raw_df['이름'].astype(str)
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
        st.sidebar.success("파일이 성공적으로 업로드되었습니다.")
    else:
        st.sidebar.warning("엑셀 파일을 업로드해주세요.")
else :
    pass

st.sidebar.header("3. 변수 선택")
# 기존 반편성 검사 선택 시 연속형 변수 및 범주형 변수 강제 선택
if 'prev_existing_type' not in st.session_state:
    st.session_state['prev_existing_type'] = existing_type
if st.session_state['prev_existing_type'] != existing_type:
    print(f'{existing_type} existing type changed, rerunning...')
    st.session_state['prev_existing_type'] = existing_type
try:
    if existing_type == 'A Type-능력': # 연속형만
        continuous_variable_default = ['종합지수']
        discrete_variable_default = []
    elif existing_type == 'B Type-인성': # 연속형 + 이산형
        print('b type selected')
        continuous_variable_default = ['5요인_외향성', '5요인_신경증']
        discrete_variable_default = ['상담필요']
        counseling_col = ['대인관계 힘듬', '매우 불행', '만성적 학업부진', '자긍심 낮음', '사회적 고립', '심리적 충격', '폭력 경향성', '반사회적 성향', '매우 심한 우울증', 'ADHD 가능성']
        st.session_state['continuous_variable_default'] = continuous_variable_default
        st.session_state['discrete_variable_default'] = discrete_variable_default
        raw_df['상담필요'] = np.where(raw_df[counseling_col].apply(lambda row : (row == 'V').sum(), axis=1) >= 1, '1', '0')
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
    elif existing_type == 'C Type-학습': # 연속형만
        continuous_variable_default = ['LQ지수']
        discrete_variable_default = []
    elif existing_type == 'Compact Type': # 연속형 + 이산형
        continuous_variable_default = ['기초학습역량_언어_T점수', '기초학습역량_논리수학_T점수', '기초학습역량_공간_T점수']
        discrete_variable_default = ['상담필요']
        counseling_col = ['우울 수준', '정서충격 수준', '특이성 수준', '예민성 수준', '부정적 대인정서 수준', '과잉행동 수준']
        st.session_state['continuous_variable_default'] = continuous_variable_default
        st.session_state['discrete_variable_default'] = discrete_variable_default
        raw_df['상담필요'] = np.where(raw_df[counseling_col].apply(lambda row : (row == '높음').sum(), axis=1) >= 1, '1', '0')
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
    elif existing_type == 'A+B Type': # 연속형 + 이산형
        continuous_variable_default = ['5요인_외향성', '5요인_신경증', '종합지수']
        counseling_col = ['대인관계 힘듬', '매우 불행', '만성적 학업부진', '자긍심 낮음', '사회적 고립', '심리적 충격', '폭력 경향성', '반사회적 성향', '매우 심한 우울증', 'ADHD 가능성']
        discrete_variable_default = ['상담필요']
        st.session_state['continuous_variable_default'] = continuous_variable_default
        st.session_state['discrete_variable_default'] = discrete_variable_default
        raw_df['상담필요'] = np.where(raw_df[counseling_col].apply(lambda row : (row == 'V').sum(), axis=1) >= 1, '1', '0')
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
    elif existing_type == 'B+C Type': # 연속형 + 이산형
        continuous_variable_default = ['5요인_외향성', '5요인_신경증', 'LQ지수']
        counseling_col = ['대인관계 힘듬', '매우 불행', '만성적 학업부진', '자긍심 낮음', '사회적 고립', '심리적 충격', '폭력 경향성', '반사회적 성향', '매우 심한 우울증', 'ADHD 가능성']
        discrete_variable_default = ['상담필요']
        st.session_state['continuous_variable_default'] = continuous_variable_default
        st.session_state['discrete_variable_default'] = discrete_variable_default
        raw_df['상담필요'] = np.where(raw_df[counseling_col].apply(lambda row : (row == 'V').sum(), axis=1) >= 1, '1', '0')
        st.session_state['raw_df'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
    elif existing_type == 'A+C Type': # 연속형만
        continuous_variable_default = ['종합지수', 'LQ지수']
        discrete_variable_default = []
    else :
        discrete_variable_default = []
    st.session_state['continuous_variable_default'] = continuous_variable_default
    st.session_state['discrete_variable_default'] = discrete_variable_default
except:
    pass

# 연속형 변수 선택
print('existing_type:', existing_type)
print('continuous_variable_default:', st.session_state.get('continuous_variable_default', []))
print('discrete_variable_default:', st.session_state.get('discrete_variable_default', []))
try:
    continuous_variable = st.sidebar.multiselect(
        "연속형 변수를 선택하세요",
        options=st.session_state.get('cols', []) if 'cols' in st.session_state else st.session_state['continuous_variable_default'],
        default=st.session_state['continuous_variable_default'] if existing_type != 'Custom Type' else None,
        help="시험 점수와 같은 연속형 변수를 선택하세요.", key="continuous_variable_multiselect")
    if continuous_variable:
        st.session_state['continuous_variable'] = continuous_variable
        st.sidebar.success("변수 선택이 완료되었습니다.")
    else:
        st.sidebar.warning("변수를 선택해주세요.")
    # 범주형 변수 선택
    discrete_variable = st.sidebar.multiselect(
        "범주형 변수를 선택하세요",
        options=st.session_state.get('cols', []) if 'cols' in st.session_state else st.session_state['discrete_variable_default'],
        default=st.session_state['discrete_variable_default'] if existing_type != 'Custom Type' else None,
        help="성별과 같은 범주형 변수를 선택하세요.", key="discrete_variable_multiselect")
    if discrete_variable:
        st.session_state['discrete_variable'] = discrete_variable
        st.sidebar.success("변수 선택이 완료되었습니다.")
    else:
        st.session_state['discrete_variable'] = []
        st.sidebar.warning("변수를 선택해주세요.")
except:
    pass

# =============== 본문 영역 ===============
st.title("🔧 그룹 분류 파이프라인")

# 본문 탭 구성
tabs = st.tabs(["🔍 명렬표 & 검사결과 비교", "🧪 변수 생성", "⚙️ 분류 알고리즘", "🧠 그룹 분류", "🧑‍🤝‍🧑 학생 관계 재배정", "📊 분류 후 분포 확인", "🔁  이동 및 교환", "📤 배정 결과 내보내기"])

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

        #! 앞단에서 모두 생성함
        # 검사 결과 프레임에서 병합할 열 만들기
        # 학년반번호(5자리) + 성별(1자리) + 이름
        # raw_df['학년반번호'] = raw_df['학년반번호'].astype(str)
        # if raw_df['성별'].dtype == 'O':  # object 타입(문자열)이면 변환
        #     raw_df['성별'] = raw_df['성별'].map({'남': '1', '여': '2'}).astype(str)
        # else:
        #     raw_df['성별'] = raw_df['성별'].astype(str)
        # raw_df['이름'] = raw_df['이름'].astype(str)
        # raw_df['merge_key'] = raw_df['학년반번호'] + raw_df['성별'] + raw_df['이름']

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
            # 결시생 확인
            absent_merged_df = merged_df[merged_df['_merge'] == 'left_only']
            st.write(f"결시생 수: {absent_merged_df.shape[0]}명")
            st.dataframe(absent_merged_df, use_container_width=True)
            st.session_state['absent_merged_df'] = absent_merged_df
            # 동명이인 확인
            dup_names_merged_df = merged_df[merged_df.duplicated('이름_명렬표', keep=False)]
            st.write(f"동명이인 수 : {dup_names_merged_df.shape[0]}명")
            st.dataframe(dup_names_merged_df, use_container_width=True)
            st.session_state['dup_names_merged_df'] = dup_names_merged_df
            # 특수학생 확인
            if '특수학생' in merged_df.columns:
                special_student_df = merged_df[merged_df['특수학생'] == 1]
                st.write(f"특수학생 수 : {special_student_df.shape[0]}명")
                st.dataframe(special_student_df, use_container_width=True)
            else:
                st.info("명렬표에 특수학생 정보가 없어 생략됩니다.")
            # 전출예정학생 확인
            if '전출예정' in merged_df.columns:
                transfer_student_df = merged_df[merged_df['전출예정'] == 1]
                st.write(f"전출예정학생 수 : {transfer_student_df.shape[0]}명")
                st.dataframe(transfer_student_df, use_container_width=True)
            else:
                st.info("명렬표에 전출예정학생 정보가 없어 생략됩니다.")
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
                st.session_state['absent_merged_df'] = merged_df[merged_df['결시생'] == 1]
                st.session_state['dup_names_merged_df'] = merged_df[merged_df['동명이인_ID'].notna()]
        else:
            st.warning("병합을 진행해주세요.")

# [1] 변수 생성 탭
with tabs[1]:
    st.header("변수 생성")
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
    st.write("설정에서 선택한 변수를 활용해 변수를 생성할 수 있습니다.")
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
        if 'merged_df' in st.session_state:
            df = st.session_state['merged_df']
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
        st.session_state['merged_df'] = df
        # 연속형 변수 업데이트
        available_continuous_variables = st.session_state['continuous_variable'] + [st.session_state[f'var_{i+1}']['name'] for i in range(num_variables)]
        st.session_state['available_continuous_variables'] = available_continuous_variables
        # 범주형 변수 업데이트
        available_discrete_variables = st.session_state['discrete_variable']
        st.session_state['available_discrete_variables'] = available_discrete_variables
        # 데이터프레임 표시
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.session_state['available_continuous_variables'] = st.session_state['continuous_variable']
        st.session_state['available_discrete_variables'] = st.session_state['discrete_variable']

# [2] 분류 알고리즘
with tabs[2]:
    #st.header("분류 방법 선택")
    st.write("집단을 분류하고자 할때 사용할 방법을 선택할 수 있습니다.")
    try:
        available_continuous_variables = st.session_state['available_continuous_variables']
        available_discrete_variables = st.session_state['available_discrete_variables']

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
                st.subheader(f"{n+1}순위 정렬 변수")
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
            # 우선순위가 높은 정렬변수는 뒤에 오도록 순서 반전
            selected_sort_variable = {k : v for k, v in reversed(selected_sort_variable.items())}
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
    st.subheader("남여 합/분반 및 집단 수 설정")
    try:
        # 성별 분류 선택
        sex_classification = st.selectbox(
            "남여 합반/분반을 선택해 주세요.",
            options=["합반", "분반", "남학교", "여학교"],
            help="업로드 파일에 '성별' 컬럼이 있는지 꼭 확인해 주세요."
        )
        merged_df = st.session_state['merged_df']
        st.session_state['sex_classification'] = sex_classification
        try:
            if sex_classification == '분반' and merged_df['성별_명렬표'].nunique() == 2:
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
            elif sex_classification == '합반' and merged_df['성별_명렬표'].nunique() == 2:
                group_count = st.number_input(
                    "분류할 집단의 개수를 입력하세요",
                    min_value=2, max_value=10, value=2,
                    help="분류할 집단의 개수를 입력하세요."
                )
                st.session_state['group_count'] = group_count
            elif sex_classification == '남학교' and merged_df['성별_명렬표'].nunique() == 1:
                group_count = st.number_input(
                    "분류할 집단의 개수를 입력하세요",
                    min_value=2, max_value=10, value=2,
                    help="분류할 집단의 개수를 입력하세요."
                )
                st.session_state['group_count'] = group_count
            elif sex_classification == '여학교' and merged_df['성별_명렬표'].nunique() == 1:
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
        index=1,
        help="학생 명렬표에 선택 과목에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['subject_based_classification'] = subject_based_classification
    # 과목별로 그룹 수 설정
    if subject_based_classification == "예" and sex_classification != '분반' and 'merged_df' in st.session_state and 'group_count' in st.session_state:
        subject_name_list = st.session_state['merged_df']['선택과목'].unique().tolist() if '선택과목' in st.session_state['merged_df'].columns else []
        subject_group_counts = {}
        for subject in subject_name_list:
            group_count = st.number_input(
                f"{subject}의 그룹 수를 입력하세요",
                min_value=1, max_value=10, value=1,
                help=f"{subject}의 그룹 수를 입력하세요."
            )
            subject_group_counts[subject] = group_count
        st.session_state['subject_group_counts'] = subject_group_counts
        if sum(subject_group_counts.values()) != st.session_state['group_count']:
            st.error("과목별 그룹 수의 합이 전체 그룹 수와 일치하지 않습니다. 다시 확인해주세요.")
        else :
            pass
    elif subject_based_classification == "예" and sex_classification == '분반' and 'merged_df' in st.session_state and 'male_class_count' in st.session_state and 'female_class_count' in st.session_state:
        subject_name_list = st.session_state['merged_df']['선택과목'].unique().tolist() if '선택과목' in st.session_state['merged_df'].columns else []
        gender_list = [1,2]
        gender_subject_group_counts = {}
        for gender in gender_list:
            for subject in subject_name_list:
                group_count = st.number_input(
                    f"{'남자' if gender == 1 else '여자'}의 {subject} 그룹 수를 입력하세요",
                    min_value=0, max_value=10, value=1,
                    help=f"{'남자' if gender == 1 else '여자'}의 {subject} 그룹 수를 입력하세요."
                )
                gender_subject_group_counts[f"{gender}_{subject}"] = group_count
        st.session_state['gender_subject_group_counts'] = gender_subject_group_counts
        print(gender_subject_group_counts)
        if sum([v for k, v in gender_subject_group_counts.items() if k.startswith('1_')]) != st.session_state['male_class_count'] or sum([v for k, v in gender_subject_group_counts.items() if k.startswith('2_')]) != st.session_state['female_class_count']:
            st.error("과목별 그룹 수의 합이 전체 그룹 수와 일치하지 않습니다. 다시 확인해주세요.")
        else:
            pass
    else:
        pass

    # ! 여기서 부터 아래에 있는 이산형변수는 모두 그룹별 균형 배정이 필요함
    # 결시 학생 처리
    st.subheader("결시생 처리")
    absent_student_handling = st.radio(
        "결시생을 그룹별로 균형있게 배정하시겠습니까?",
        options=["예", "아니오"],
        index=0,
        help="학생 명렬표에 결시생에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['absent_student_handling'] = absent_student_handling
    # 특수 학생 처리
    st.subheader("특수 학생 처리")
    special_student_handling = st.radio(
        "특수 학생을 그룹별로 균형있게 배정하시겠습니까?",
        options=["예", "아니오"],
        index=0,
        help="학생 명렬표에 특수 학생에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['special_student_handling'] = special_student_handling
    # 운동부 학생 처리
    st.subheader("운동부 학생 처리")
    athlete_student_handling = st.radio(
        "운동부 학생을 그룹별로 균형있게 배정하시겠습니까?",
        options=["예", "아니오"],
        index=0,
        help="학생 명렬표에 운동부 학생에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['athlete_student_handling'] = athlete_student_handling
    # 전출학생 처리
    st.subheader("전출 예정 학생 처리")
    transfer_student_handling = st.radio(
        "전출 예정 학생을 그룹별로 균형있게 배정하시겠습니까?",
        options=["예", "아니오"],
        index=0,
        help="학생 명렬표에 전출 예정 학생에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['transfer_student_handling'] = transfer_student_handling
    # 출신 학교 기반 분류
    st.subheader("출신 학교 기반 분류 여부 (추후 개발)")
    school_based_classification = st.radio(
        "출신 학교을 고려해 그룹별로 균형있게 배정하시겠습니까?",
        options=["예", "아니오"],
        index=1,
        help="학생 명렬표에 출신 학교에 대한 정보가 있는 경우 처리 가능합니다."
    )
    st.session_state['school_based_classification'] = school_based_classification

    # if st.session_state.get('group_count', 0) > 0:
    #     full_group_names = []
    #     for i in range(st.session_state['group_count']):
    #         group_name = st.text_input(f"{i+1}번째 반 이름을 입력하세요", value=f"{i+1} 반")
    #         full_group_names.append(group_name)
    #     st.session_state['full_group_names'] = full_group_names
    # else:
    #     st.warning(f"집단 이름 설정 중 오류가 발생했습니다.")
    
    if st.button("그룹 분류 시작"):
        try:
            if all(k in st.session_state for k in ['merged_df', 'selected_algorithm', 'selected_sort_variable_dict', 'selected_discrete_variable', 'sex_classification', 'group_count', 'subject_based_classification', 'absent_student_handling', 'special_student_handling', 'school_based_classification']):
                from init_group_assign import tuple_from_df, suitable_bin_value, init_group_assign
                from cost_group_move_v2 import compute_ideal_discrete_freq, cost_group_move, compute_group_discrete_freq, compute_group_total_cost, compute_group_diff_and_sign, compute_continuous_cost, compute_discrete_cost
                # 병합된 데이터프레임 불러오기
                df = st.session_state['merged_df'] # 앞에서 결시생, 동명이인 처리까지 완료된 데이터프레임
                # 사용자가 성별을 선택한 경우 병합 후에 성별_명렬표로 명시
                selected_discrete_variable = ['성별_명렬표' if var == '성별' else var for var in st.session_state['selected_discrete_variable']]
                st.session_state['selected_discrete_variable'] = selected_discrete_variable
                # 결시생, 특수학생, 운동부, 전출학생, 출신학교 분리 처리
                ## 분리 순서에 따라 우선 순위가 달라질 수 있음
                ## 유지보수에 용이하도록 아래와 같이 리팩터링
                split_df_rules = {
                    'special_student_handling': {'flag_col' : '특수학생', 'save_session_key': 'special_student_df', 'external': False},
                    'transfer_student_handling': {'flag_col' : '전출예정', 'save_session_key': 'transfer_student_df', 'external': False},
                    'athlete_student_handling': {'flag_col' : '운동부', 'save_session_key': 'athlete_student_df', 'external': False},
                    'absent_student_handling': {'flag_col' : '결시생', 'save_session_key': 'absent_merged_df', 'external': True}
                }
                try:
                    for split_rule, rule_info in split_df_rules.items():
                        print(f"Processing split rule: {split_rule}")
                        # 설정이 예가 아니면 생략
                        if st.session_state.get(split_rule, '아니오') != '예':
                            st.session_state[rule_info['save_session_key']] = pd.DataFrame()
                            st.warning(f"명렬표에 {rule_info['flag_col']} 정보가 없어 생략됩니다.")
                            continue
                        # 외부 세션 참조 여부 확인
                        if rule_info['external']:
                            print("  Using external session data for splitting.")
                            ## 외부에서 분리된 데이터프레임 불러오기
                            source_df = st.session_state.get(rule_info['save_session_key'], pd.DataFrame()).copy()
                        else: ## 외부에 없는 경우 merged_df에서 분리
                            if rule_info['flag_col'] not in df.columns: ## 분리할 컬럼이 없는 경우 생략
                                st.warning(f"명렬표에 {rule_info['flag_col']} 정보가 없어 생략됩니다.")
                                st.session_state[rule_info['save_session_key']] = pd.DataFrame()
                                continue
                            else: ## 분리할 컬럼이 있는 경우
                                print(f"  Splitting based on column: {rule_info['flag_col']}")
                                source_df = df[df[rule_info['flag_col']] == 1].copy() ## 분리한 기준 컬럼으로 '1'인 행 분리
                                st.session_state[rule_info['save_session_key']] = source_df
                        # 중복 방지, 앞에서 분리된 학생 제외 처리
                        for prev_rule in split_df_rules:
                            print(f"  Checking previous rule: {prev_rule}")
                            if prev_rule == split_rule: ## 현재 규칙과 동일하면 이전 규칙 검토 루프 탈출
                                print(f"    Skipping current rule as it is the same as previous rule.")
                                break
                            else:
                                prev_df = st.session_state.get(split_df_rules[prev_rule]['save_session_key'], pd.DataFrame())
                                if not prev_df.empty:
                                    source_df = source_df[~source_df['merge_key'].isin(prev_df['merge_key'])]
                                else:
                                    pass
                        # 세션에 저장
                        st.session_state[rule_info['save_session_key']] = source_df
                        # 원본 데이터프레임에서 분리된 학생 제외 처리
                        df = df[~df['merge_key'].isin(source_df['merge_key'])]
                except Exception as e:
                    st.error(f"결시생, 특수학생, 운동부, 전출학생 분리 처리 중 오류가 발생했습니다: {e}")

                #! 출신학교 기반 분리 처리(추후 개발)
                # if st.session_state['school_based_classification'] == '예':
                #     #! 추후 개발
                #     df = df
                # else:
                #     st.session_state['school_based_df'] = pd.DataFrame()
                # 기존 선택한 정렬할 연속형 변수 불러오기
                selected_sort_variable_dict = st.session_state['selected_sort_variable_dict']
                col_names = list(selected_sort_variable_dict.keys())
                # 정렬할 변수 튜플화
                tuples = tuple_from_df(df, col_names) # 앞에서 중요한 정렬변수는 뒤에 오도록 순서 반전 했음
                # 남학교 or 여학교-의미없음-선택과목없음
                if st.session_state['sex_classification'] in ['남학교', '여학교'] and st.session_state['subject_based_classification'] == '아니오':
                    print('남학교 or 여학교, 합반, 선택과목 없음으로 성별 비율 균형 고려하여 그룹 배정 시작')
                    # 적절한 bin_value 찾기
                    sorted_idx, sorted_x, final_bin_value = suitable_bin_value(tuples, st.session_state['group_count'])
                    # 초기 그룹 배정
                    group_assign = init_group_assign(tuples, st.session_state['group_count'], final_bin_value)
                    group_assign = [int(g_n)+1 for g_n in group_assign]
                    # group_assign 데이터 프레임과 병합
                    group_assign_df = df.copy(deep=True)
                    group_assign_df['초기그룹'] = group_assign
                    st.session_state['group_assign_df'] = group_assign_df
                    # cost 함수 기반으로 그룹 배정 최적화
                    group_assign_df = cost_group_move(50, 0.5, 100, 1, group_assign_df, selected_discrete_variable, selected_sort_variable_dict)
                    st.session_state['group_assign_df'] = group_assign_df
                    st.success("초기 그룹 분류가 완료되었습니다.")
                    group_assign_df.to_excel('group_assign_df_관계배정전.xlsx', index=False) #! 초기 그룹 배정 저장

                # 남학교 or 여학교-의미없음-선택과목있음
                elif st.session_state['sex_classification'] in ['남학교', '여학교'] and st.session_state['subject_based_classification'] == '예' and st.session_state['subject_group_counts']:
                    print('남학교 or 여학교, 합반, 선택과목 있음으로 성별 비율 균형 고려하여 그룹 배정 시작')
                    # 선택한 과목 기반으로 데이터프레임 분리
                    subject_group_dict = dict(tuple(df.groupby('선택과목'))) # {'과목명': 데이터프레임}
                    # 분리된 데이터프레임 각각 처리
                    group_assign_df = pd.DataFrame()
                    start_group_number = 0 # 그룹 번호 조정을 위한 변수 -> 그룹명과 매칭하기 위해
                    for subject, subject_df in subject_group_dict.items():
                        subject_group_count = st.session_state['subject_group_counts'].get(subject, 0) # 과목별 그룹 수 가지고오기 (ex 한문 2개, 일본어 1개 등)
                        st.info(f"선택과목 : {subject}, 학생 수 : {subject_df.shape[0]}, 할당된 그룹 수 : {subject_group_count}")
                        subject_tuples = tuple_from_df(subject_df, col_names) # 정렬할 변수 튜플화
                        sorted_idx, sorted_x, final_bin_value = suitable_bin_value(subject_tuples, subject_group_count) # 과목별 분리된 데이터에서 적절한 bin_value 탐색
                        group_assign = init_group_assign(subject_tuples, subject_group_count, final_bin_value) # 과목별 초기 그룹 배정
                        group_assign = [int(g_n)+1 for g_n in group_assign]
                        # 그룹 번호 조정
                        group_assign = [g_n + start_group_number for g_n in group_assign]
                        start_group_number = start_group_number + len(np.unique(group_assign)) # 다음 과목 그룹 번호 조정을 위해
                        # group_assign과 subject_df 병합
                        subject_df['초기그룹'] = group_assign
                        group_assign_df = pd.concat([group_assign_df, subject_df], axis=0)
                    st.session_state['group_assign_df'] = group_assign_df
                    # cost 함수 기반으로 그룹 배정 최적화
                    group_assign_df = cost_group_move(50, 0.5, 100, 1, group_assign_df, selected_discrete_variable, selected_sort_variable_dict)
                    st.session_state['group_assign_df'] = group_assign_df
                    st.success("초기 그룹 분류가 완료되었습니다.")
                    group_assign_df.to_excel('group_assign_df_관계배정전.xlsx', index=False)

                # 남여공학-분반-선택과목없음
                elif st.session_state['sex_classification'] == '분반' and st.session_state['subject_based_classification'] == '아니오':
                    print('남여공학, 분반, 선택과목 없음으로 성별 비율 균형 고려하여 그룹 배정 시작')
                    # 선택한 과목 기반으로 데이터프레임 분리
                    gender_group_dict = dict(tuple(df.groupby('성별_명렬표'))) # {'성별': 데이터프레임}
                    # 분리된 데이터프레임 각각 처리
                    group_assign_df = pd.DataFrame()
                    start_group_number = 0
                    for gender, gender_df in gender_group_dict.items():
                        gender_group_count = st.session_state['male_class_count'] if gender == '1' else st.session_state['female_class_count'] # 성별에 따른 그룹 수 할당
                        st.info(f"성별 : {gender}, 학생 수 : {gender_df.shape[0]}, 할당된 그룹 수 : {gender_group_count}")
                        gender_tuples = tuple_from_df(gender_df, col_names)
                        sorted_idx, sorted_x, final_bin_value = suitable_bin_value(gender_tuples, gender_group_count)
                        gender_group_assign = init_group_assign(gender_tuples, gender_group_count, final_bin_value)
                        gender_group_assign = [int(g_n)+1 for g_n in gender_group_assign]
                        # 그룹 번호 조정
                        gender_group_assign = [g_n + start_group_number for g_n in gender_group_assign]
                        start_group_number = start_group_number + len(np.unique(gender_group_assign))
                        # group_assign과 gender_df 병합
                        gender_df['초기그룹'] = gender_group_assign
                        # cost 함수 기반으로 그룹 배정 최적화
                        if "성별_명렬표" in selected_discrete_variable: # 이미 group by로 성별을 분리했으니 성별은 제외하고 처리
                            selected_discrete_variable.remove("성별_명렬표")
                        else:
                            pass
                        gender_group_assign_df = cost_group_move(50, 0.5, 100, 1, gender_df, selected_discrete_variable, selected_sort_variable_dict)
                        group_assign_df = pd.concat([group_assign_df, gender_group_assign_df], axis=0)
                    st.session_state['group_assign_df'] = group_assign_df
                    st.success("초기 그룹 분류가 완료되었습니다.")
                    group_assign_df.to_excel('group_assign_df_관계배정전.xlsx', index=False)

                # 남여공학-분반-선택과목있음
                elif st.session_state['sex_classification'] == '분반' and st.session_state['subject_based_classification'] == '예':
                    print('남여공학, 분반, 선택과목 있음으로 성별 비율 균형 고려하여 그룹 배정 시작')
                    # 성별, 선택한 과목 기반으로 데이터프레임 분리
                    gender_group_dict = dict(tuple(df.groupby(['성별_명렬표', '선택과목']))) # {('성별', '과목명'): 데이터프레임}
                    # 분리된 데이터프레임 각각 처리
                    group_assign_df = pd.DataFrame()
                    start_group_number = 0
                    for (gender, subject), gender_subject_df in gender_group_dict.items(): # gender_subject_df : 특정 성별, 특정 과목만 있는 데이터프레임
                        gender_subject_group_count = st.session_state['gender_subject_group_counts'].get((f'{gender}_{subject}'), 0)
                        st.info(f"성별: {gender}, 선택과목 : {subject}, 학생수: {gender_subject_df.shape[0]}, 할당된 그룹 수 : {gender_subject_group_count}")
                        gender_tuples = tuple_from_df(gender_subject_df, col_names)
                        sorted_idx, sorted_x, final_bin_value = suitable_bin_value(gender_tuples, gender_subject_group_count)
                        group_assign = init_group_assign(gender_tuples, gender_subject_group_count, final_bin_value)
                        group_assign = [int(g_n)+1 for g_n in group_assign]
                        # 그룹 번호 조정
                        group_assign = [g_n + start_group_number for g_n in group_assign]
                        start_group_number = start_group_number + len(np.unique(group_assign))
                        # group_assign과 gender_subject_df 병합
                        gender_subject_df['초기그룹'] = group_assign
                        # cost 함수 기반으로 그룹 배정 최적화
                        if "성별_명렬표" in selected_discrete_variable: # 이미 group by로 성별을 분리했으니 성별은 제외하고 처리
                            selected_discrete_variable.remove("성별_명렬표")
                        else:
                            pass
                        gender_subject_df = cost_group_move(50, 0.5, 100, 1, gender_subject_df, selected_discrete_variable, selected_sort_variable_dict)
                        group_assign_df = pd.concat([group_assign_df, gender_subject_df], axis=0)
                    st.session_state['group_assign_df'] = group_assign_df
                    st.success("초기 그룹 분류가 완료되었습니다.")
                    group_assign_df.to_excel('group_assign_df_관계배정전.xlsx', index=False)

                elif st.session_state['sex_classification'] == '합반' and st.session_state['subject_based_classification'] == '아니오':
                    print('남여공학, 합반, 선택과목 없음으로 성별 비율 균형 고려하여 그룹 배정 시작')
                    # 적절한 bin_value 찾기
                    sorted_idx, sorted_x, final_bin_value = suitable_bin_value(tuples, st.session_state['group_count'])
                    # 초기 그룹 배정
                    group_assign = init_group_assign(tuples, st.session_state['group_count'], final_bin_value)
                    group_assign = [int(g_n)+1 for g_n in group_assign]
                    st.session_state['group_assign'] = group_assign
                    # group_assign과 merged_df 병합
                    group_assign_df = df.copy(deep=True)
                    group_assign_df['초기그룹'] = group_assign
                    st.session_state['group_assign_df'] = group_assign_df
                    # cost 함수 기반으로 그룹 배정 최적화
                    print('초기 배정 병합 후 이산형 변수 열 확인', )
                    group_assign_df = cost_group_move(50, 0.5, 100, 1, group_assign_df, selected_discrete_variable, selected_sort_variable_dict)
                    st.session_state['group_assign_df'] = group_assign_df
                    st.success("초기 그룹 분류가 완료되었습니다.")
                    group_assign_df.to_excel('group_assign_df_관계배정전.xlsx', index=False)

                elif st.session_state['sex_classification'] == '합반' and st.session_state['subject_based_classification'] == '예':
                    print('남여공학, 합반, 선택과목 있음으로 성별 비율 균형 고려하여 그룹 배정 시작')
                    # 선택한 과목 기반으로 데이터프레임 분리
                    subject_group_dict = dict(tuple(df.groupby('선택과목'))) # {'과목명': 데이터프레임}
                    # 분리된 데이터프레임 각각 처리
                    group_assign_df = pd.DataFrame()
                    start_group_number = 0
                    for subject, subject_df in subject_group_dict.items():
                        subject_group_count = st.session_state['subject_group_counts'].get(subject, 0) # 과목별 그룹 수 가지고오기
                        st.info(f"선택과목: {subject}, 학생 수: {subject_df.shape[0]}, 할당된 그룹 수: {subject_group_count}")
                        subject_tuples = tuple_from_df(subject_df, col_names)
                        sorted_idx, sorted_x, final_bin_value = suitable_bin_value(subject_tuples, subject_group_count)
                        subject_group_assign = init_group_assign(subject_tuples, subject_group_count, final_bin_value)
                        subject_group_assign = [int(g_n)+1 for g_n in subject_group_assign]
                        # 그룹 번호 조정
                        subject_group_assign = [g_n + start_group_number for g_n in subject_group_assign]
                        start_group_number = start_group_number + len(np.unique(subject_group_assign))
                        # group_assign과 subject_df 병합
                        subject_df['초기그룹'] = subject_group_assign
                        # cost 함수 기반으로 그룹 배정 최적화
                        subject_group_assign_df = cost_group_move(50, 0.5, 100, 1, subject_df, selected_discrete_variable, selected_sort_variable_dict)
                        group_assign_df = pd.concat([group_assign_df, subject_group_assign_df], axis=0)
                    st.session_state['group_assign_df'] = group_assign_df
                    st.success("초기 그룹 분류가 완료되었습니다.")
                    group_assign_df.to_excel('group_assign_df_관계배정전.xlsx', index=False)
                else:
                    st.error("그룹 분류에 필요한 설정이 올바르게 되어있는지 확인해주세요.")
            else:
                st.error("그룹 분류에 필요한 설정이 올바르게 되어있는지 확인해주세요.")

            # 특수학생 처리
            ## 특수학생 그룹별로 균일하게 배치
            ## 특수학생이 결시할 경우 결시생이 아닌 특수학생 취급
            group_assign_df = st.session_state['group_assign_df']
            if st.session_state['special_student_handling'] == '예' and '특수학생' in group_assign_df.columns:
                try:
                    # 케이스별 groupby로 기준 설정
                    special_sex_choice = st.session_state['sex_classification']
                    special_subject_choice = st.session_state['subject_based_classification']
                    if special_sex_choice == '분반' and special_subject_choice == '예':
                        groupby_cols = ['성별_명렬표', '선택과목']
                    elif special_sex_choice == '분반' and special_subject_choice == '아니오':
                        groupby_cols = ['성별_명렬표']
                    elif special_sex_choice == '합반' and special_subject_choice == '예':
                        groupby_cols = ['선택과목']
                    elif special_sex_choice == '합반' and special_subject_choice == '아니오':
                        groupby_cols = [] # 전체 그룹 대상으로
                    elif special_sex_choice in ['남학교', '여학교'] and special_subject_choice == '예':
                        groupby_cols = ['선택과목']
                    elif special_sex_choice in ['남학교', '여학교'] and special_subject_choice == '아니오':
                        groupby_cols = [] # 전체 그룹 대상이지만 남학교/여학교로 성별은 이미 하나임
                    else:
                        groupby_cols = []
                    # 그룹 단위별 특수학생 배정
                    group_assign_df = st.session_state['group_assign_df'] # 그룹고정 열 포함된 데이터프레임
                    group_assign_df['그룹고정'] = False
                    special_student_df = st.session_state['special_student_df'] # 앞에서 분리한 특수학생 데이터프레임
                    special_student_df['그룹고정'] = False
                    special_assign_results = []
                    if groupby_cols:
                        grouped_dfs = dict(tuple(group_assign_df.groupby(groupby_cols))) # 그룹별 데이터프레임 생성해서 dict로 저장
                        for sub_group_keys, sub_group_df in grouped_dfs.items():
                            # 특수학생 중 해당 그룹에 맞는 학생 필터링
                            if groupby_cols:
                                if isinstance(sub_group_keys, tuple): # 다중 조건일 때
                                    condition = pd.Series(True, index=special_student_df.index) # 특수학생 인덱스에 맞춰 true 시리즈 생성
                                    for col, key in zip(groupby_cols, sub_group_keys): # col : key 쌍으로 반복
                                        condition &= (special_student_df[col] == key) # col : key 조건 추가하여 &=으로 조건 누적
                                else: # 단일 조건일 때
                                    condition = (special_student_df[groupby_cols[0]] == sub_group_keys)
                                filtered_special_student_df = special_student_df[condition].copy() # true로 남은 인덱스를 가지고 필터링
                            else: # groupby_cols가 없는 경우 전체 특수학생 대상
                                filtered_special_student_df = special_student_df.copy()
                            # 해당 그룹에 맞는 결시생이 없는 경우
                            if filtered_special_student_df.empty:
                                special_assign_results.append(sub_group_df) # 특수학생이 없는 경우 기존 그룹 데이터프레임 그대로 추가
                                continue
                            # group_df에서 그룹 번호별 현재 인원수 파악
                            # 돌아가면서 특수학생 배정 & 특수학생은 해당 그룹 고정 옵션 추가
                            sub_group_counts = sub_group_df['초기그룹'].value_counts().to_dict() # groupby된 데이터프레임에서 그룹별 인원수 파악
                            g_idx = 0
                            sorted_sub_group_no = sorted(sub_group_counts, key=sub_group_counts.get) # 인원수 적은 그룹부터 정렬 후 키값만 리스트로 반환
                            for idx, row in filtered_special_student_df.iterrows():
                                # 인원 오름차순에 따라 결시생 순환 배정
                                filtered_special_student_df.loc[idx, '초기그룹'] = sorted_sub_group_no[g_idx]
                                filtered_special_student_df.loc[idx, '그룹고정'] = True
                                g_idx = (g_idx + 1) % len(sorted_sub_group_no) # 다음 그룹 인덱스로 순환
                            # 그룹 배정된 결시생과 해당 그룹 데이터프레임 병합
                            mergered_group_assign_df = pd.concat([sub_group_df, filtered_special_student_df], axis=0)
                            special_assign_results.append(mergered_group_assign_df)
                        # 모든 그룹별 결시생 배정 결과 병합
                        group_assign_df = pd.concat(special_assign_results, axis=0)
                        st.session_state['group_assign_df'] = group_assign_df
                        st.success("특수학생 균등 배정이 완료되었습니다. 분류 후 분포 확인 탭에서 결과를 확인하세요.")
                        group_assign_df.to_excel('group_assign_df_특수학생배정완료.xlsx', index=False) #! 특수학생 배정 저장
                    elif groupby_cols == []: # 전체 그룹 대상으로 균등 배정
                        group_counts = group_assign_df['초기그룹'].value_counts().to_dict() # groupby된 데이터프레임에서 그룹별 인원수 파악
                        g_idx = 0
                        sorted_group_no = sorted(group_counts, key=group_counts.get) # 인원수 적은 그룹부터 정렬 후 키값만 리스트로 반환
                        for idx, row in special_student_df.iterrows():
                            # 인원 오름차순에 따라 결시생 순환 배정
                            special_student_df.loc[idx, '초기그룹'] = sorted_group_no[g_idx]
                            special_student_df.loc[idx, '그룹고정'] = True
                            g_idx = (g_idx + 1) % len(sorted_group_no) # 다음 그룹 인덱스로 순환
                        # 그룹 배정된 결시생과 기존 그룹 데이터프레임 병합
                        group_assign_df = pd.concat([group_assign_df, special_student_df], axis=0)
                        st.session_state['group_assign_df'] = group_assign_df
                    else:
                        st.error("특수학생 균등 배정 중 오류가 발생했습니다. 그룹화 기준이 올바른지 확인해주세요.")
                        traceback.print_exc()
                except Exception as e:
                    st.error(f"특수학생 균등 배정 중 오류가 발생했습니다: {e}")
            else :
                pass
            # 전출학생 처리
            ## 전출학생 그룹별로 균일하게 배치
            ## 추후 학생 번호 부여시에 마지막 번호로 처리
            if st.session_state['transfer_student_handling'] == '예' and '전출예정' in group_assign_df.columns:
                try:
                    # 케이스별 groupby로 기준 설정
                    transfer_sex_choice = st.session_state['sex_classification']
                    transfer_subject_choice = st.session_state['subject_based_classification']
                    if transfer_sex_choice == '분반' and transfer_subject_choice == '예':
                        groupby_cols = ['성별_명렬표', '선택과목']
                    elif transfer_sex_choice == '분반' and transfer_subject_choice == '아니오':
                        groupby_cols = ['성별_명렬표']
                    elif transfer_sex_choice == '합반' and transfer_subject_choice == '예':
                        groupby_cols = ['선택과목']
                    elif transfer_sex_choice == '합반' and transfer_subject_choice == '아니오':
                        groupby_cols = [] # 전체 그룹 대상으로
                    elif transfer_sex_choice in ['남학교', '여학교'] and transfer_subject_choice == '예':
                        groupby_cols = ['선택과목']
                    elif transfer_sex_choice in ['남학교', '여학교'] and transfer_subject_choice == '아니오':
                        groupby_cols = [] # 전체 그룹 대상이지만 남학교/여학교로 성별은 이미 하나임
                    else:
                        groupby_cols = []
                    # 그룹 단위별 전출학생 배정
                    group_assign_df = st.session_state['group_assign_df'] # 그룹고정 열 포함된 데이터프레임
                    transfer_student_df = st.session_state['transfer_student_df']
                    transfer_student_df['그룹고정'] = False
                    transfer_assign_results = []
                    if groupby_cols:
                        grouped_dfs = dict(tuple(group_assign_df.groupby(groupby_cols))) # 그룹별 데이터프레임 생성해서 dict로 저장
                        for sub_group_keys, sub_group_df in grouped_dfs.items():
                            # 특수학생 중 해당 그룹에 맞는 학생 필터링
                            if groupby_cols:
                                if isinstance(sub_group_keys, tuple): # 다중 조건일 때
                                    condition = pd.Series(True, index=special_student_df.index) # 특수학생 인덱스에 맞춰 true 시리즈 생성
                                    for col, key in zip(groupby_cols, sub_group_keys): # col : key 쌍으로 반복
                                        condition &= (special_student_df[col] == key) # col : key 조건 추가하여 &=으로 조건 누적
                                else: # 단일 조건일 때
                                    condition = (special_student_df[groupby_cols[0]] == sub_group_keys)
                                filtered_transfer_student_df = transfer_student_df[condition].copy() # true로 남은 인덱스를 가지고 필터링
                            else:
                                filtered_transfer_student_df = transfer_student_df.copy()
                            # 해당 그룹에 맞는 결시생이 없는 경우 건너뛰기
                            if filtered_transfer_student_df.empty:
                                transfer_assign_results.append(sub_group_df) # 전출학생이이 없는 경우 기존 그룹 데이터프레임 그대로 추가
                                continue
                            # group_df에서 그룹 번호별 현재 인원수 파악
                            # 돌아가면서 특수학생 배정 & 특수학생은 해당 그룹 고정 옵션 추가
                            sub_group_counts = sub_group_df['초기그룹'].value_counts().to_dict() # groupby된 데이터프레임에서 그룹별 인원수 파악
                            g_idx = 0
                            sorted_sub_group_no = sorted(sub_group_counts, key=sub_group_counts.get) # 인원수 적은 그룹부터 정렬 후 키값만 리스트로 반환
                            for idx, row in filtered_transfer_student_df.iterrows():
                                # 인원 오름차순에 따라 결시생 순환 배정
                                filtered_transfer_student_df.loc[idx, '초기그룹'] = sorted_sub_group_no[g_idx]
                                filtered_transfer_student_df.loc[idx, '그룹고정'] = True
                                g_idx = (g_idx + 1) % len(sorted_sub_group_no) # 다음 그룹 인덱스로 순환
                            # 그룹 배정된 결시생과 해당 그룹 데이터프레임 병합
                            mergered_group_assign_df = pd.concat([sub_group_df, filtered_transfer_student_df], axis=0)
                            transfer_assign_results.append(mergered_group_assign_df)
                        # 모든 그룹별 결시생 배정 결과 병합
                        group_assign_df = pd.concat(transfer_assign_results, axis=0)
                        st.session_state['group_assign_df'] = group_assign_df
                        st.success("전출학생 균등 배정이 완료되었습니다. 분류 후 분포 확인 탭에서 결과를 확인하세요.")
                        group_assign_df.to_excel('group_assign_df_전출학생배정완료.xlsx', index=False) #! 전출학생 배정 저장
                    elif groupby_cols == []: # 전체 그룹 대상으로 균등 배정
                        group_counts = group_assign_df['초기그룹'].value_counts().to_dict() # groupby된 데이터프레임에서 그룹별 인원수 파악
                        g_idx = 0
                        sorted_group_no = sorted(group_counts, key=group_counts.get) # 인원수 적은 그룹부터 정렬 후 키값만 리스트로 반환
                        for idx, row in transfer_student_df.iterrows():
                            # 인원 오름차순에 따라 결시생 순환 배정
                            transfer_student_df.loc[idx, '초기그룹'] = sorted_group_no[g_idx]
                            transfer_student_df.loc[idx, '그룹고정'] = True
                            g_idx = (g_idx + 1) % len(sorted_group_no) # 다음 그룹 인덱스로 순환
                        # 그룹 배정된 결시생과 기존 그룹 데이터프레임 병합
                        group_assign_df = pd.concat([group_assign_df, transfer_student_df], axis=0)
                        st.session_state['group_assign_df'] = group_assign_df
                    else:
                        st.error("전출학생 균등 배정 중 오류가 발생했습니다. 그룹화 기준이 올바른지 확인해주세요.")
                except Exception as e:
                    st.error(f"전출학생 균등 배정 중 오류가 발생했습니다: {e}")
            else:
                pass
            # 운동부 처리
            ## 운동부 학생 그룹별로 균일하게 배치
            ## 추후 학생 번호 부여시에 마지막 번호로 처리
            if st.session_state['athlete_student_handling'] == '예' and '운동부' in group_assign_df.columns:
                try:
                    # 케이스별 groupby로 기준 설정
                    athlete_sex_choice = st.session_state['sex_classification']
                    athlete_subject_choice = st.session_state['subject_based_classification']
                    if athlete_sex_choice == '분반' and athlete_subject_choice == '예':
                        groupby_cols = ['성별_명렬표', '선택과목']
                    elif athlete_sex_choice == '분반' and athlete_subject_choice == '아니오':
                        groupby_cols = ['성별_명렬표']
                    elif athlete_sex_choice == '합반' and athlete_subject_choice == '예':
                        groupby_cols = ['선택과목']
                    elif athlete_sex_choice == '합반' and athlete_subject_choice == '아니오':
                        groupby_cols = [] # 전체 그룹 대상으로
                    elif athlete_sex_choice in ['남학교', '여학교'] and athlete_subject_choice == '예':
                        groupby_cols = ['선택과목']
                    elif athlete_sex_choice in ['남학교', '여학교'] and athlete_subject_choice == '아니오':
                        groupby_cols = [] # 전체 그룹 대상이지만 남학교/여학교로 성별은 이미 하나임
                    else:
                        groupby_cols = []
                    # 그룹 단위별 전출학생 배정
                    group_assign_df = st.session_state['group_assign_df'] # 그룹고정 열 포함된 데이터프레임
                    athlete_student_df = st.session_state['athlete_student_df']
                    athlete_student_df['그룹고정'] = False
                    athlete_assign_results = []
                    if groupby_cols:
                        grouped_dfs = dict(tuple(group_assign_df.groupby(groupby_cols))) # 그룹별 데이터프레임 생성해서 dict로 저장
                        for sub_group_keys, sub_group_df in grouped_dfs.items():
                            # 운동부 중 해당 그룹에 맞는 학생 필터링
                            if groupby_cols:
                                if isinstance(sub_group_keys, tuple): # 다중 조건일 때
                                    condition = pd.Series(True, index=athlete_student_df.index) # 운동부 인덱스에 맞춰 true 시리즈 생성
                                    for col, key in zip(groupby_cols, sub_group_keys): # col : key 쌍으로 반복
                                        condition &= (athlete_student_df[col] == key) # col : key 조건 추가하여 &=으로 조건 누적
                                else: # 단일 조건일 때
                                    condition = (athlete_student_df[groupby_cols[0]] == sub_group_keys)
                                filtered_athlete_student_df = athlete_student_df[condition].copy() # true로 남은 인덱스를 가지고 필터링
                            else:
                                filtered_athlete_student_df = athlete_student_df.copy()
                            # 해당 그룹에 맞는 결시생이 없는 경우 건너뛰기
                            if filtered_athlete_student_df.empty:
                                athlete_assign_results.append(sub_group_df) # 운동부가 없는 경우 기존 그룹 데이터프레임 그대로 추가
                                continue
                            # group_df에서 그룹 번호별 현재 인원수 파악
                            # 돌아가면서 운동부 배정 & 운동부는 해당 그룹 고정 옵션 추가
                            sub_group_counts = sub_group_df['초기그룹'].value_counts().to_dict() # groupby된 데이터프레임에서 그룹별 인원수 파악
                            g_idx = 0
                            sorted_sub_group_no = sorted(sub_group_counts, key=sub_group_counts.get) # 인원수 적은 그룹부터 정렬 후 키값만 리스트로 반환
                            for idx, row in filtered_athlete_student_df.iterrows():                                
                                # 인원 오름차순에 따라 결시생 순환 배정
                                filtered_athlete_student_df.loc[idx, '초기그룹'] = sorted_sub_group_no[g_idx]
                                filtered_athlete_student_df.loc[idx, '그룹고정'] = True
                                g_idx = (g_idx + 1) % len(sorted_sub_group_no) # 다음 그룹 인덱스로 순환
                            # 그룹 배정된 결시생과 해당 그룹 데이터프레임 병합
                            mergered_group_assign_df = pd.concat([sub_group_df, filtered_athlete_student_df], axis=0)
                            athlete_assign_results.append(mergered_group_assign_df)
                        # 모든 그룹별 결시생 배정 결과 병합
                        group_assign_df = pd.concat(athlete_assign_results, axis=0)
                        st.session_state['group_assign_df'] = group_assign_df
                        st.success("운동부 균등 배정이 완료되었습니다. 분류 후 분포 확인 탭에서 결과를 확인하세요.")
                        group_assign_df.to_excel('group_assign_df_운동부배정완료.xlsx', index=False) #! 운동부 배정 저장
                    elif groupby_cols == []: # 전체 그룹 대상으로 균등 배정
                        group_counts = group_assign_df['초기그룹'].value_counts().to_dict() # groupby된 데이터프레임에서 그룹별 인원수 파악
                        g_idx = 0
                        sorted_group_no = sorted(group_counts, key=group_counts.get) # 인원수 적은 그룹부터 정렬 후 키값만 리스트로 반환
                        for idx, row in athlete_student_df.iterrows():
                            # 인원 오름차순에 따라 결시생 순환 배정
                            athlete_student_df.loc[idx, '초기그룹'] = sorted_group_no[g_idx]
                            athlete_student_df.loc[idx, '그룹고정'] = True
                            g_idx = (g_idx + 1) % len(sorted_group_no) # 다음 그룹 인덱스로 순환
                        # 그룹 배정된 결시생과 기존 그룹 데이터프레임 병합
                        group_assign_df = pd.concat([group_assign_df, athlete_student_df], axis=0)
                        st.session_state['group_assign_df'] = group_assign_df
                    else:
                        st.error("운동부 균등 배정 중 오류가 발생했습니다. 그룹화 기준이 올바른지 확인해주세요.")
                except Exception as e:
                    st.error(f"운동부 균등 배정 중 오류가 발생했습니다: {e}")
            else:
                pass
            # 결시생 처리
            ## 결시생을 그룹별로 균일하게 배치하는데, 성별을 고려해서 골고루 배치해야함
            ## 그러나 특정한 경우 결시생이 하나의 그룹에 몰릴 수 있음
            ### 1.그룹별 성별 편차 산출 2.음... 애매하네 균등배정도 되어야하는데 관계 재배정할때 틀어질 확률이 높은데
            if st.session_state['absent_student_handling'] == '예' and not st.session_state['absent_merged_df'].empty:
                try:
                    # 케이스별 groupby로 기준 설정
                    absent_sex_choice = st.session_state['sex_classification']
                    absent_subject_choice = st.session_state['subject_based_classification']
                    if absent_sex_choice == '분반' and absent_subject_choice == '예':
                        groupby_cols = ['성별_명렬표', '선택과목']
                    elif absent_sex_choice == '분반' and absent_subject_choice == '아니오':
                        groupby_cols = ['성별_명렬표']
                    elif absent_sex_choice == '합반' and absent_subject_choice == '예':
                        groupby_cols = ['선택과목']
                    elif absent_sex_choice == '합반' and absent_subject_choice == '아니오':
                        groupby_cols = [] # 전체 그룹 대상으로
                    elif absent_sex_choice in ['남학교', '여학교'] and absent_subject_choice == '예':
                        groupby_cols = ['선택과목']
                    elif absent_sex_choice in ['남학교', '여학교'] and absent_subject_choice == '아니오':
                        groupby_cols = [] # 전체 그룹 대상이지만 남학교/여학교로 성별은 이미 하나임
                    else:
                        groupby_cols = []
                    # 그룹 단위별 결시생 배정
                    group_assign_df = st.session_state['group_assign_df']
                    absent_df = st.session_state['absent_merged_df']
                    absent_df['그룹고정'] = False
                    absent_assign_results = []
                    if groupby_cols:
                        grouped_dfs = dict(tuple(group_assign_df.groupby(groupby_cols))) # 그룹별 데이터프레임 생성해서 dict로 저장
                        for sub_group_keys, sub_group_df in grouped_dfs.items():
                            # 결시생 중 해당 그룹에 맞는 학생 필터링
                            if groupby_cols:
                                if isinstance(sub_group_keys, tuple): # 다중 조건일 때
                                    condition = pd.Series(True, index=absent_df.index) # 결시생 인덱스에 맞춰 true 시리즈 생성
                                    for col, key in zip(groupby_cols, sub_group_keys): # col : key 쌍으로 반복
                                        condition &= (absent_df[col] == key) # col : key 조건 추가하여 &=으로 조건 누적
                                else: # 단일 조건일 때
                                    condition = (absent_df[groupby_cols[0]] == sub_group_keys)
                                filtered_absent_df = absent_df[condition].copy() # true로 남은 인덱스를 가지고 필터링
                            else:
                                filtered_absent_df = absent_df.copy()
                            # 해당 그룹에 맞는 결시생이 없는 경우 건너뛰기
                            if filtered_absent_df.empty:
                                absent_assign_results.append(sub_group_df) # 결시생이 없는 경우 기존 그룹 데이터프레임 그대로 추가
                                continue
                            # group_df에서 그룹 번호별 현재 인원수 파악
                            # 돌아가면서 결시생 배정 & 결시생은 해당 그룹 고정 옵션 추가
                            sub_group_counts = sub_group_df['초기그룹'].value_counts().to_dict() # groupby된 데이터프레임에서 그룹별 인원수 파악
                            print(f"{sub_group_keys}: {sub_group_counts}")
                            g_idx = 0
                            sorted_sub_group_no = sorted(sub_group_counts, key=sub_group_counts.get) # 인원수 적은 그룹부터 정렬 후 키값만 리스트로 반환
                            for idx, row in filtered_absent_df.iterrows():
                                # 인원 오름차순에 따라 결시생 순환 배정
                                filtered_absent_df.loc[idx, '초기그룹'] = sorted_sub_group_no[g_idx]
                                print(f"결시생 {row['merge_key']} -> 그룹 {sorted_sub_group_no[g_idx]}")
                                filtered_absent_df.loc[idx, '그룹고정'] = True
                                g_idx = (g_idx + 1) % len(sorted_sub_group_no) # 다음 그룹 인덱스로 순환
                            # 그룹 배정된 결시생과 해당 그룹 데이터프레임 병합
                            mergered_group_assign_df = pd.concat([sub_group_df, filtered_absent_df], axis=0)
                            absent_assign_results.append(mergered_group_assign_df)
                        # 모든 그룹별 결시생 배정 결과 병합
                        group_assign_df = pd.concat(absent_assign_results, axis=0)
                        st.session_state['group_assign_df'] = group_assign_df
                        st.success("결시생 균등 배정이 완료되었습니다. 분류 후 분포 확인 탭에서 결과를 확인하세요.")
                        group_assign_df.to_excel('group_assign_df_결시생배정완료.xlsx', index=False) #! 결시생 배정 저장
                    elif groupby_cols == []: # 전체 그룹 대상으로 균등 배정
                        group_counts = group_assign_df['초기그룹'].value_counts().to_dict() # groupby된 데이터프레임에서 그룹별 인원수 파악
                        g_idx = 0
                        sorted_group_no = sorted(group_counts, key=group_counts.get) # 인원수 적은 그룹부터 정렬 후 키값만 리스트로 반환
                        for idx, row in absent_df.iterrows():
                            # 인원 오름차순에 따라 결시생 순환 배정
                            absent_df.loc[idx, '초기그룹'] = sorted_group_no[g_idx]
                            absent_df.loc[idx, '그룹고정'] = True
                            g_idx = (g_idx + 1) % len(sorted_group_no) # 다음 그룹 인덱스로 순환
                        # 그룹 배정된 결시생과 기존 그룹 데이터프레임 병합
                        group_assign_df = pd.concat([group_assign_df, absent_df], axis=0)
                        st.session_state['group_assign_df'] = group_assign_df
                    else:
                        st.error("결시생 균등 배정 중 오류가 발생했습니다. 그룹화 기준이 올바른지 확인해주세요.")
                        traceback.print_exc()
                except Exception as e:
                    st.error(f"결시생 균등 배정 중 오류가 발생했습니다: {e}")
            else:
                pass
            # 균형 배정된 학생(특수학생, 전출학생, 운동부, 결시생 등) 그룹별 빈도 확인
            initial_sex_choice = st.session_state['sex_classification']
            initial_subject_choice = st.session_state['subject_based_classification']
            if initial_sex_choice == '분반' and initial_subject_choice == '예':
                groupby_cols = ['성별_명렬표', '선택과목']
            elif initial_sex_choice == '분반' and initial_subject_choice == '아니오':
                groupby_cols = ['성별_명렬표']
            elif initial_sex_choice == '합반' and initial_subject_choice == '예':
                groupby_cols = ['선택과목']
            elif initial_sex_choice == '합반' and initial_subject_choice == '아니오':
                groupby_cols = [] # 전체 그룹 대상으로
            elif initial_sex_choice in ['남학교', '여학교'] and initial_subject_choice == '예':
                groupby_cols = ['선택과목']
            elif initial_sex_choice in ['남학교', '여학교'] and initial_subject_choice == '아니오':
                groupby_cols = [] # 전체 그룹 대상이지만 남학교/여학교로 성별은 이미 하나임
            else:
                groupby_cols = []
            groupby_cols = ['초기그룹'] + groupby_cols if groupby_cols else ['초기그룹']
            candidate_cols = ['특수학생', '전출예정', '운동부', '결시생']
            existing_cols = [col for col in candidate_cols if col in group_assign_df.columns]
            freq_df = (group_assign_df.groupby(groupby_cols)[existing_cols].sum().astype(int))
            st.markdown("##### 그룹별 배정된 특이분류학생(특수학생, 전출예정, 운동부, 결시생 등) 현황")
            st.dataframe(freq_df, use_container_width=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            st.error(f"그룹 분류 중 오류가 발생했습니다: {e}")

# [4] 학생 관계 배정-------------------------------------------------
with tabs[4]:
    st.subheader("학생 관계 재배정")
    st.write("학생 간의 관계를 고려하여 기존 그룹 배정을 조정합니다.")

    # 주체 및 대상 검색 후 선택 및 설정
    if 'group_assign_df' in st.session_state:
        df = st.session_state['group_assign_df']
        all_students = sorted(df['merge_key'].unique().tolist())

        # 세션 초기화
        if 'relationship_dict' not in st.session_state:
            st.session_state['relationship_dict'] = {}

        # 🔍 1️⃣ 주체 학생 검색 및 선택
        st.markdown("##### ① 관계를 설정할 학생 선택")
        search_main = st.text_input("주체 학생 이름 검색")
        filtered_main = [s for s in all_students if search_main in s] if search_main else all_students

        selected_main = st.selectbox(
            "주체 학생 선택 (학년+반+번호+이름)",
            options=filtered_main,
            help="검색 후 관계를 설정할 학생을 선택하세요."
        )

        if selected_main:
            # 🔍 2️⃣ 대상 학생 검색 및 필터링
            st.markdown(f"##### ② **{selected_main}** 학생과의 관계 설정")
            search_target = st.text_input("대상 학생 이름 검색")
            target_candidates = [s for s in all_students if s != selected_main]
            filtered_targets = [s for s in target_candidates if search_target in s] if search_target else target_candidates

            if not filtered_targets:
                st.warning("검색 결과가 없습니다.")
            else:
                relations = st.session_state['relationship_dict'].get(selected_main, {})

                # 대상 학생별 관계 선택
                for target in filtered_targets:
                    prev_value = relations.get(target, 0)
                    options = {"무관": 0, "같은 반": 1, "다른 반": -1}
                    reverse_options = {v: k for k, v in options.items()}
                    try:
                        init_index = list(options.values()).index(int(prev_value))
                    except:
                        init_index = 0  # 기본 "무관"
                    relation = st.selectbox(
                        f"{selected_main} ↔ {target}",
                        options=list(options.keys()),
                        index=init_index,
                        key=f"{selected_main}_{target}",
                    )
                    relations[target] = options[relation]

                # 관계 저장 버튼
                if st.button(f"💾 {selected_main}의 관계 저장"):
                    st.session_state['relationship_dict'][selected_main] = relations
                    print(st.session_state['relationship_dict'])
                    st.success(f"{selected_main}의 관계가 저장되었습니다.")

        # 관계 현황 보기
        st.markdown("#### 저장된 관계 요약 및 관리")
        if st.session_state['relationship_dict']:
            rel_df = pd.DataFrame.from_dict(st.session_state['relationship_dict'], orient='index').fillna(0)
            st.dataframe(rel_df, use_container_width=True)

            col1, col2 = st.columns([1, 1])

            # 특정 학생 관계 삭제
            st.markdown("##### 특정 학생 관계 삭제")
            delete_student = st.selectbox(
                "관계를 삭제할 학생 선택",
                options=["(선택 없음)"] + list(st.session_state['relationship_dict'].keys())
            )
            if delete_student != "(선택 없음)" and st.button("❌ 선택한 학생 관계 삭제"):
                del st.session_state['relationship_dict'][delete_student]
                st.warning(f"{delete_student}의 관계가 삭제되었습니다.")
            # 전체 초기화 버튼
            if st.button("🧹 모든 관계 초기화"):
                st.session_state['relationship_dict'].clear()
                st.warning("모든 관계가 초기화되었습니다.")

        else:
            st.info("아직 저장된 관계가 없습니다. 학생을 선택해 관계를 설정하세요.")
        
        st.divider()
        # 그룹 재배정 버튼
        if st.button("🔄 관계 기반 그룹 재배정 실행"):

            if 'group_assign_df' in st.session_state and 'relationship_dict' in st.session_state:
                # 동명이인 처리
                relationship_dict = st.session_state['relationship_dict'] # 관계 딕셔너리
                cleaned_rel_dict = {} # 관계 중 0(무관) 제거하기 위해
                for a, rels in relationship_dict.items():
                    new_rels = {b: v for b, v in rels.items() if v != 0}
                    if new_rels:
                        cleaned_rel_dict[a] = new_rels
                relationship_dict = cleaned_rel_dict
                group_assign_df = st.session_state['group_assign_df'] # 그룹 배정 데이터프레임
                # 동명이인 관계 자동 추가
                dup_df = group_assign_df[group_assign_df['동명이인_ID'].notna()]
                dup_groups = dup_df.groupby('동명이인_ID')['merge_key'].apply(list) # 동명이인 그룹 딕셔너리
                for _, same_name_keys in dup_groups.items():
                    for i in range(len(same_name_keys)):
                        for j in range(i+1, len(same_name_keys)):
                            a, b = same_name_keys[i], same_name_keys[j]
                            relationship_dict.setdefault(a, {})[b] = -1
                            relationship_dict.setdefault(b, {})[a] = -1
                st.session_state['relationship_dict'] = relationship_dict
                st.info("동명이인 관계가 다른반으로 추가되었습니다.")

                # 관계(relationship_dict) 텍스트 저장
                with open('relationship_dict.txt', 'w', encoding='utf-8') as f:
                    f.write(str(st.session_state['relationship_dict']))
                # 그룹 재배정 로직 실행
                selected_discrete_variable = st.session_state.get('selected_discrete_variable', [])
                selected_discrete_variable = ['성별_명렬표' if var == '성별' else var for var in selected_discrete_variable]
                selected_sort_variable_dict = st.session_state.get('selected_sort_variable_dict', {})

                from assign_relation_groups_optimal import (
                    find_relation_groups_minimal,
                    relation_groups_to_dict,
                    assign_relation_groups_optimal,
                    merge_optimal_assignments
                )
                from cost_group_move_v2 import cost_group_move

                # 케이스별 groupby 기준 설정
                sex_cls = st.session_state['sex_classification']
                subject_cls = st.session_state['subject_based_classification']

                if sex_cls == '분반' and subject_cls == '예':
                    groupby_cols = ['성별_명렬표', '선택과목']
                elif sex_cls == '분반' and subject_cls == '아니오':
                    groupby_cols = ['성별_명렬표']
                elif sex_cls == '합반' and subject_cls == '예':
                    groupby_cols = ['선택과목']
                elif sex_cls in ['남학교', '여학교'] and subject_cls == '예':
                    groupby_cols = ['선택과목']
                else:
                    groupby_cols = []  # 전체 단위

                # 그룹 단위별 관계 재배정 수행
                group_assign_df = st.session_state['group_assign_df']
                relationship_dict = st.session_state['relationship_dict']
                final_results = []

                if groupby_cols:
                    grouped_dfs = dict(tuple(group_assign_df.groupby(groupby_cols)))

                    for group_key, sub_df in grouped_dfs.items():
                        st.write(f"🔁 관계 기반 재배정 중... 그룹 단위: {group_key}, 학생 수: {len(sub_df)}")

                        # 관계딕셔너리 중 현재 그룹에 속한 학생만 필터링
                        valid_students = set(sub_df['merge_key'].values)
                        sub_rel_dict = {
                            a: {b: v for b, v in rels.items() if b in valid_students}
                            for a, rels in relationship_dict.items()
                            if a in valid_students
                        }

                        if not sub_rel_dict:
                            st.info(f"{group_key}: 관계 정보 없음, 기존 그룹 유지")
                            final_results.append(sub_df)
                            continue

                        # 관계 그룹 탐색 및 재배정
                        groups = find_relation_groups_minimal(
                            sub_rel_dict,
                            max_iter=10,
                            target_n_groups=sub_df['초기그룹'].nunique(),
                            verbose=False
                        )
                        ##! 관계 그룹이 그룹 수보다 많은 경우 오류 처리
                        if len(groups) > sub_df['초기그룹'].nunique():
                            st.error(f"{group_key}에서 관계 그룹 수가 그룹 수보다 많아 재배정 불가합니다.")
                            st.stop()
                        relationship_group_dict, relationship_group_df_dict = relation_groups_to_dict(groups, sub_df)
                        remaining_df, best_assignment, best_total_cost = assign_relation_groups_optimal(
                            sub_df, relationship_group_dict, relationship_group_df_dict, selected_discrete_variable
                        )
                        final_df = merge_optimal_assignments(remaining_df, best_assignment, relationship_group_df_dict)

                        # 그룹 내 균형 조정
                        final_df = cost_group_move(
                            50, 0.01, 100, 1,
                            final_df,
                            selected_discrete_variable,
                            selected_sort_variable_dict
                        )
                        final_results.append(final_df)

                else:
                    # 전체 단위로 관계 재배정
                    st.write("🔁 전체 단위로 관계 기반 재배정 중...")
                    groups = find_relation_groups_minimal(
                        relationship_dict,
                        max_iter=10,
                        target_n_groups=group_assign_df['초기그룹'].nunique(),
                        verbose=True
                    )
                    relationship_group_dict, relationship_group_df_dict = relation_groups_to_dict(groups, group_assign_df)
                    remaining_df, best_assignment, best_total_cost = assign_relation_groups_optimal(
                        group_assign_df, relationship_group_dict, relationship_group_df_dict, selected_discrete_variable
                    )
                    final_df = merge_optimal_assignments(remaining_df, best_assignment, relationship_group_df_dict)
                    final_df = cost_group_move(
                        50, 0.01, 100, 1,
                        final_df,
                        selected_discrete_variable,
                        selected_sort_variable_dict
                    )
                    final_results.append(final_df)

                # 결과 병합 및 저장
                final_group_assign_df = pd.concat(final_results, ignore_index=True)
                st.session_state['final_group_assign_df'] = final_group_assign_df
                final_group_assign_df.to_excel('final_group_assign_df.xlsx', index=False)
                st.success("🎉 관계 기반 그룹 재배정이 완료되었습니다.")
                # 관계 설정이 걸린 학생들 결과 확인
                st.subheader("관계 설정이 적용된 학생들 결과 확인")
                relationship_dict = st.session_state['relationship_dict']
                # 관계 설정이 걸린 학생 목록 추출
                related_students = set()
                for a, rels in relationship_dict.items():
                    related_students.add(a)
                    related_students.update(rels.keys())
                # 관계 걸린 학생만 필터링
                related_df = final_group_assign_df[final_group_assign_df['merge_key'].isin(related_students)]
                # 시각화 출력
                if related_df.empty:
                    st.info("현재 관계 설정이 걸려 있는 학생이 없습니다.")
                else:
                    st.write(f"총 {len(related_df)}명")
                    st.dataframe(related_df, use_container_width=True)
                    # 필요하다면 관계 컬럼 표시용 summary도 추가 가능
                    relation_summary = []
                    for a, rels in relationship_dict.items():
                        for b, v in rels.items():
                            relation_summary.append({"학생A": a, "학생B": b, "관계": "같은 반" if v==1 else "다른 반"})
                    relation_summary_df = pd.DataFrame(relation_summary)
                    relation_summary_df.to_excel('relation_summary_df.xlsx', index=False)
                    # relation_summary_df과 related_df의 그룹 배정 결과만 병합
                    relation_summary_df['학생A_그룹'] = relation_summary_df['학생A'].map(final_group_assign_df.set_index('merge_key')['초기그룹'])
                    relation_summary_df['학생B_그룹'] = relation_summary_df['학생B'].map(final_group_assign_df.set_index('merge_key')['초기그룹'])
                    with st.expander("🔍 관계 상세 보기"):
                        st.dataframe(relation_summary_df, use_container_width=True)

            else:
                st.warning("먼저 그룹 배정(group_assign_df)을 생성해주세요.")
    else:
        st.warning("먼저 그룹 배정(group_assign_df)을 생성해주세요.")

# [5] 분포 시각화
## 해당 소스의 대부분은 gpt 활용하여 작성됨
with tabs[5]:
    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader("📊 분류 후 평균 및 빈도 확인")
    st.write("집단 분류 후 각 집단의 평균 및 범주형 분포를 확인하고, 특정 학생을 이동시켜 변화를 시뮬레이션할 수 있습니다.")

    # 세션에서 데이터 가져오기
    if 'final_group_assign_df' not in st.session_state:
        st.warning("먼저 그룹 배정을 완료해주세요.")
        st.stop()
    
    df = st.session_state['final_group_assign_df']
    discrete_vars = st.session_state.get('selected_discrete_variable', [])
    discrete_vars = ['성별_명렬표' if var == '성별' else var for var in discrete_vars]
    continuous_vars = list(st.session_state.get('selected_sort_variable_dict', {}).keys())

    # -------------------------------------------------------------
    # ① 그룹별 이산형 변수 빈도 시각화
    # -------------------------------------------------------------
    st.markdown("### 🎯 그룹별 이산형 변수 분포")
    # 그룹별 크기 시각화
    group_size_df = (
        df.groupby('초기그룹')['merge_key']
        .count()
        .reset_index(name='학생 수')
        .sort_values('학생 수', ascending=False)
    )
    fig_size = px.bar(
        group_size_df,
        x='초기그룹',
        y='학생 수',
        color_discrete_sequence=["#4C78A8"],
        title="📊 그룹별 학생 수 분포",
        text='학생 수'
    )
    st.plotly_chart(fig_size, use_container_width=True)

    if not discrete_vars:
        st.info("선택한 이산형 변수가 없습니다.")
    else:
        selected_discrete = st.selectbox("이산형 변수 선택", discrete_vars)
        freq_df = (
            df.groupby(['초기그룹', selected_discrete])
              .size()
              .reset_index(name='빈도')
        )
        fig_cat = px.bar(
            freq_df, x='초기그룹', y='빈도', color=selected_discrete,
            barmode='stack', title=f"그룹별 {selected_discrete} 분포"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # -------------------------------------------------------------
    # ② 그룹별 연속형 변수 평균 시각화
    # -------------------------------------------------------------
    st.markdown("### 📈 그룹별 연속형 변수 평균")

    if not continuous_vars:
        st.info("연속형 변수가 없습니다.")
    else:
        selected_continuous = st.selectbox("연속형 변수 선택", continuous_vars)
        df_filtered = df[df['결시생'] == 0][['초기그룹', selected_continuous]]  # 결시생 제외
        mean_df = (
            df_filtered.groupby('초기그룹')[selected_continuous]
              .mean()
              .reset_index()
              .rename(columns={selected_continuous: '평균'})
        )
        mean_df['평균'] = mean_df['평균'].round(2)
        fig_mean = px.bar(
            mean_df, x='초기그룹', y='평균', title=f"그룹별 {selected_continuous} 평균 비교",
            text='평균'
        )
        st.plotly_chart(fig_mean, use_container_width=True)

# [6] 특정 교환 및 이동
## 해당 소스의 대부분은 gpt 활용하여 작성됨
with tabs[6]:
    import plotly.express as px
    import plotly.graph_objects as go
    move_swap_choice = st.selectbox('특정 학생을 이동할지 교환할지 선택해주세요.' , ['학생 이동', '학생 교환'], key='move_or_swap_choice')
    st.session_state['move_or_swap_choice'] = move_swap_choice
    if st.session_state['move_or_swap_choice'] == '학생 이동':
        st.markdown("#### 학생 이동 시뮬레이션")
        st.write("특정 학생을 다른 그룹으로 이동시켜 평균 및 빈도 변화 시뮬레이션을 수행할 수 있습니다.")
        final_sex_choice = st.session_state['sex_classification']
        final_subject_choice = st.session_state['subject_based_classification']
        df = st.session_state['final_group_assign_df']
        selected_discrete_variable = st.session_state.get('selected_discrete_variable', [])
        selected_discrete_variable = ['성별_명렬표' if var == '성별' else var for var in selected_discrete_variable]
        selected_sort_variable = st.session_state.get('selected_sort_variable_dict', {}).keys()
        # 이동할 그룹 선택
        source_group_no = st.selectbox("이동시킬 그룹 선택", list(map(int, sorted(df['초기그룹'].unique().tolist()))))
        # 해당 그룹의 학생들만 필터링
        source_group_df = df[df['초기그룹'] == source_group_no]
        st.dataframe(source_group_df[['이름_명렬표']+selected_discrete_variable+list(selected_sort_variable)], use_container_width=True)
        # 이동할 학생 선택
        selected_student = st.selectbox(f"{source_group_no}번 그룹에서 이동할 학생 선택", sorted(source_group_df['merge_key'].unique().tolist()))
        # 선택한 학생의 필터링된 그룹 리스트 생성
        ## 케이스별 groupby 기준 설정
        if final_sex_choice == '분반' and final_subject_choice == '예':
            groupby_cols = ['성별_명렬표', '선택과목']
        elif final_sex_choice == '분반' and final_subject_choice == '아니오':
            groupby_cols = ['성별_명렬표']
        elif final_sex_choice == '합반' and final_subject_choice == '예':
            groupby_cols = ['선택과목']
        elif final_sex_choice == '합반' and final_subject_choice == '아니오':
            groupby_cols = [] # 전체 그룹 대상으로
        elif final_sex_choice in ['남학교', '여학교'] and final_subject_choice == '예':
            groupby_cols = ['선택과목']
        elif final_sex_choice in ['남학교', '여학교'] and final_subject_choice == '아니오':
            groupby_cols = [] # 전체 그룹 대상이지만 남학교/여학교로 성별은 이미 하나임
        else:
            groupby_cols = []
        selected_row = df[df['merge_key'] == selected_student].iloc[0]
        if groupby_cols:
            # 선택한 학생이 속한 그룹 키
            group_keys = tuple(selected_row[col] for col in groupby_cols)
            # 전체 그룹
            all_group_df = df.groupby(groupby_cols)
            # 같은 그룹 키를 가진 학생들만 필터링
            candidate_groups_df = all_group_df.get_group(group_keys)
            # 선택한 학생이 속한 그룹 제외
            exception_candidate_groups = candidate_groups_df.loc[candidate_groups_df['초기그룹'] != selected_row['초기그룹'], '초기그룹'].unique().tolist()
            exception_candidate_groups = [int(g_n) for g_n in exception_candidate_groups]
        else:
            exception_candidate_groups = df.loc[df['초기그룹'] != selected_row['초기그룹'], '초기그룹'].unique().tolist()
            exception_candidate_groups = [int(g_n) for g_n in exception_candidate_groups]
        current_group = int(df.loc[df['merge_key'] == selected_student, '초기그룹'].values[0]) # 출발 그룹 번호
        target_group = st.selectbox("이동할 대상 그룹 선택", sorted(exception_candidate_groups)) # 도착할 그룹 선택 -> 도착 그룹 번호
        # 가상 이동 수행
        sim_df = df.copy(deep=True)
        sim_df.loc[sim_df['merge_key'] == selected_student, '초기그룹'] = target_group
        # 이동 전후 평균 비교
        before_mean = df.groupby('초기그룹')[selected_continuous].mean().reset_index().rename(columns={selected_continuous: '이동 전'})
        after_mean = sim_df.groupby('초기그룹')[selected_continuous].mean().reset_index().rename(columns={selected_continuous: '이동 후'})
        compare_mean = pd.merge(before_mean, after_mean, on='초기그룹', how='outer')
        st.markdown("#### 이동 전후 평균 비교")
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(x=compare_mean['초기그룹'], y=compare_mean['이동 전'], name='이동 전'))
        fig_compare.add_trace(go.Bar(x=compare_mean['초기그룹'], y=compare_mean['이동 후'], name='이동 후'))
        fig_compare.update_layout(barmode='group', title=f"이동 전후 {selected_continuous} 평균 변화")
        st.plotly_chart(fig_compare, use_container_width=True)
        # 이동 전후 이산형 변수 빈도 비교
        st.markdown("#### 이동 전후 이산형 분포 비교")
        highlight_groups = [current_group, target_group]  # 강조할 그룹
        if discrete_vars:
            st.markdown("#### 🎯 이동 전후 이산형 변수별 변화")
            for selected_discrete_for_sim in discrete_vars:
                before_freq = (
                    df.groupby(['초기그룹', selected_discrete_for_sim])
                    .size().reset_index(name='이동 전')
                )
                after_freq = (
                    sim_df.groupby(['초기그룹', selected_discrete_for_sim])
                    .size().reset_index(name='이동 후')
                )
                freq_compare = pd.merge(before_freq, after_freq, on=['초기그룹', selected_discrete_for_sim], how='outer').fillna(0)

                # (a) 절대 빈도 비교
                freq_melted = freq_compare.melt(
                    id_vars=['초기그룹', selected_discrete_for_sim],
                    value_vars=['이동 전', '이동 후'],
                    var_name='상태', value_name='빈도'
                )

                fig_stacked = px.bar(
                    freq_melted,
                    x='초기그룹',
                    y='빈도',
                    color=selected_discrete_for_sim,
                    facet_col='상태',
                    barmode='stack',
                    text='빈도',
                    title=f"📊 {selected_discrete_for_sim} - 이동 전후 누적빈도 비교",
                    color_discrete_sequence=['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B']
                )
                # 🔥 강조 처리 (이동 출발 / 대상 그룹)
                for trace in fig_stacked.data:
                    opacities = [1.0 if x in highlight_groups else 0.3 for x in trace.x]
                    trace.marker.opacity = opacities

                fig_stacked.update_traces(texttemplate='%{text}', textposition='inside')
                fig_stacked.update_layout(
                    yaxis_title="빈도수",
                    xaxis_title="그룹",
                    legend_title=selected_discrete_for_sim,
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    margin=dict(t=50, b=40, l=40, r=40),
                    annotations=[
                        dict(
                            x=current_group,
                            y=0,
                            text="⬆ 이동 출발",
                            showarrow=False,
                            yshift=10,
                            font=dict(color="red", size=12)
                        ),
                        dict(
                            x=target_group,
                            y=0,
                            text="⬆ 이동 대상",
                            showarrow=False,
                            yshift=10,
                            font=dict(color="red", size=12)
                        )
                    ]
                )

                st.plotly_chart(fig_stacked, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 변경 적용"):
                st.session_state['final_group_assign_df'] = sim_df
                sim_df.to_excel('final_group_assign_df_수동이동적용.xlsx', index=False)
                st.success(f"학생 {selected_student}이(가) {current_group} → {target_group} 그룹으로 이동이 적용되었습니다.")

        with col2:
            if st.button("↩️ 변경 취소"):
                st.session_state['final_group_assign_df'] = df
                st.info("이동이 취소되고 원래 그룹 배정으로 복원되었습니다.")
    elif st.session_state['move_or_swap_choice'] == '학생 교환':
        st.markdown("#### 학생 이동 시뮬레이션")
        st.write("특정 학생을 다른 그룹으로 이동시켜 평균 및 빈도 변화 시뮬레이션을 수행할 수 있습니다.")
        final_sex_choice = st.session_state['sex_classification']
        final_subject_choice = st.session_state['subject_based_classification']
        df = st.session_state['final_group_assign_df']
        selected_discrete_variable = st.session_state.get('selected_discrete_variable', [])
        selected_discrete_variable = ['성별_명렬표' if var == '성별' else var for var in selected_discrete_variable]
        selected_sort_variable = st.session_state.get('selected_sort_variable_dict', {}).keys()
        # 이동할 그룹 선택
        source_group_no = st.selectbox("이동시킬 그룹 선택", list(map(int, sorted(df['초기그룹'].unique().tolist()))))
        # 해당 그룹의 학생들만 필터링
        source_group_df = df[df['초기그룹'] == source_group_no]
        st.dataframe(source_group_df[['이름_명렬표']+selected_discrete_variable+list(selected_sort_variable)], use_container_width=True)
        # 이동할 학생 선택
        selected_student = st.selectbox(f"{source_group_no}번 그룹에서 이동할 학생 선택", sorted(source_group_df['merge_key'].unique().tolist()))
        # 선택한 학생의 필터링된 그룹 리스트 생성
        ## 케이스별 groupby 기준 설정
        if final_sex_choice == '분반' and final_subject_choice == '예':
            groupby_cols = ['성별_명렬표', '선택과목']
        elif final_sex_choice == '분반' and final_subject_choice == '아니오':
            groupby_cols = ['성별_명렬표']
        elif final_sex_choice == '합반' and final_subject_choice == '예':
            groupby_cols = ['선택과목']
        elif final_sex_choice == '합반' and final_subject_choice == '아니오':
            groupby_cols = [] # 전체 그룹 대상으로
        elif final_sex_choice in ['남학교', '여학교'] and final_subject_choice == '예':
            groupby_cols = ['선택과목']
        elif final_sex_choice in ['남학교', '여학교'] and final_subject_choice == '아니오':
            groupby_cols = [] # 전체 그룹 대상이지만 남학교/여학교로 성별은 이미 하나임
        else:
            groupby_cols = []
        selected_row = df[df['merge_key'] == selected_student].iloc[0]
        if groupby_cols:
            # 선택한 학생이 속한 그룹 키
            group_keys = tuple(selected_row[col] for col in groupby_cols)
            # 전체 그룹
            all_group_df = df.groupby(groupby_cols)
            # 같은 그룹 키를 가진 학생들만 필터링
            candidate_groups_df = all_group_df.get_group(group_keys)
            # 선택한 학생이 속한 그룹 제외
            exception_candidate_groups = candidate_groups_df.loc[candidate_groups_df['초기그룹'] != selected_row['초기그룹'], '초기그룹'].unique().tolist()
            exception_candidate_groups = [int(g_n) for g_n in exception_candidate_groups]
        else:
            exception_candidate_groups = df.loc[df['초기그룹'] != selected_row['초기그룹'], '초기그룹'].unique().tolist()
            exception_candidate_groups = [int(g_n) for g_n in exception_candidate_groups]
        current_group = int(df.loc[df['merge_key'] == selected_student, '초기그룹'].values[0]) # 출발 그룹 번호
        target_group = st.selectbox("이동할 대상 그룹 선택", sorted(exception_candidate_groups)) # 도착할 그룹 선택 -> 도착 그룹 번호

    st.divider()
    st.markdown("#### 학생 교환 시뮬레이션")
    st.write("선택한 학생과 다른 그룹의 유사한 학생을 교환하여 평균 및 분포를 보완합니다.")
    # 유사도 판단용 변수
    selected_discrete_for_swap = st.multiselect("교환 유사도 판단용 이산형 변수 선택", discrete_vars)
    selected_continuous_for_swap = st.selectbox("교환 유사도 판단용 연속형 변수 선택", continuous_vars)
    st.session_state['move_swap_flag'] = False
    if st.button("교환 시뮬레이션 실행"):
        # 선택한 그룹에서 유사한 학생 탐색
        ## 선택한 그룹 전체
        target_df = df[df['초기그룹'] == target_group].copy()
        ## 선택한 학생의 이산형 정보와 동일한 학생 필터링
        filter_df = pd.DataFrame([selected_row[selected_discrete_for_swap].to_dict()]) # 필터링용 데이터프레임 생성
        filtered_df = target_df.merge(filter_df, on=selected_discrete_for_swap, how='inner') # 필터링용 데이터프레임과 선택한 그룹 전체와 이너 조인으로 이산형 필터링
        
        if filtered_df.empty:
            st.warning("교환할 유사한 학생이 없습니다.")
        else:
            # 연속형 변수 기준으로 교환했을 때 이상적인 평균에 가장 근접한 학생 선택
            ideal_continuous_mean = df[selected_continuous_for_swap].mean()
            opt_sim_pair_continuous_diffs = {}
            for idx, row in filtered_df.iterrows():
                # 선택한 학생과 교환 시뮬레이션
                opt_sim_df = df.copy(deep=True) # 원본 데이터프레임 복사
                # 최적의 교환 상대를 구하기 위해 가상의 교환 수행
                opt_sim_df.loc[opt_sim_df['merge_key'] == selected_student, '초기그룹'] = row['초기그룹'] # 선택한 학생을 대상 그룹으로 이동
                opt_sim_df.loc[opt_sim_df['merge_key'] == row['merge_key'], '초기그룹'] = current_group # 대상 그룹 중 한 학생을 선택한 학생의 그룹으로 이동
                new_mean = abs(ideal_continuous_mean - opt_sim_df.loc[opt_sim_df['초기그룹']==current_group, selected_continuous_for_swap].mean())+abs(ideal_continuous_mean - opt_sim_df.loc[opt_sim_df['초기그룹']==target_group, selected_continuous_for_swap].mean())
                opt_sim_pair_continuous_diffs[(selected_student, row['merge_key'])] = new_mean
            # new_mean 기준으로 최소값 선택
            best_pair = min(opt_sim_pair_continuous_diffs, key=lambda x: opt_sim_pair_continuous_diffs[x])
            best_cost = opt_sim_pair_continuous_diffs[best_pair]
            swap_candidate_row = filtered_df[filtered_df['merge_key'] == best_pair[1]].iloc[0] # best_pair[1]이 교환 대상 학생
            
            st.markdown("#### 👥 교환 대상 학생 정보")
            st.dataframe(selected_row.to_frame().T, use_container_width=True)
            st.dataframe(swap_candidate_row.to_frame().T, use_container_width=True)
            st.info(f"유사한 학생 탐색 완료: **{swap_candidate_row['merge_key']}**, "
                    f"이산형 = {swap_candidate_row[selected_discrete_for_swap].to_dict()}, "
                    f"연속형 = {selected_continuous_for_swap}")
            
            # 교환을 적용할지 말지 결정하기 위해 시뮬레이션
            sim_df = df.copy(deep=True)
            sim_df.loc[sim_df['merge_key'] == selected_student, '초기그룹'] = swap_candidate_row['초기그룹']
            sim_df.loc[sim_df['merge_key'] == swap_candidate_row['merge_key'], '초기그룹'] = current_group
            st.session_state['sim_df'] = sim_df # 시뮬레이션 데이터프레임 세션에 저장

            # (1) 교환 전후 연속형 평균 비교
            before_mean = (
                df.groupby('초기그룹')[selected_continuous_for_swap]
                .mean().reset_index().rename(columns={selected_continuous_for_swap: '교환 전'})
            )
            after_mean = (
                sim_df.groupby('초기그룹')[selected_continuous_for_swap]
                .mean().reset_index().rename(columns={selected_continuous_for_swap: '교환 후'})
            )
            compare_mean = pd.merge(before_mean, after_mean, on='초기그룹', how='outer')
            
            st.markdown("#### 📊 교환 전후 평균 비교")
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(x=compare_mean['초기그룹'], y=compare_mean['교환 전'], name='교환 전'))
            fig_compare.add_trace(go.Bar(x=compare_mean['초기그룹'], y=compare_mean['교환 후'], name='교환 후'))
            fig_compare.update_layout(
                barmode='group',
                title=f"교환 전후 {selected_continuous_for_swap} 평균 변화",
                yaxis_title="평균값"
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # (2) 교환 전후 이산형 변수 누적빈도 비교
            st.markdown("#### 🎯 교환 전후 이산형 변수별 누적빈도 변화")
            highlight_groups = [current_group, target_group]

            for selected_discrete_for_sim in discrete_vars:
                before_freq = (
                    df.groupby(['초기그룹', selected_discrete_for_sim])
                    .size().reset_index(name='교환 전')
                )
                after_freq = (
                    sim_df.groupby(['초기그룹', selected_discrete_for_sim])
                    .size().reset_index(name='교환 후')
                )
                freq_compare = pd.merge(
                    before_freq, after_freq,
                    on=['초기그룹', selected_discrete_for_sim],
                    how='outer'
                ).fillna(0)

                # long 형식으로 변환
                freq_melted = freq_compare.melt(
                    id_vars=['초기그룹', selected_discrete_for_sim],
                    value_vars=['교환 전', '교환 후'],
                    var_name='상태',
                    value_name='빈도'
                )

                # 누적 막대그래프 (facet으로 전/후 분리)
                fig_stacked = px.bar(
                    freq_melted,
                    x='초기그룹',
                    y='빈도',
                    color=selected_discrete_for_sim,
                    facet_col='상태',
                    barmode='stack',
                    text='빈도',
                    title=f"{selected_discrete_for_sim} - 교환 전후 누적빈도 비교",
                    color_discrete_sequence=['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B']
                )
                # 강조 처리
                for trace in fig_stacked.data:
                    opacities = [1.0 if x in highlight_groups else 0.3 for x in trace.x]
                    trace.marker.opacity = opacities 
                fig_stacked.update_traces(texttemplate='%{text}', textposition='inside')
                fig_stacked.update_layout(
                    yaxis_title="빈도수",
                    xaxis_title="그룹",
                    legend_title=selected_discrete_for_sim,
                    margin=dict(t=50, b=40, l=40, r=40),
                    annotations=[
                        dict(x=current_group, y=0, text="⬆ 교환 출발", showarrow=False,
                            yshift=10, font=dict(color="red", size=12)),
                        dict(x=target_group, y=0, text="⬆ 교환 대상", showarrow=False,
                            yshift=10, font=dict(color="red", size=12))
                    ]
                )
                st.plotly_chart(fig_stacked, use_container_width=True)

            # (3) 적용 버튼
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 교환 변경 적용"):
                    st.session_state['final_group_assign_df'] = st.session_state['sim_df']
                    st.session_state['final_group_assign_df'].to_excel('final_group_assign_df_수동교환적용.xlsx', index=False)
                    print('교환 적용된 데이터프레임 저장 완료')
                    st.success(f"학생 {selected_student} ↔ {swap_candidate_row['merge_key']} 교환이 적용되었습니다.")

            with col2:
                if st.button("↩️ 교환 변경 취소"):
                    st.session_state['final_group_assign_df'] = df

# [7] 배정 결과 내보내기
## 해당 소스의 대부분은 gpt 활용하여 작성됨
with tabs[7]:
    st.subheader("최종 배정 결과 내보내기")
    st.write("최종 그룹 배정 결과를 엑셀 파일로 내보낼 수 있습니다.")

    if 'final_group_assign_df' not in st.session_state:
        st.warning("먼저 그룹 배정을 완료해주세요.")
        st.stop()
    
    final_df = st.session_state['final_group_assign_df']
    # 그룹 번호순으로 나열
    ## 그룹 내 이름 가나다순 번호 부여
    ## 그룹 내 성별로 분류하여 번호 부여
    ## 전출학생 마지막 번호 부여
    ## 운동부 마지막 번호 부여

    # 엑셀 파일로 저장
    output_filename = '최종_그룹_배정_결과.xlsx'
    final_df.to_excel(output_filename, index=False)

    # 다운로드 버튼
    with open(output_filename, 'rb') as f:
        st.download_button(
            label="⬇️ 최종 그룹 배정 결과 다운로드",
            data=f,
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.success("최종 그룹 배정 결과가 엑셀 파일로 저장되었습니다.")


# streamlit run c:/Users/USER/group_classification/pipeline_v5.py
# streamlit run /Users/mac/insight_/group_classification/pipeline_v5.py