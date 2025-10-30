from turtle import pd


def compute_ideal_discrete_freq(init_grouped_df, selected_discrete_variable):
    """
    이상적인(ideal) 이산형 변수별 그룹 빈도수를 계산하는 함수

    Parameters
    ----------
    df : pandas.DataFrame
        학생 정보가 포함된 데이터프레임 (각 행은 1명에 해당)
    discrete_vars : list
        이산형 변수(범주형 변수) 이름의 리스트
        예: ['성별_명렬표', '상담 필요']
    group_col : str
        그룹을 나타내는 컬럼명 (예: '초기그룹')
        → 그룹 개수를 계산하기 위함

    Returns
    -------
    ideal_freq : dict
        이상적인 빈도 구조를 담은 딕셔너리
        구조 예시:
        {
            'population': {
                '성별_명렬표': {1: 17.5, 2: 17.5},
                '상담 필요': {0: 30.0, 1: 5.0}
            }
        }

    Notes
    -----
    - 각 이산형 변수별 전체 빈도를 계산한 뒤,
      그룹 수로 나누어 "그룹별 이상적인 빈도"를 구함.
    - 예를 들어 전체 남학생이 35명이고 그룹이 2개면,
      각 그룹의 이상적인 남학생 수는 35 / 2 = 17.5명.
    """

    try:
        # ✅ 그룹 개수 확인 (전체 그룹이 몇 개인지)
        groups = init_grouped_df['초기그룹'].unique()
        group_n = len(groups)

        # ✅ 결과 저장용 딕셔너리 초기화
        ideal_freq = {'population': {}}

        # ✅ 각 이산형 변수별 이상적인 빈도 계산
        for var in selected_discrete_variable:
            # 전체 데이터에서 각 범주 값의 개수(count) 계산
            total_counts = init_grouped_df[var].value_counts()

            # 그룹 수로 나누어 "이상적인 그룹당 빈도"로 변환
            ideal_freq['population'][var] = (total_counts / group_n).to_dict()
    except Exception as e:
        print("Error during compute_ideal_discrete_freq execution:", str(e))
        raise e
    return ideal_freq

def compute_group_discrete_freq(init_grouped_df, selected_discrete_variable):
    """
    각 그룹별 실제 이산형(범주형) 변수의 빈도 분포를 계산하는 함수

    Parameters
    ----------
    df : pandas.DataFrame
        학생 정보가 포함된 데이터프레임
        (각 행은 한 명의 학생, group_col은 해당 학생이 속한 그룹)
    discrete_vars : list
        그룹 균형을 맞추기 위해 고려할 이산형 변수 이름 리스트
        예: ['성별_명렬표', '상담 필요']
    group_col : str
        그룹 구분 컬럼명 (예: '초기그룹')

    Returns
    -------
    group_freq : dict
        각 그룹의 실제 범주별 빈도 분포를 담은 딕셔너리

        예시 구조:
        {
            0: {  # 그룹 0
                '성별_명렬표': {1: 10, 2: 8},
                '상담 필요': {0: 15, 1: 3}
            },
            1: {  # 그룹 1
                '성별_명렬표': {1: 12, 2: 6},
                '상담 필요': {0: 14, 1: 4}
            }
        }

    Notes
    -----
    - 각 그룹별(subset)로 데이터프레임을 나눈 뒤,
      이산형 변수별로 value_counts()를 계산합니다.
    - 즉, “현재 그룹 구성에서 실제로 각 범주 값이 몇 명씩 있는가”를 확인합니다.
    """

    try:
        # ✅ 고유한 그룹 목록 추출
        groups = init_grouped_df['초기그룹'].unique()

        # ✅ 결과 저장용 딕셔너리 초기화
        group_freq = {}

        # ✅ 그룹별 이산형 변수 빈도 계산
        for g in groups:
            # 해당 그룹에 속한 데이터만 추출
            subset = init_grouped_df[init_grouped_df['초기그룹'] == g]

            # 그룹 g의 결과 저장용 하위 딕셔너리 초기화
            group_freq[g] = {}

            # 각 이산형 변수별 value_counts 계산
            for var in selected_discrete_variable:
                group_freq[g][var] = subset[var].value_counts().to_dict()
    except Exception as e:
        print("Error during compute_group_discrete_freq execution:", str(e))
        raise e
    return group_freq

def compute_group_total_cost(ideal_freq, group_freq, selected_discrete_variable):
    import numpy as np
    """
    각 그룹의 총 불균형도(total cost)를 계산하는 함수

    Parameters
    ----------
    ideal_freq : dict
        이상적인 빈도 정보 (compute_ideal_discrete_freq()의 결과)
        예시:
        {
            'population': {
                '성별_명렬표': {1: 17.5, 2: 17.5},
                '상담 필요': {0: 30.0, 1: 5.0}
            }
        }

    group_freq : dict
        실제 그룹별 범주 빈도 정보 (compute_group_discrete_freq()의 결과)
        예시:
        {
            0: {'성별_명렬표': {1: 10, 2: 8}, '상담 필요': {0: 15, 1: 3}},
            1: {'성별_명렬표': {1: 12, 2: 6}, '상담 필요': {0: 14, 1: 4}},
            ...
        }

    selected_discrete_variable : list
        계산에 포함할 이산형 변수명 리스트
        예: ['성별_명렬표', '상담 필요']

    Returns
    -------
    group_total_cost : dict
        각 그룹의 총 불균형도(total cost) 딕셔너리
        예시:
        {
            0: 9.0,
            1: 6.5,
            2: 8.2
        }

    Notes
    -----
    - 각 그룹에 대해 모든 이산형 변수의 각 범주별로
      ideal_count와 actual_count의 차이(|diff|)를 계산하고,
      이를 전부 합산하여 그룹의 total cost를 구함.
    - 즉, 그룹의 total cost가 클수록 이상적인 분포와의 차이가 큼.
    """

    group_total_cost_square = {}
    group_total_cost = {}
    try:
        # ✅ 각 그룹별로 계산
        for g in group_freq.keys():
            total_cost_square = 0
            total_cost = 0

            # 각 이산형 변수별로 편차 계산
            for var in selected_discrete_variable:
                for key, ideal_count in ideal_freq['population'][var].items():
                    # 해당 범주의 실제 빈도를 가져오되 없으면 0으로 처리
                    actual_count = group_freq[g][var].get(key, 0)
                    diff = np.sign(ideal_count - actual_count)*(ideal_count - actual_count)**2 # 양수 : 이상치보다 부족, 음수 : 이상치보다 과잉
                    diff_square = (ideal_count - actual_count)**2 #! 제곱하여 편차가 커질 수록 더 큰 페널티 부여

                    # 편차 누적
                    total_cost_square += diff_square # 불균형도 계산
                    total_cost += diff # 방향성을 포함한 계산

            # 그룹별 총 불균형도 저장
            group_total_cost[g] = total_cost
            group_total_cost_square[g] = total_cost_square
    except Exception as e:
        print("Error during compute_group_total_cost execution:", str(e))
        raise e
    return group_total_cost, group_total_cost_square

def compute_group_diff_and_sign(ideal_freq, group_freq, selected_discrete_variable):
    """
    각 그룹별 이산형 변수의 차이(diff)와 방향(sign)을 계산하는 함수

    Parameters
    ----------
    ideal_freq : dict
        이상적인 빈도 정보 (compute_ideal_discrete_freq()의 결과)
        예시:
        {
            'population': {
                '성별_명렬표': {1: 17.5, 2: 17.5},
                '상담 필요': {0: 30.0, 1: 5.0}
            }
        }

    group_freq : dict
        실제 그룹별 범주 빈도 정보 (compute_group_discrete_freq()의 결과)
        예시:
        {
            0: {'성별_명렬표': {1: 10, 2: 8}, '상담 필요': {0: 15, 1: 3}},
            1: {'성별_명렬표': {1: 12, 2: 6}, '상담 필요': {0: 14, 1: 4}}
        }

    discrete_vars : list
        계산 대상 이산형 변수명 리스트
        예: ['성별_명렬표', '상담 필요']

    Returns
    -------
    group_diff_cost : dict
        각 그룹의 변수별 diff와 sign을 담은 딕셔너리
        구조 예시:
        {
            0: {
                '성별_명렬표_diff': {1: 7.5, 2: 9.5},
                '성별_명렬표_sign': {1: +1.0, 2: +1.0},
                '상담 필요_diff': {0: 15.0, 1: 2.0},
                '상담 필요_sign': {0: +1.0, 1: +1.0}
            },
            1: { ... },
            ...
        }

    Notes
    -----
    - diff = ideal_count - actual_count
      → 양수(+)면 ideal보다 부족, 음수(–)면 ideal보다 과잉.
    - sign = np.sign(diff)
      → +1 = 증가 필요, –1 = 감소 필요, 0 = 이상적.
    - swap 후보 탐색 시,
      sign이 반대인 그룹끼리 교환 가능성 탐색에 활용함.
    """
    import numpy as np

    group_diff_cost = {}
    try:
        for g in group_freq.keys():
            group_diff_cost[g] = {}

            for var in selected_discrete_variable:
                # diff, sign 각각을 별도 dict로 저장
                diff_dict = {}
                sign_dict = {}

                # 각 범주(key)별로 ideal 대비 actual 차이 계산
                for key, ideal_count in ideal_freq['population'][var].items():
                    actual_count = group_freq[g][var].get(key, 0)
                    diff = ideal_count - actual_count
                    diff_dict[key] = diff
                    sign_dict[key] = np.sign(diff) # 이상치보다 작으면 +1, 크면 -1, 같으면 0

                # 변수별 결과를 group_diff_cost에 저장
                group_diff_cost[g][f'{var}_diff'] = diff_dict
                group_diff_cost[g][f'{var}_sign'] = sign_dict
    except Exception as e:
        print("Error during compute_group_diff_and_sign execution:", str(e))
        raise e
    return group_diff_cost

# 연속형 비용함수 설정
def compute_continuous_cost(init_grouped_df, s_row, t_row, selected_sort_variable): # selected_sort_variable : 단일 변수(1순위)
    import numpy as np
    try:
        pop_mean = float(init_grouped_df[selected_sort_variable].mean())

        source_group_df = init_grouped_df[init_grouped_df['초기그룹'] == s_row['초기그룹']]
        target_group_df = init_grouped_df[init_grouped_df['초기그룹'] == t_row['초기그룹']]
        
        # 이동 전 이상 평균으로 부터 거리 계산
        before_dist = (
            np.abs(source_group_df[selected_sort_variable].mean() - pop_mean) +
            np.abs(target_group_df[selected_sort_variable].mean() - pop_mean)
        )

        # 시뮬레이션 이동
        simulated_df = init_grouped_df.copy()
        simulated_df.loc[s_row.name, '초기그룹'] = t_row['초기그룹']

        # 이동 후 그룹별 평균 계산
        group_mean_after = simulated_df.groupby('초기그룹')[selected_sort_variable].mean()

        # 이동 후 이상 평균으로 부터 거리 계산
        after_dist = ( # 값이 작을수록 개선
            np.abs(group_mean_after.loc[t_row['초기그룹']] - pop_mean) +
            np.abs(group_mean_after.loc[s_row['초기그룹']] - pop_mean)
        )

        continuous_cost = before_dist - after_dist # 값이 클수록 개선
    except Exception as e:
        print("Error during compute_continuous_cost execution:", str(e))
        raise e
    return continuous_cost

def compute_discrete_cost(group_diff_cost, s_row, t_row, selected_discrete_variable):
    """
    이산형 불균형 비용 함수를 계산하는 함수 (학생 단위 이동 판단 개선 버전)

    Parameters
    ----------
    group_diff_cost : dict
        그룹별 이산형 변수별 diff(=ideal - current) 및 sign 정보를 담은 딕셔너리
    s_row : pandas.Series
        이동 대상 학생 (출발 그룹 정보 포함)
    t_row : pandas.Series
        이동 대상 학생 (도착 그룹 정보 포함)
    selected_discrete_variable : list
        고려할 이산형 변수 목록 (예: ['성별_명렬표', '상담 필요'])

    Returns
    -------
    discrete_cost_change : float
        이동으로 인한 비용 변화량 (값이 작을수록 개선)
        이동이 불가능한 경우 np.inf (무한대 비용) 반환
    """
    import copy
    import numpy as np
    import pandas as pd
    try:
        # deepcopy로 원본 안전하게 복사
        new_group_diff = copy.deepcopy(group_diff_cost)

        source_group = s_row['초기그룹']
        target_group = t_row['초기그룹']

        # -------------------------------------------------
        # 학생 단위 이동 허용 여부 판단
        # -------------------------------------------------
        move_allowed = False  # 기본값: 이동 불가
        for var in selected_discrete_variable:
            source_cat = s_row[var]
            #print(f"변수: {var}, 출발 그룹: {source_group}, 도착 그룹: {target_group}, 카테고리: {source_cat}")
            if pd.isna(source_cat):
                #print(f"출발 학생({s_row['merge_key']})의 해당 변수 {var} 값이 NaN이므로 이동 불가 처리")
                return -np.inf  # NaN 값인 경우 즉시 이동 불가 처리
            sign_val = group_diff_cost[source_group][f'{var}_sign'][source_cat] #! s_row의 var가 nan인 경우가 존재할 수 있음

            if sign_val > 0:  # 부족 상태면 즉시 이동 금지
                return -np.inf
            elif sign_val < 0:  # 과잉 상태 존재 시 이동 고려
                move_allowed = True

        # 모든 변수에서 균형(0)이거나 부족(+)이면 이동 금지
        if not move_allowed:
            return -np.inf

        # -------------------------------------------------
        # 이동 시나리오에 따른 비용 계산
        # -------------------------------------------------
        before_cost, after_cost = 0, 0

        for var in selected_discrete_variable:
            source_cat = s_row[var]

            # diff 값 갱신: source → target
            new_group_diff[source_group][f'{var}_diff'][source_cat] += 1
            new_group_diff[target_group][f'{var}_diff'][source_cat] -= 1

            # 그룹별 cost 계산 함수 정의
            def group_cost(g):
                return sum(abs(diff_val) for diff_val in new_group_diff[g][f'{var}_diff'].values())

            # 이동 후 비용
            after_cost += group_cost(source_group) + group_cost(target_group)

            # 이동 전 비용
            for g in [source_group, target_group]:
                before_cost += sum(abs(v) for v in group_diff_cost[g][f'{var}_diff'].values())

        # -------------------------------------------------
        # 최종 비용 변화량 계산
        # -------------------------------------------------
        discrete_cost_change = abs(after_cost - before_cost)
    except Exception as e:
        print("Error during compute_discrete_cost execution:", str(e))
        raise e
    return discrete_cost_change

def compute_size_cost(init_grouped_df, s_row, t_row):
    """
    이동 전후 그룹 크기 균형 개선 정도를 계산하는 함수

    Parameters
    ----------
    init_grouped_df : pd.DataFrame
        전체 학생 데이터프레임 (초기 그룹 정보 포함)
    s_row, t_row : pd.Series
        출발 그룹과 도착 그룹의 학생 행

    Returns
    -------
    delta_balance : float
        이동 후 크기 불균형이 줄어든 정도 (양수면 개선, 음수면 악화)
    """
    import numpy as np

    # 그룹 크기 계산
    group_sizes = init_grouped_df.groupby('초기그룹').size()
    group_mean_size = group_sizes.mean()

    source_group = s_row['초기그룹']
    target_group = t_row['초기그룹']

    # 이동 전 불균형도 (절대값 기준)
    before = abs(group_sizes[source_group] - group_mean_size) + abs(group_sizes[target_group] - group_mean_size)

    # 이동 후 예상 크기 (이동 시뮬레이션)
    after_source_size = group_sizes[source_group] - 1
    after_target_size = group_sizes[target_group] + 1

    after = abs(after_source_size - group_mean_size) + abs(after_target_size - group_mean_size)

    # 개선량 계산 (양수면 개선)
    delta_balance = before - after
    return delta_balance

def cost_group_move(max_iter, tolerance, w_discrete, w_continuous, init_grouped_df, selected_discrete_variable, selected_sort_variable_dict):
    import numpy as np
    print("cost_group_move 인자 확인메롱")
    print(f"max_iter: {max_iter}, tolerance: {tolerance}, w_discrete: {w_discrete}, w_continuous: {w_continuous}")
    print(f"selected_discrete_variable: {selected_discrete_variable}, selected_sort_variable: {selected_sort_variable_dict}")
    try:
        # 그룹이 하나만 있는 경우 바로 반환
        if init_grouped_df['초기그룹'].nunique() == 1:
            print("그룹이 하나만 있어 바로 반환합니다.")
            return init_grouped_df
        # -------------------------------------------------
        # 이산형 변수를 선택하지 않은 경우
        # -------------------------------------------------
        if not selected_discrete_variable :
            print("이산형 변수가 선택되지 않아 이산형 비용 함수 계산을 건너뜁니다.")
            selected_sort_variable = list(selected_sort_variable_dict.keys())[-1] # selected_sort_variable_dict : 딕셔너리 형태로 입력됨 (오른쪽으로 갈수록 우선순위 높음)

            # 초기 그룹 설정에서의 비용 계산(갱신용), 연속형 변수와 그룹 크기만 반영
            prev_group_mean = [init_grouped_df[init_grouped_df['초기그룹'] == g][selected_sort_variable].mean() for g in init_grouped_df['초기그룹'].unique()]
            prev_pop_mean = init_grouped_df[selected_sort_variable].mean()
            prev_diff_cost = sum([abs(gm - prev_pop_mean) for gm in prev_group_mean]) # 그룹 별 평균과 전체 평균의 차이의 절대값 합 -> 값이 클수록 불균형
            prev_mean_size = init_grouped_df.groupby('초기그룹').size().mean() # 이상적인 그룹 크기
            prev_size_cost = sum([abs(len(init_grouped_df[init_grouped_df['초기그룹'] == g]) - prev_mean_size) for g in init_grouped_df['초기그룹'].unique()]) # 그룹 크기 불균형도 -> 값이 클수록 불균형
            prev_total_cost = prev_diff_cost + 10 * prev_size_cost
            # 이동 기록
            move_history = []

            for iter_num in range(max_iter):
                print(f"\n======= Iteration {iter_num+1} =======")
                # 연속형 변수는 1순위 변수로만 비용함수 계산
                # 이동 전 연속형 변수 출력
                print("##############################")
                print("이동 전 그룹 연속형 변수 평균:")
                print(init_grouped_df.groupby('초기그룹')[selected_sort_variable].mean())
                group_sizes = init_grouped_df.groupby('초기그룹').size()
                print(group_sizes)
                group_mean_size = group_sizes.mean()
                # 이상적인 평균치 산출
                pop_mean = float(init_grouped_df[selected_sort_variable].mean())
                # 그룹별 총 불균형도 계산 <- 여기에 연속형뿐만 아니라 그룹 n크기도 반영해야됨.
                group_abs_mean_cost = {}
                groups = init_grouped_df['초기그룹'].unique()
                group_n_diff = {}
                group_tanh = {} # softsign 또는 tahn 함수를 활용해서 방향성과 그 정도를 반영하는 방법도 고려해야함.
                for g in groups:
                    group_df = init_grouped_df[init_grouped_df['초기그룹'] == g]
                    group_mean = group_df[selected_sort_variable].mean()
                    group_mean_diff = abs(group_mean - pop_mean) # 절대값
                    group_n_diff[g] = len(group_df) - group_mean_size # 그룹의 크기를 방향성 있도록 반영
                    group_abs_mean_cost[g] = group_mean_diff # 그룹의 평균은 절대값으로 반영
                    group_tanh[g] = np.tanh(group_mean_size - len(group_df)) # 이상적 그룹 빈도보다 크면 -, 작으면 +, 같으면 0
                # group_n_diff 높은 그룹 선택
                print(group_n_diff)
                source_group_idx = max(group_n_diff, key=group_n_diff.get) # 평균적인 그룹 크기보다 학생 수가 가장 많은 그룹 선택, 이상치와 같은 경우 0인 그룹도 선택됨
                print("이동할 그룹:", source_group_idx)
                # 이동할 그룹과 n수가 반대 방향을 가지는 그룹 중 편차 큰 그룹 선택
                target_tanh = -group_tanh[source_group_idx] # group_tanh 값이 0인 경우도 고려해야함.
                print("타깃 그룹의 tanh 방향:", target_tanh)
                match_group_tanh_idx = [g for g in group_tanh if np.sign(group_tanh[g]) == np.sign(target_tanh)] # 이동할 그룹과 반대 방향을 가지는 그룹들 <- 여기서 이동할 그룹은 제외됨
                print("타깃 그룹 후보:", match_group_tanh_idx)
                # 만약 match_group_tanh_idx 가 비어있다면 그룹 크기가 작은 순으로 선택
                if not match_group_tanh_idx:
                    print("타깃 그룹 후보가 없어 그룹 크기가 작은 순으로 선택합니다.")
                    group_n = {g: group_sizes[g] for g in groups if g != source_group_idx}
                    match_group_tanh_idx = sorted(group_n, key=group_n.get)[:2] # 그룹 크기가 작은 상위 2개 그룹 선택
                # 평균차 뿐만 아니라 크기 차도 반영하여 선택 가능하도록 복합 점수 로직 추가
                match_group_score = {}
                for g in match_group_tanh_idx:
                    match_group_mean_diff = group_abs_mean_cost[g]
                    match_group_n_diff = abs(group_sizes[g] - group_mean_size)
                    match_group_score[g] = match_group_mean_diff + 10 * match_group_n_diff # 그룹 크기가 더 중요하도록 설정 + 스케일 차이
                print("타깃 그룹 후보의 비용 점수:", match_group_score)
                match_group_idx = sorted(match_group_score, key=match_group_score.get, reverse=True)[:3] # 반대 방향 그룹 중 평균 편차 큰 상위 3개 그룹 선택
                print("매칭 그룹:", set(match_group_idx))
                print("##############################")
                source_group_df = init_grouped_df[init_grouped_df['초기그룹']==source_group_idx]
                target_group_df = init_grouped_df[init_grouped_df['초기그룹'].isin(match_group_idx)] # match_group_idx : list 형태
                # 모든 쌍에 대해 비용 계산을 담은 딕셔너리
                pair_costs = {}
                for s_idx, s_row in source_group_df.iterrows():
                    for t_idx, t_row in target_group_df.iterrows():
                        # 여기서 s_row와 t_row를 비교하여 교환 가능성을 탐색
                        #! 그룹 고정된 학생이면 교환 계산에서 생략
                        if s_row.get('그룹고정', False) or t_row.get('그룹고정', False):
                            continue

                        # 혹시 모르니 그룹이 같으면 생략 조건 설정 
                        if s_row['초기그룹'] == t_row['초기그룹']:
                            continue
                        #elif len(source_group_df) <= len(target_group_df): # 이동하는 그룹의 수보다 도착 그룹의 수가 더 많거나 같으면 비용계산 X -> 앞에서 적은 그룹은 많은 그룹 이동을 방지하도록 설정했으니 이 조건은 불필요
                        #    continue
                        elif len(source_group_df) == len(target_group_df):
                            print("출발 그룹과 도착 그룹의 학생 수가 동일하여 연속형 변수 비용만 계산")
                            cont_cost = compute_continuous_cost(init_grouped_df, s_row, t_row, selected_sort_variable) # 연속형 변수 비용 계산 (값이 클수록 개선)
                            size_cost = 0
                        else:
                        # 연속형 변수 비용 계산
                            cont_cost = compute_continuous_cost(init_grouped_df, s_row, t_row, selected_sort_variable) # 연속형 변수 비용 계산 (값이 클수록 개선)
                            # size_penalty = abs(group_sizes[s_row['초기그룹']] - group_mean_size) + abs(group_sizes[t_row['초기그룹']] - group_mean_size) # 그룹 크기 패널티 계산 (불균형, 특이한 경우 불균형 해소가 불가할 수 있음)
                            size_cost = compute_size_cost(init_grouped_df, s_row, t_row) # 그룹 크기 패널티 계산 (값이 클수록 개선)
                        # 총 비용 계산
                        total_cost = w_continuous * cont_cost + 100 * size_cost
                        print(f"쌍 ({s_idx}, {t_idx}) 연속형 비용: {cont_cost}, 크기 패널티: {size_cost}, 총 비용: {total_cost}")
                        pair_costs[(s_idx, t_idx)] = total_cost
                # 최대 비용 쌍 선택
                best_pair = max(pair_costs, key=lambda x: pair_costs[x])
                best_cost = pair_costs[best_pair]
                print("최고 효율 쌍:", best_pair)
                print("비용:", best_cost) # 먼가 이상한데
                # 실제 그룹 이동
                idx_s, idx_t = best_pair
                # 그룹 실제 이동하는거 확인
                print("++이동하는 정보++")
                print(f"그룹 이동 완료: {init_grouped_df.loc[idx_s, '초기그룹']} -> {init_grouped_df.loc[idx_t, '초기그룹']}")
                print("이동 학생 정보:")
                print(init_grouped_df.loc[idx_s, selected_discrete_variable])
                print("+++++++++++++++")
                init_grouped_df.loc[idx_s, '초기그룹'] = init_grouped_df.loc[idx_t, '초기그룹']
                # 이동 후 비용 계산
                new_group_mean = init_grouped_df.groupby('초기그룹')[selected_sort_variable].mean()
                new_pop_mean = init_grouped_df[selected_sort_variable].mean()
                new_diff_cost = sum([abs(gm - new_pop_mean) for gm in new_group_mean]) # 그룹 별 평균과 전체 평균의 차이의 절대값 합
                new_size_cost = sum([abs(len(init_grouped_df[init_grouped_df['초기그룹'] == g]) - prev_mean_size) for g in init_grouped_df['초기그룹'].unique()]) # 그룹 크기 불균형도
                new_total_cost = new_diff_cost + 10 * new_size_cost
                print(f"이전 총 비용: {prev_total_cost}, 새로운 총 비용: {new_total_cost}")
                # 비용 개선폭 계산
                improvement = abs(prev_total_cost - new_total_cost)
                print(f"비용 개선폭: {improvement:.6f}")
                # 이동 후 그룹별 평균과 분포 출력 (확인용)
                group_freq = compute_group_discrete_freq(init_grouped_df, selected_discrete_variable)
                print("Group Frequency:")
                print(group_freq)
                # 개선폭 기준 만족 시 중단
                if improvement < tolerance:
                    print("개선 폭이 작아 중단합니다.")
                    break
                prev_total_cost = new_total_cost


        # -------------------------------------------------
        # 이산형, 연속형 변수를 모두 선택한 경우
        # -------------------------------------------------   
        else :
            print("이산형, 연속형 변수 모두 선택되어 비용 함수 계산을 진행합니다.")
            selected_sort_variable = list(selected_sort_variable_dict.keys())[-1] # selected_sort_variable_dict : 딕셔너리 형태로 입력됨 (오른쪽으로 갈수록 우선순위 높음)
            print(f"선택된 연속형 변수: {selected_sort_variable}")
            ideal_freq = compute_ideal_discrete_freq(init_grouped_df, selected_discrete_variable)
            print("이상적인 이산형 변수 빈도수:")
            print(ideal_freq)
            # 이동 기록
            move_history = []

            # 이전 총 비용 계산, 연속형, 이산형 모두 반영
            prev_group_mean = [init_grouped_df[init_grouped_df['초기그룹'] == g][selected_sort_variable].mean() for g in init_grouped_df['초기그룹'].unique()]
            print("이전 그룹별 연속형 변수 평균:")
            print(prev_group_mean)
            prev_pop_mean = init_grouped_df[selected_sort_variable].mean()
            print(f"전체 연속형 변수 평균: {prev_pop_mean}")
            prev_diff_cost = sum([abs(gm - prev_pop_mean) for gm in prev_group_mean]) # 그룹 별 평균과 전체 평균의 차이의 절대값 합 -> 값이 클수록 불균형
            print(f"이전 연속형 불균형도: {prev_diff_cost}")
            group_freq = compute_group_discrete_freq(init_grouped_df, selected_discrete_variable) # 각 그룹별 이산형 실제 빈도
            print("이전 그룹별 이산형 변수 빈도수:")
            print(group_freq)
            _, group_total_cost_square = compute_group_total_cost(ideal_freq, group_freq, selected_discrete_variable) # 각 그룹별 이산형 총 불균형도
            #prev_disc_cost = {k: abs(v) for k, v in prev_disc_cost.items()} # 절대값 변환
            print("이전 그룹별 이산형 불균형도:")
            print(group_total_cost_square)
            prev_disc_total_cost = sum(group_total_cost_square.values())
            print(f"이전 이산형 총 불균형도: {prev_disc_total_cost}")
            print(f"이전 연속형 비용: {prev_diff_cost}, 이전 이산형 비용: {prev_disc_total_cost}")
            print(w_discrete)
            prev_total_cost = prev_diff_cost + 10 * prev_disc_total_cost
            print(f"이전 총 비용: {prev_total_cost}")

            for iter_num in range(max_iter):
                print(f"\n======= Iteration {iter_num+1} =======")
                # 그룹별 총 불균형도 및 diff, sign 계산
                group_freq = compute_group_discrete_freq(init_grouped_df, selected_discrete_variable) # 각 그룹별 이산형 실제 빈도
                group_total_cost, group_total_cost_square = compute_group_total_cost(ideal_freq, group_freq, selected_discrete_variable) # 각 그룹별 이산형 총 불균형도
                group_diff_cost = compute_group_diff_and_sign(ideal_freq, group_freq, selected_discrete_variable) # 각 그룹별 이산형 diff, sign
                groups = init_grouped_df['초기그룹'].unique()

                # 이동 전 이상현 빈도 출력
                print("##############################")
                print("이전 그룹 이상적 빈도:")
                print(ideal_freq)
                print("이전 그룹 빈도 분포:")
                print(group_freq)
                print("그룹별 총 불균형도:")
                print(group_total_cost)
                
                # 불균형이 가장 큰 그룹 탐색
                source_group_idx = min(group_total_cost, key=group_total_cost.get) # 이상치보다 많은 그룹에서 이동을 해야함. 그렇기 때문에 해당 값이 음수(이상-실제)일수록 이동 우선순위가 높음
                print("##############################")
                print(f"최고 편차 그룹 : {source_group_idx}")
                print("이산변수 차이 및 부호 값:")
                print(group_diff_cost[source_group_idx])

                # 타깃 그룹 후보 제한 추가
                # source_group와 반대 방향을 지니고 있는 그룹들로 제한
                print("##############################")
                reversed_target = {}
                for var in selected_discrete_variable:
                    reversed_target[var] = {}
                    for key, value in group_diff_cost[source_group_idx][f'{var}_sign'].items():
                        reversed_target[var][key] = -value
                print(reversed_target) # 이렇게 제한하니까 이산형 변수가 늘어날수록 매칭 그룹이 없는 경우가 발생함.
                # 이거와 일치하는 방향의 그룹
                match_threshold = 0.6 # 일치하는 그룹이 없는 경우를 대비한 임계값
                match_group_idx = []
                for g in groups:
                    if g == source_group_idx:
                        continue
                    total_keys = 0
                    match_count = 0
                    for var in selected_discrete_variable:
                        for key, target_sign in reversed_target[var].items():
                            total_keys += 1
                            current_sign = group_diff_cost[g][f'{var}_sign'].get(key, 0)
                            if current_sign == target_sign:
                                match_count += 1
                    if total_keys == 0:
                        continue
                    match_ratio = match_count / total_keys
                    # 임계값 이상인 경우 매칭 그룹으로 간주
                    if match_ratio >= match_threshold:
                        match_group_idx.append(g)
                print("매칭 그룹 후보:", match_group_idx)
                if match_group_idx == []:
                    print("일치하는 매칭 그룹이 없습니다. 그룹 전체로 진행됩니다.")
                    # 모든 그룹을 대상으로 탐색할 경우 시간 비용이 너무 큼
                    # match_group_idx = [g for g in groups if g != source_group_idx]
                    match_group_idx = sorted(group_total_cost, key=group_total_cost.get, reverse=True)[:2] # group_total_cost(이상-실제)값이기 때문에 이상치보다 작은 그룹 중 편차 큰 상위 2개 그룹 선택

                print("매칭 그룹:", set(match_group_idx))
                print("##############################")
                source_group_df = init_grouped_df[init_grouped_df['초기그룹']==source_group_idx]
                target_group_df = init_grouped_df[init_grouped_df['초기그룹'].isin(match_group_idx)] # match_group_idx : list 형태

                # 연속형 변수는 1순위 변수로만 비용함수 계산
                selected_sort_variable = list(selected_sort_variable_dict.keys())[-1] # selected_sort_variable_dict : 딕셔너리 형태로 입력됨 (오른쪽으로 갈수록 우선순위 높음)
                # 모든 쌍에 대해 비용 계산을 담은 딕셔너리
                pair_costs = {}
                for s_idx, s_row in source_group_df.iterrows():
                    for t_idx, t_row in target_group_df.iterrows():
                        # 여기서 s_row와 t_row를 비교하여 교환 가능성을 탐색
                        #! 그룹 고정된 학생이면 교환 계산에서 생략
                        if s_row.get('그룹고정', False) or t_row.get('그룹고정', False):
                            continue
                        
                        # 혹시 모르니 조건 설정
                        if s_row['초기그룹'] == t_row['초기그룹']:
                            continue
                        # 이산형 변수 비용 계산
                        disc_cost = compute_discrete_cost(group_diff_cost, s_row, t_row, selected_discrete_variable) # 이산형 변수 비용 계산 (값이 클수록 개선)
                        # 연속형 변수 비용 계산
                        cont_cost = compute_continuous_cost(init_grouped_df, s_row, t_row, selected_sort_variable) # 연속형 변수 비용 계산 (값이 클수록 개선)
                        # 총 비용 계산
                        total_cost = w_discrete * disc_cost + w_continuous * cont_cost

                        pair_costs[(s_idx, t_idx)] = total_cost
                # 최대 비용 쌍 선택
                best_pair = max(pair_costs, key=lambda x: pair_costs[x])
                best_cost = pair_costs[best_pair]
                print("최고 효율 쌍:", best_pair)
                print("비용:", best_cost)
                idx_s, idx_t = best_pair
                # 실제 이동 전 기록 저장
                # move_key = (s_idx, init_grouped_df.loc[idx_s, '초기그룹'], t_idx, init_grouped_df.loc[idx_t, '초기그룹'])
                # if move_key in move_history:
                #     print("이미 이동한 쌍이라 중단합니다.")
                #     break
                # 실제 그룹 이동
                print("++이동하는 정보++")
                print(f"그룹 이동 완료: {init_grouped_df.loc[idx_s, '초기그룹']} -> {init_grouped_df.loc[idx_t, '초기그룹']}")
                print("이동 학생 정보:")
                print(init_grouped_df.loc[idx_s, selected_discrete_variable])
                print("+++++++++++++++")
                init_grouped_df.loc[idx_s, '초기그룹'] = init_grouped_df.loc[idx_t, '초기그룹']
                # 이동 기록 업데이트
                # move_history.append(move_key)
                # 이동 후 비용 계산
                new_group_mean = init_grouped_df.groupby('초기그룹')[selected_sort_variable].mean()
                new_pop_mean = init_grouped_df[selected_sort_variable].mean()
                new_diff_cost = sum([abs(gm - new_pop_mean) for gm in new_group_mean]) # 그룹 별 평균과 전체 평균의 차이의 절대값 합
                group_freq = compute_group_discrete_freq(init_grouped_df, selected_discrete_variable)
                _, new_disc_cost_square = compute_group_total_cost(ideal_freq, group_freq, selected_discrete_variable) # 각 그룹별 이산형 총 불균형도
                #new_disc_cost = {k: abs(v) for k, v in new_disc_cost.items()} # 절대값 변환
                new_disc_total_cost = sum(new_disc_cost_square.values())
                new_total_cost = new_diff_cost + 10 * new_disc_total_cost
                print(f"이전 총 비용: {prev_total_cost}, 새로운 총 비용: {new_total_cost}")
                # 비용 개선폭 계산
                improvement = abs(prev_total_cost - new_total_cost)
                print(f"비용 개선폭: {improvement:.6f}")
                # 이동 후 그룹별 평균과 분포 출력
                group_freq = compute_group_discrete_freq(init_grouped_df, selected_discrete_variable)
                print("Group Frequency:")
                print(group_freq)
                if improvement < tolerance:
                    print("개선 폭이 작아 중단합니다.")
                    break
                prev_total_cost = new_total_cost
    except Exception as e:
        print("Error during cost_group_move execution:", str(e))
        raise e
    return init_grouped_df