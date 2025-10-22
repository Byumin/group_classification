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

    import pandas as pd

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

    import pandas as pd

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

    return group_freq

def compute_group_total_cost(ideal_freq, group_freq, selected_discrete_variable):
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

    group_total_cost = {}

    # ✅ 각 그룹별로 계산
    for g in group_freq.keys():
        total_cost = 0

        # 각 이산형 변수별로 편차 계산
        for var in selected_discrete_variable:
            for key, ideal_count in ideal_freq['population'][var].items():
                # 해당 범주의 실제 빈도를 가져오되 없으면 0으로 처리
                actual_count = group_freq[g][var].get(key, 0)
                diff = ideal_count - actual_count

                # 절댓값으로 편차 누적
                total_cost += diff

        # 그룹별 총 불균형도 저장
        group_total_cost[g] = total_cost

    return group_total_cost

import numpy as np

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

    group_diff_cost = {}

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
                sign_dict[key] = np.sign(diff)

            # 변수별 결과를 group_diff_cost에 저장
            group_diff_cost[g][f'{var}_diff'] = diff_dict
            group_diff_cost[g][f'{var}_sign'] = sign_dict

    return group_diff_cost

# 연속형 비용함수 설정
def compute_continuous_cost(init_grouped_df, s_row, t_row, selected_sort_variable): # selected_sort_variable : 단일 변수(1순위)
    import numpy as np
    
    pop_mean = float(init_grouped_df[selected_sort_variable].mean())

    source_group_df = init_grouped_df[init_grouped_df['초기그룹'] == s_row['초기그룹']]
    target_group_df = init_grouped_df[init_grouped_df['초기그룹'] == t_row['초기그룹']]

    before_dist = (
        np.abs(source_group_df[selected_sort_variable].mean() - pop_mean) +
        np.abs(target_group_df[selected_sort_variable].mean() - pop_mean)
    )

    # 시뮬레이션 이동
    simulated_df = init_grouped_df.copy()
    simulated_df.loc[s_row.name, '초기그룹'] = t_row['초기그룹']

    # 이동 후 그룹별 평균 계산
    group_mean_after = simulated_df.groupby('초기그룹')[selected_sort_variable].mean()

    # 이동 후 거리 계산
    after_dist = (
        np.abs(group_mean_after.loc[t_row['초기그룹']] - pop_mean) +
        np.abs(group_mean_after.loc[s_row['초기그룹']] - pop_mean)
    )

    continuous_cost = abs(after_dist - before_dist)
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

    # deepcopy로 원본 안전하게 복사
    new_group_diff = copy.deepcopy(group_diff_cost)

    source_group = s_row['초기그룹']
    target_group = t_row['초기그룹']

    # -------------------------------------------------
    # 1️⃣ 학생 단위 이동 허용 여부 판단
    # -------------------------------------------------
    move_allowed = False  # 기본값: 이동 불가

    for var in selected_discrete_variable:
        source_cat = s_row[var]
        sign_val = group_diff_cost[source_group][f'{var}_sign'][source_cat]

        if sign_val > 0:  # 부족 상태면 즉시 이동 금지
            return -np.inf
        elif sign_val < 0:  # 과잉 상태 존재 시 이동 고려
            move_allowed = True

    # 모든 변수에서 균형(0)이거나 부족(+)이면 이동 금지
    if not move_allowed:
        return -np.inf

    # -------------------------------------------------
    # 2️⃣ 이동 시나리오에 따른 비용 계산
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
    # 3️⃣ 최종 비용 변화량 계산
    # -------------------------------------------------
    discrete_cost_change = abs(after_cost - before_cost)
    return discrete_cost_change

def cost_group_move(max_iter, tolerance, w_discrete, w_continuous, init_grouped_df, selected_discrete_variable, selected_sort_variable_dict):
    print("cost_group_move 인자 확인")
    print(f"max_iter: {max_iter}, tolerance: {tolerance}, w_discrete: {w_discrete}, w_continuous: {w_continuous}")
    print(f"selected_discrete_variable: {selected_discrete_variable}, selected_sort_variable: {selected_sort_variable_dict}")

    # -------------------------------------------------
    # 이산형 변수를 선택하지 않은 경우
    # -------------------------------------------------
    if not selected_discrete_variable :
        print("이산형 변수가 선택되지 않아 이산형 비용 함수 계산을 건너뜁니다.")
        prev_total_cost = float('inf')
        for iter_num in range(max_iter):
            print(f"\n======= Iteration {iter_num+1} =======")
            # 연속형 변수는 1순위 변수로만 비용함수 계산
            selected_sort_variable = list(selected_sort_variable_dict.keys())[-1] # selected_sort_variable_dict : 딕셔너리 형태로 입력됨 (오른쪽으로 갈수록 우선순위 높음)
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
            group_total_cost = {}
            groups = init_grouped_df['초기그룹'].unique()
            for g in groups:
                group_df = init_grouped_df[init_grouped_df['초기그룹'] == g]
                group_mean = group_df[selected_sort_variable].mean()
                group_total_cost[g] = pop_mean - group_mean # 양수면 이상치보다 적은 평균, 음수면 이상치보다 큰 평균
            # group_total_cost 기준 이상치보다 제일 높은 평균의 그룹 탐색
            source_group_idx = min(group_total_cost, key=group_total_cost.get)
            print("이동할 그룹:", source_group_idx)
            # group_total_cost 기준 이상치보다 하단에서 3개의 그룹 선택
            match_group_idx = sorted(group_total_cost, key=group_total_cost.get, reverse=True)[:3] # 큰값 기준 3개
            print("매칭 그룹:", set(match_group_idx))
            print("##############################")
            source_group_df = init_grouped_df[init_grouped_df['초기그룹']==source_group_idx]
            target_group_df = init_grouped_df[init_grouped_df['초기그룹'].isin(match_group_idx)] # match_group_idx : list 형태
            # 모든 쌍에 대해 비용 계산을 담은 딕셔너리
            pair_costs = {}
            for s_idx, s_row in source_group_df.iterrows():
                for t_idx, t_row in target_group_df.iterrows():
                    # 여기서 s_row와 t_row를 비교하여 교환 가능성을 탐색
                    # 혹시 모르니 조건 설정
                    if s_row['초기그룹'] == t_row['초기그룹']:
                        continue
                    elif len(source_group_df) <= len(target_group_df): # 이동하는 그룹의 수보다 도착 그룹의 수가 더 많거나 같으면 비용계산 X
                        continue
                    # 연속형 변수 비용 계산
                    cont_cost = compute_continuous_cost(init_grouped_df, s_row, t_row, selected_sort_variable)
                    size_penalty = abs(group_sizes[s_row['초기그룹']] - group_mean_size) + abs(group_sizes[t_row['초기그룹']] - group_mean_size)
                    # 총 비용 계산
                    print(f"쌍 ({s_idx}, {t_idx}) 연속형 비용: {cont_cost}, 크기 패널티: {size_penalty}")
                    total_cost = w_continuous * cont_cost + 100 * size_penalty
                    pair_costs[(s_idx, t_idx)] = total_cost
            # 최대 비용 쌍 선택
            best_pair = max(pair_costs, key=lambda x: pair_costs[x])
            best_cost = pair_costs[best_pair]
            print("최고 효율 쌍:", best_pair)
            print("비용:", best_cost)
            # 실제 그룹 이동
            idx_s, idx_t = best_pair
            # 그룹 실제 이동하는거 확인
            print("++이동하는 정보++")
            print(f"그룹 이동 완료: {init_grouped_df.loc[idx_s, '초기그룹']} -> {init_grouped_df.loc[idx_t, '초기그룹']}")
            print("이동 학생 정보:")
            print(init_grouped_df.loc[idx_s, selected_discrete_variable])
            print("+++++++++++++++")
            init_grouped_df.loc[idx_s, '초기그룹'] = init_grouped_df.loc[idx_t, '초기그룹']
            improvement = abs(prev_total_cost - best_cost)
            print(f"비용 개선폭: {improvement:.6f}")
            # 이동 후 그룹별 평균과 분포 출력
            group_freq = compute_group_discrete_freq(init_grouped_df, selected_discrete_variable)
            print("Group Frequency:")
            print(group_freq)
            if improvement < tolerance:
                print("개선 폭이 작아 중단합니다.")
                break
            prev_total_cost = best_cost


    # -------------------------------------------------
    # 이산형, 연속형 변수를 모두 선택한 경우
    # -------------------------------------------------   
    else :
        print("이산형, 연속형 변수 모두 선택되어 비용 함수 계산을 진행합니다.")
        ideal_freq = compute_ideal_discrete_freq(init_grouped_df, selected_discrete_variable)
        prev_total_cost = float('inf')
        for iter_num in range(max_iter):
            print(f"\n======= Iteration {iter_num+1} =======")
            # 그룹별 총 불균형도 및 diff, sign 계산
            group_freq = compute_group_discrete_freq(init_grouped_df, selected_discrete_variable) # 각 그룹별 이산형 실제 빈도
            group_total_cost = compute_group_total_cost(ideal_freq, group_freq, selected_discrete_variable) # 각 그룹별 이산형 총 불균형도
            group_diff_cost = compute_group_diff_and_sign(ideal_freq, group_freq, selected_discrete_variable) # 각 그룹별 이산형 diff, sign
            groups = init_grouped_df['초기그룹'].unique()

            # 이동 전 이상현 빈도 출력
            print("##############################")
            print("이전 그룹 빈도 분포:")
            print(group_freq)
            print("그룹별 총 불균형도:")
            print(group_total_cost)
            print("##############################")
            
            # 불균형이 가장 큰 그룹 탐색
            source_group_idx = min(group_total_cost, key=group_total_cost.get)
            print("##############################")
            print(f"최고 편차 그룹 : {source_group_idx}")
            print("이산변수 차이 및 부호 값:")
            print(group_diff_cost[source_group_idx])
            print("##############################")
            print("------- 후보 타깃 그룹 탐색 -------")
            for g in groups:
                if g == source_group_idx:
                    continue
                print(f"그룹 {g}의 이산변수 차이 및 부호 값:")
                print(group_diff_cost[g])
            print("##############################")

            # 타깃 그룹 후보 제한 추가
            # source_group와 반대 방향을 지니고 있는 그룹들로 제한
            print("##############################")
            reversed_target = {}
            for var in selected_discrete_variable:
                reversed_target[var] = {}
                for key, value in group_diff_cost[source_group_idx][f'{var}_sign'].items():
                    reversed_target[var][key] = -value
            print(reversed_target)
            # 이거와 일치하는 방향의 그룹
            match_group_idx = []
            for g in groups:
                all_match = True
                for var in selected_discrete_variable:
                    for key, value in reversed_target[var].items():
                        if group_diff_cost[g][f'{var}_sign'].get(key, 0) != value:
                            all_match = False
                            break
                    if not all_match:
                        break
                if all_match:
                    match_group_idx.append(g)
                    print(f"Matched group: {g}")
                    print("이산변수 차이 및 부호 값:")
                    print(group_diff_cost[g])
            if match_group_idx == []:
                print("일치하는 매칭 그룹이 없습니다. 그룹 전체로 진행됩니다.")
                # 모든 그룹을 대상으로 탐색할 경우 시간 비용이 너무 큼
                # match_group_idx = [g for g in groups if g != source_group_idx]
                match_group_idx = sorted(group_total_cost, key=group_total_cost.get, reverse=True)[:3]

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
                    # 혹시 모르니 조건 설정
                    if s_row['초기그룹'] == t_row['초기그룹']:
                        continue
                    # 이산형 변수 비용 계산
                    disc_cost = compute_discrete_cost(group_diff_cost, s_row, t_row, selected_discrete_variable)
                    # 연속형 변수 비용 계산
                    cont_cost = compute_continuous_cost(init_grouped_df, s_row, t_row, selected_sort_variable)
                    # 총 비용 계산
                    total_cost = w_discrete * disc_cost + w_continuous * cont_cost

                    pair_costs[(s_idx, t_idx)] = total_cost
            # 최대 비용 쌍 선택
            best_pair = max(pair_costs, key=lambda x: pair_costs[x])
            best_cost = pair_costs[best_pair]
            print("최고 효율 쌍:", best_pair)
            print("비용:", best_cost)
            # 실제 그룹 이동
            idx_s, idx_t = best_pair
            # 그룹 실제 이동하는거 확인
            print("++이동하는 정보++")
            print(f"그룹 이동 완료: {init_grouped_df.loc[idx_s, '초기그룹']} -> {init_grouped_df.loc[idx_t, '초기그룹']}")
            print("이동 학생 정보:")
            print(init_grouped_df.loc[idx_s, selected_discrete_variable])
            print("+++++++++++++++")
            init_grouped_df.loc[idx_s, '초기그룹'] = init_grouped_df.loc[idx_t, '초기그룹']
            improvement = abs(prev_total_cost - best_cost)
            print(f"비용 개선폭: {improvement:.6f}")
            # 이동 후 그룹별 평균과 분포 출력
            group_freq = compute_group_discrete_freq(init_grouped_df, selected_discrete_variable)
            print("Group Frequency:")
            print(group_freq)
            if improvement < tolerance:
                print("개선 폭이 작아 중단합니다.")
                break
            prev_total_cost = best_cost
    return init_grouped_df