import numpy as np
import pandas as pd


import numpy as np
def compute_swap_discrete_cost(s_row, t_row, selected_discrete_variable):
    """
    이산형 변수 비용 계산
    - 서로 다르면 1
    - 같으면 0
    """
    cost = 0
    for col in selected_discrete_variable:
        if s_row[col] != t_row[col]:
            cost += 1
    return cost

def compute_swap_continuous_cost(s_row, t_row, cont_vars):
    """ 연속형 변수 기반 유클리드 거리 계산 """
    diffs = []
    for col in cont_vars:
        s_val = s_row[col]
        t_val = t_row[col]
        # 결측치 방어
        if pd.isna(s_val) or pd.isna(t_val):
            continue
        diffs.append((s_val - t_val) ** 2)

    if len(diffs) == 0:
        return 0.0
    return np.sqrt(np.mean(diffs))

def compute_group_cost_after_swap(counts, expected, source_group, target_group):
    """
    source_group의 특이분류학생 ↔ target_group의 일반 학생 swap 후 전체 그룹 비용 계산
    """
    cost = 0
    for g, v in counts.items():
        if g == source_group:
            diff = (v-1) - expected
        elif g == target_group:
            diff = (v+1) - expected
        else:
            diff = v - expected
        cost += diff ** 2
    return cost



def cost_group_swap_special_v2(max_iter_per_col, w_discrete, w_continuous, init_grouped_df, special_cols, relationship_dict, selected_discrete_variable, selected_sort_variable_dict):
    """
    특이분류학생 균등 배치:
    - 출발/도착 그룹은 special_cols 기준 반 별 특이분류학생 수 기대값 대비 편차로 결정
    - 각 출발그룹 기준으로 가능한 모든 학생 swap 후보 평가 후, 비용(그룹 분포 개선 + 학생 유사도)이 최소인 swap 1건을 수행
    - 학생 교환은 이산형&연속형 변수 유사도를 최소화하는 방향으로 진행
    """
    with open("swap_log.txt", "w", encoding="utf-8") as f:
        f.write("=== Swap Log Start ===\n")
    df = init_grouped_df.copy()
    cont_vars = list(selected_sort_variable_dict.keys())
    same_class_students = {student for student, relations in relationship_dict.items() if 1 in relations.values()}
    df['special_sum'] = df[special_cols].sum(axis=1)
    multi_special_students = set(df.loc[df['special_sum'] >= 2, 'merge_key']) # 다중 특이분류 학생
    print("다중 특이분류 학생 >>> ", multi_special_students)

    swapped_students = set() # 이미 교환된 학생 기록
    for col in special_cols:
        print(f"\n===== [{col}] 유사도 기반 swap 시작 =====")
        swap_count = 0
        for iter_num in range(max_iter_per_col):
            # 반별 편차 계산
            summary = df.groupby('초기그룹')[special_cols].sum().reset_index()
            expected = summary[col].sum() / summary['초기그룹'].nunique()
            summary['편차'] = summary[col] - expected
            # 출발그룹 선정
            source_groups = summary.loc[summary['편차'] >= 1, '초기그룹'].tolist() # 여러개 나올 수 있음
            # 도착그룹 선정
            target_groups = summary.loc[summary['편차'] <= -1, '초기그룹'].tolist() # 여러개 나올 수 있음
            # 출발, 도착그룹 특이 조건 처리
            if not source_groups and not target_groups:
                print(f"[{col}] 모든 반이 기대값 ±1 이내 → 종료")
                break
            if source_groups and not target_groups:
                min_val = summary['편차'].min()
                target_groups = summary.loc[summary['편차'] == min_val, '초기그룹'].tolist()
            elif target_groups and not source_groups:
                max_val = summary['편차'].max()
                source_groups = summary.loc[summary['편차'] == max_val, '초기그룹'].tolist()
            #### 출발, 도착그룹이 정해지면
            
            counts = dict(zip(summary['초기그룹'], summary[special_cols].sum(axis=1))) # 초기그룹별 특이분류학생 합
            total_expected = summary[special_cols].sum().sum() / summary['초기그룹'].nunique()
            # 그룹 간 교환했을때 그룹 대상인 비용 계산
            group_costs = {} # (출발그룹, 도착그룹) : 비용 (이상치와 비교해서 얼마나 개선되는지, 작을수록 좋음)
            for sg in source_groups:
                for tg in target_groups:
                    group_costs[(sg, tg)] = compute_group_cost_after_swap(counts, total_expected, sg, tg)
            
            # 각 출발그룹마다 각 도착그룹의 학생
            swap_candidates = [] # 출발그룹마다 교환 후보 초기화
            for source_group in source_groups:
                source_students = df[ # 출발그룹의 특정 특이분류 학생들 중에서 관계그룹 0,-1 인 학생들
                    (df['초기그룹'] == source_group) &
                    (df[col] == 1) & # 문자형은 "1"로 수정 필요
                    (~df['merge_key'].isin(same_class_students)) & # 관계그룹 0,-1 인 학생들
                    (~df['merge_key'].isin(swapped_students)) # 이미 교환된 학생 제외
                ]
                # 다중 특이분류학생이 있으면 필터링
                multi_filtered_source_students = source_students[(~source_students['merge_key'].isin(multi_special_students))] # 다중 특이분류학생 제거한 후보
                if not multi_filtered_source_students.empty:
                    source_students = multi_filtered_source_students
                else: # 만약 없으면 원래 source_students 유지
                    pass

                for s_idx, s_row in source_students.iterrows():
                    s_name = s_row['merge_key']
                    neg_targets = {t for t, v in relationship_dict.get(s_name, {}).items() if v == -1}
                    
                    for t_group in target_groups:
                        base_group_cost = group_costs.get((source_group, t_group), 0)
                        target_students = df[ # 도착그룹의 특이분류학생이 아닌 일반 학생들 중에서 관계그룹 -1이 아닌 학생들
                            (df['초기그룹'] == t_group) &
                            (df['special_sum'] == 0) & # 일반 학생 # 문자형은 수정 필요
                            (~df['merge_key'].isin(neg_targets)) &
                            (~df['merge_key'].isin(swapped_students)) # 이미 교환된 학생 제외
                        ]
                        if target_students.empty:
                            continue
                        
                        # t_group 안에서 모든 학생과의 유사도 계산
                        for t_idx, t_row in target_students.iterrows():
                            disc_cost = compute_swap_discrete_cost(s_row, t_row, selected_discrete_variable) # 학생 교환 시 이산형 상태 유사도 비용
                            cont_cost = compute_swap_continuous_cost(s_row, t_row, cont_vars) # 학생 교환 시 연속형 상태 유사도 비용
                            sim_cost = w_discrete * disc_cost + w_continuous * cont_cost
                            total_cost = 200*base_group_cost + sim_cost # 가중치 부여해 그룹 분포 개선을 우선시
                            swap_candidates.append({
                                "sg": source_group,
                                "tg": t_group,
                                "s_idx": s_idx,
                                "s_name": s_name,
                                "t_idx": t_idx,
                                "t_name": t_row['merge_key'],
                                "total_cost": total_cost
                            })
            if not swap_candidates:
                continue
            best_swap = min(swap_candidates, key=lambda x: x["total_cost"])
            
            sg = best_swap["sg"]
            tg = best_swap["tg"]
            s_idx = best_swap["s_idx"]
            t_idx = best_swap["t_idx"]

            # swap 진행
            df.loc[s_idx, '초기그룹'] = tg
            df.loc[t_idx, '초기그룹'] = sg

            # 교환된 학생 기록
            swapped_students.add(best_swap['s_name'])
            swapped_students.add(best_swap['t_name'])
            print("이미 교환된 학생 : ", swapped_students)
            
            swap_count += 1

            with open("swap_log.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"[{col}] Iter {iter_num} | "
                    f"{best_swap['s_name']} ({sg}→{tg}) ↔ "
                    f"{best_swap['t_name']} ({tg}→{sg}) | "
                    f"cost={best_swap['total_cost']:.4f}\n"
                )
    return df
