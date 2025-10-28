# 관계 그룹 탐색 함수
def find_relation_groups(relation_dict):
    from collections import defaultdict, deque

    # 1️⃣ 모든 학생 목록 추출 (key, value 모두 포함)
    all_students = set(relation_dict.keys())
    for relations in relation_dict.values():
        all_students.update(relations.keys())

    # 2️⃣ 양방향 그래프 생성 (관계 == 1 인 경우만)
    graph = defaultdict(set)
    for student, relations in relation_dict.items():
        for other, relation in relations.items():
            if relation == 1:
                graph[student].add(other)
                graph[other].add(student)

    # 3️⃣ 방문 관리 및 BFS 탐색
    visited = set()
    groups = []

    for student in all_students:
        if student not in visited:
            visited.add(student)
            group = set([student])
            queue = deque([student])

            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        group.add(neighbor)
                        queue.append(neighbor)

            groups.append(group)

    return groups

def relation_groups_to_dict(groups, group_assign_df):
    """
    관계 그룹 리스트를 딕셔너리로 변환
    groups : 관계 그룹 리스트
    group_assign_df : 학생들의 그룹 배정 데이터프레임
    반환값: 관계 그룹 딕셔너리 (키: 관계그룹명, 값: 관계그룹 데이터프레임)
    """
    from copy import deepcopy

    # 관계 그룹 이름 부여
    relationship_group_dict = {}
    for i, group in enumerate(groups):
        rel_group_name = f"관계그룹_{i+1}"
        relationship_group_dict[rel_group_name] = group

    relationship_group_df_dict = {}
    for key, names in relationship_group_dict.items():
        relationship_group_df_dict[key] = group_assign_df[group_assign_df['merge_key'].isin(names)]

    return relationship_group_dict, relationship_group_df_dict

def assign_relation_groups_optimal(group_assign_df, relationship_group_dict, relationship_group_df_dict, selected_discrete_variable):
    """
    관계그룹을 전체 그룹에 최적으로 배정 (헝가리안 알고리즘 기반)
    group_assign_df : 전체 학생들의 그룹 배정 데이터프레임
    relationship_group_dict : 관계 그룹 객체들을 담은 딕셔너리 (키: 관계그룹명, 값: 관계그룹 학생명 리스트)
    relationship_group_df_dict : 관계 그룹 객체들을 담은 딕셔너리 (키: 관계그룹명, 값: 관계그룹 데이터프레임)
    selected_discrete_variable : 이산형 변수 리스트
    반환값: 최적 배정 딕셔너리, 최소 총 비용
    """
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    from scipy.optimize import linear_sum_assignment
    from cost_group_move import compute_group_discrete_freq, compute_group_total_cost, compute_ideal_discrete_freq

    relation_group_keys = list(relationship_group_df_dict.keys())
    group_candidates = list(group_assign_df['초기그룹'].unique())
    remaining_df = group_assign_df[~group_assign_df['merge_key'].isin(set().union(*relationship_group_dict.values()))]
    
    R, G = len(relation_group_keys), len(group_candidates)
    print(f"관계그룹 수: {R}, 전체 그룹 수: {G}")

    # 이상적인 분포 계산
    ideal_freq = compute_ideal_discrete_freq(group_assign_df, selected_discrete_variable)

    # 비용 행렬 초기화
    cost_matrix = np.zeros((R, G))
    
    print("\n=== 비용 행렬 계산 중 ===")
    for i, rel_key in enumerate(relation_group_keys):
        rel_df = relationship_group_df_dict[rel_key]
        for j, g in enumerate(group_candidates):
            temp_df = deepcopy(remaining_df)
            rel_copy = rel_df.copy()
            rel_copy['초기그룹'] = g
            temp_df = pd.concat([temp_df, rel_copy], ignore_index=False)

            after_group_freq = compute_group_discrete_freq(temp_df, selected_discrete_variable)
            after_group_total_cost = compute_group_total_cost(ideal_freq, after_group_freq, selected_discrete_variable)
            total_cost = sum(abs(v) for v in after_group_total_cost.values())
            
            cost_matrix[i, j] = total_cost
        print(f"관계그룹 {rel_key} 완료.")

    print("\n비용 행렬:")
    print(pd.DataFrame(cost_matrix, index=relation_group_keys, columns=group_candidates))

    # 헝가리안 알고리즘 실행
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    best_assignment = {relation_group_keys[i]: group_candidates[j] for i, j in zip(row_ind, col_ind)}
    best_total_cost = cost_matrix[row_ind, col_ind].sum()

    print("\n✅ 최적 배정 결과:")
    for rel, grp in best_assignment.items():
        print(f" - 관계그룹 {rel} → 그룹 {grp}")

    print(f"\n🔻 최소 총 비용: {best_total_cost:.4f}")
    return remaining_df, best_assignment, best_total_cost

def merge_optimal_assignments(remaining_df, best_assignment, relationship_group_df_dict):
    """
    헝가리안 알고리즘 결과(best_assignment)를 반영하여
    관계그룹을 실제로 배정한 완전한 데이터프레임을 생성하는 함수.

    Parameters
    ----------
    remaining_df : pd.DataFrame
        관계그룹이 빠진 상태의 원본 그룹배정 데이터프레임.
    best_assignment : dict
        {관계그룹 이름 : 배정할 그룹번호} 형태의 딕셔너리.
        예: {'R1': 'g2', 'R2': 'g4'}
    relationship_group_df_dict : dict
        {관계그룹 이름 : 관계그룹 데이터프레임} 형태의 딕셔너리.

    Returns
    -------
    final_df : pd.DataFrame
        관계그룹 배정을 모두 반영한 최종 데이터프레임.
    """
    import pandas as pd
    from copy import deepcopy

    # 원본 복사
    final_df = deepcopy(remaining_df)
    final_df["그룹고정"] = False  # 관계그룹 병합 후에도 고정 여부 컬럼 유지

    # 각 관계그룹을 배정 결과에 따라 병합
    for rel_name, target_group in best_assignment.items():
        if rel_name not in relationship_group_df_dict:
            print(f"[경고] {rel_name}는 relationship_group_df_dict에 없음 — 건너뜀.")
            continue

        # 해당 관계그룹 DataFrame 복사
        rel_df = deepcopy(relationship_group_df_dict[rel_name])

        # 관계그룹의 그룹 번호를 최적 배정된 그룹으로 변경
        rel_df['초기그룹'] = target_group
        # 관계그룹 내 모든 학생의 그룹고정 컬럼을 True로 설정
        rel_df['그룹고정'] = True

        # 병합 (ignore_index=False → 기존 인덱스 유지)
        final_df = pd.concat([final_df, rel_df], ignore_index=False)

        print(f"✅ {rel_name} → {target_group} 배정 완료 (추가된 행: {len(rel_df)})")

    # 인덱스 정리 (원하면 True로 초기화 가능)
    final_df.reset_index(drop=True, inplace=True)

    print(f"\n🎯 최종 DataFrame 완성: 총 {len(final_df)}명")
    return final_df