# 관계 그룹 탐색 함수
def find_relation_groups_minimal(relation_dict, max_iter=10, target_n_groups=None, verbose=True):
    """
    +1 관계는 묶고, -1 관계는 분리하되,
    병합 시 작은 그룹끼리 2개씩만 순차적으로 병합하며,
    그룹 수가 target_n_groups 이하로 내려가면 중단한다.
    """
    from collections import defaultdict, deque
    import copy, random

    # 전체 학생 목록 수집
    all_students = set(relation_dict.keys())
    for rels in relation_dict.values():
        all_students.update(rels.keys())

    # 그래프 구성
    graph_pos = defaultdict(set)
    graph_neg = defaultdict(set)
    for s, rels in relation_dict.items():
        for t, v in rels.items():
            if v == 1:
                graph_pos[s].add(t)
                graph_pos[t].add(s)
            elif v == -1:
                graph_neg[s].add(t)
                graph_neg[t].add(s)

    # Step 1️⃣ +1 관계 기반 연결
    visited = set()
    base_groups = []
    for s in all_students:
        if s not in visited:
            queue = deque([s])
            group = set([s])
            visited.add(s)
            while queue:
                cur = queue.popleft()
                for nb in graph_pos[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
                        group.add(nb)
            base_groups.append(group)

    # Step 2️⃣ 그룹 내 -1 관계 분리
    refined_groups = []
    for group in base_groups:
        subgroups = []
        for student in group:
            placed = False
            for sg in subgroups:
                if all((s2 not in graph_neg[student]) for s2 in sg):
                    sg.add(student)
                    placed = True
                    break
            if not placed:
                subgroups.append(set([student]))
        refined_groups.extend(subgroups)

    # refined_groups를 크기 내림차순으로 정렬
    refined_groups = sorted(refined_groups, key=lambda x: -len(x))

    # target_n_groups만큼 빈 그룹 생성
    groups = [set() for _ in range(target_n_groups)]

    # 각 refined_group 배치
    for rg in refined_groups:

        candidate_indices = []

        for gi in range(target_n_groups):
            g = groups[gi]
            conflict = False

            # 충돌 검사: g에 있는 학생과 rg의 학생이 -1 관계인지 검사
            for student in rg:
                if any((other in graph_neg[student]) for other in g):
                    conflict = True
                    break

            if not conflict:
                candidate_indices.append(gi)
        # 어떤 그룹에도 넣을 수 없을 경우 -> 오류 반환
        if not candidate_indices:
            raise ValueError(f"[오류] 관계그룹 {rg} 는 어떤 대상 그룹에도 배정될 수 없습니다. 관계 조건을 완화하거나, 반 개수를 늘려주세요.")

        # 충돌 없으면: 가장 학생 수가 적은 그룹 선택
        best_group_idx = min(candidate_indices, key=lambda gi: len(groups[gi]))
        groups[best_group_idx].update(rg)

    return groups

def find_relation_groups_optimized(relation_dict, max_iter=10, verbose=True):
    """
    +1 관계는 묶고, -1 관계는 분리하되,
    -1 관계 위배 없이 병합 가능한 그룹은 여러 번 반복적으로 병합하여 최적화한다.
    """
    from collections import defaultdict, deque
    import copy

    # 전체 학생 목록 수집
    all_students = set(relation_dict.keys())
    for rels in relation_dict.values():
        all_students.update(rels.keys())

    # 그래프 구성
    graph_pos = defaultdict(set)
    graph_neg = defaultdict(set)
    for s, rels in relation_dict.items():
        for t, v in rels.items():
            if v == 1:
                graph_pos[s].add(t)
                graph_pos[t].add(s)
            elif v == -1:
                graph_neg[s].add(t)
                graph_neg[t].add(s)

    # Step 1️⃣ +1 관계 기반 연결
    visited = set()
    base_groups = []
    for s in all_students:
        if s not in visited:
            queue = deque([s])
            group = set([s])
            visited.add(s)
            while queue:
                cur = queue.popleft()
                for nb in graph_pos[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
                        group.add(nb)
            base_groups.append(group)

    # Step 2️⃣ 그룹 내 -1 관계 분리
    refined_groups = []
    for group in base_groups:
        subgroups = []
        for student in group:
            placed = False
            for sg in subgroups:
                if all((s2 not in graph_neg[student]) for s2 in sg):
                    sg.add(student)
                    placed = True
                    break
            if not placed:
                subgroups.append(set([student]))
        refined_groups.extend(subgroups)

    # Step 3️⃣ 반복 병합 최적화
    groups = copy.deepcopy(refined_groups)

    def has_conflict(g1, g2):
        """두 그룹 사이에 -1 관계가 있으면 True"""
        for a in g1:
            a_rel = relation_dict.get(a, {})  # 안전 접근
            for b in g2:
                b_rel = relation_dict.get(b, {})  # 안전 접근
                if a_rel.get(b) == -1 or b_rel.get(a) == -1:
                    return True
        return False

    for iteration in range(max_iter):
        merged_any = False
        used = set()
        new_groups = []

        for i, g1 in enumerate(groups):
            if any(x in used for x in g1):
                continue
            merged = set(g1)
            for j, g2 in enumerate(groups):
                if i == j or any(x in used for x in g2):
                    continue
                if not has_conflict(merged, g2):  # 관계 위배 없으면 병합
                    merged |= g2
                    used |= g2
                    merged_any = True
            new_groups.append(merged)
            used |= g1

        groups = new_groups

        if verbose:
            print(f"🌀 Iter {iteration+1}: 그룹 수 = {len(groups)}")

        if not merged_any:
            if verbose:
                print("✅ 더 이상 병합 가능한 그룹이 없어 중단합니다.")
            break

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
            _, after_group_total_cost_square = compute_group_total_cost(ideal_freq, after_group_freq, selected_discrete_variable)
            total_cost = sum(abs(v) for v in after_group_total_cost_square.values())
            
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