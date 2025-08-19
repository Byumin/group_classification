from collections import defaultdict
import copy
from turtle import st

# 분배 방식 및 남녀 합반에 따른 df 생성 함수
def get_sortable_method_df(df, selected_sort_variable_dict, group_count, sortable_method, sex_classification, man_class_count, female_class_count):
    '''
    selected_sort_variable_dict: {'변수명': True/False} 형태로, True는 오름차순 정렬, False는 내림차순 정렬
    group_count: 그룹의 개수
    sortable_method: 'round-robin' 또는 'serpentine' 중 하나
    sex_classification: '합반' 또는 '분반'
    man_class_count: 남학생 그룹 수
    female_class_count: 여학생 그룹 수
    '''
    # 선택된 정렬 변수에 따라 df를 정렬하고 그룹을 할당하는 함수
    # sortable_method는 'round-robin' 또는 'serpentine' 중 하나로 가정
    # round-robin: 순차적으로 그룹을 할당 (1->2->3->4->1->2->3->4)

    # 남여 합반 분반에 따른 조건 분기
    if sex_classification == '합반':
        sex_grouped = df.groupby('성별')
        for sex, group_df in sex_grouped:
            # 그룹별로 정렬 및 그룹 할당
            if sortable_method == 'round-robin':
                group_df_sorted = group_df.sort_values(by=list(selected_sort_variable_dict.keys()), ascending=list(selected_sort_variable_dict.values())).reset_index(drop=True)
                group_df_sorted['group'] = None
                for i, idx in enumerate(group_df_sorted.index):
                    group = i % group_count
                    group_df_sorted.at[idx, 'group'] = group
            # serpentine: 그룹을 순차적으로 할당하되, 홀수 그룹은 정방향, 짝수 그룹은 역방향으로 할당 (1->2->3->4->3->2->1)
            elif sortable_method == 'serpentine':
                direction = 1 # 1: 정방향, -1: 역방향
                group = 0 # 초기 그룹 번호
                group_df_sorted = group_df.sort_values(by=list(selected_sort_variable_dict.keys()), ascending=list(selected_sort_variable_dict.values())).reset_index(drop=True)
                group_df_sorted['group'] = None
                for i, idx in enumerate(group_df_sorted.index):
                    group_df_sorted.at[idx, 'group'] = group
                    group += direction
                    if group == group_count or group < 0:
                        direction *= -1
                        group += direction
            else:
                raise ValueError(f"지원하지 않는 분배 방식입니다: {sortable_method}")

    elif sex_classification == '분반':
        # 남여 반 갯수 리스화
        class_count = [man_class_count, female_class_count]
        sex_grouped = df.groupby('성별')
        for (sex, group_df), class_count in zip(sex_grouped, class_count):
            if sortable_method == 'round-robin':
                group_df_sorted = group_df.sort_values(by=list(selected_sort_variable_dict.keys()), ascending=list(selected_sort_variable_dict.values())).reset_index(drop=True)
                group_df_sorted['group'] = None
                for i, idx in enumerate(group_df_sorted.index):
                    group = i % class_count
                    group_df_sorted.at[idx, 'group'] = group

            elif sortable_method == 'serpentine':
                direction = 1
                group = 0
                group_df_sorted = group_df.sort_values(by=list(selected_sort_variable_dict.keys()), ascending=list(selected_sort_variable_dict.values())).reset_index(drop=True)
                group_df_sorted['group'] = None
                for i, idx in enumerate(group_df_sorted.index):
                    group_df_sorted.at[idx, 'group'] = group
                    group += direction
                    if group == class_count or group < 0:
                        direction *= -1
                        group += direction
            else:
                raise ValueError(f"지원하지 않는 분배 방식입니다: {sortable_method}")

    return group_df_sorted


def run(context):
    '''
    df: DataFrame, 학생들의 정보가 담긴 DataFrame
    selected_sort_variable_dict: dict, {'변수명': True/False} 형태로, True는 오름차순 정렬, False는 내림차순 정렬
    selected_discrete_variable: list, 범주형 변수의 리스트
    selected_algorithm: str, 선택된 알고리즘 이름
    group_count: int, 그룹의 개수
    sortable_method: str, 'round-robin' 또는 'serpentine'
    group_names: list, 그룹 이름의 리스트 (선택적)
    sex_classification: str, 성별 분류 방식 ('성+이름' 또는 '이름만')
    '''
    df = context.get('df')
    selected_sort_variable_dict = context.get('selected_sort_variable_dict')
    selected_discrete_variable = context.get('selected_discrete_variable')
    selected_algorithm = context.get('selected_algorithm')
    group_count = context.get('group_count')
    sortable_method = context.get('sortable_method')
    group_names = context.get('group_names')
    sex_classification = context.get('sex_classification')
    man_class_count = context.get('man_class_count')
    female_class_count = context.get('female_class_count')

    # * 선택된 정렬 변수에 따라 df를 정렬하고 그룹을 할당
    df_sorted = get_sortable_method_df(df, selected_sort_variable_dict, group_count, sortable_method, sex_classification, man_class_count, female_class_count)

    # 범주형 변수 고려 유무에 따라 그룹화 방식 결정
    if selected_discrete_variable:
        group_stats = [defaultdict(lambda: defaultdict(int)) for _ in range(group_count)] # 각 그룹의 통계 저장용
        for _, row in df_sorted.iterrows():
            g = row['group']
            for col in selected_discrete_variable:
                val = row[col]
                group_stats[g][col][val] += 1 # 순수 정렬기반 그룹의 통계 갱신
                '''
                예시: group_stats =
                [
                    {  # group 0: 남2 여1 / 상담없음2 상담있음1
                        'sex': {1: 2, 2: 1},
                        'counseling': {0: 2, 1: 1}
                    },
                    {  # group 1: 남1 여1 / 상담없음1 상담있음1
                        'sex': {1: 1, 2: 1},
                        'counseling': {0: 1, 1: 1}
                    }
                ]
                '''

        ideal_group_stats = defaultdict(lambda: defaultdict(float)) # 각 범주형 변수의 이상적인 그룹 통계 저장용
        for _, row in df.iterrows():
            for col in selected_discrete_variable:
                val = row[col]
                ideal_group_stats[col][val] += 1 # 전체 빈도 계산
                '''
                예시: ideal_group_stats =
                [
                    'sex': {1: 15, 2: 20},
                    'counseling': {0: 30, 1: 5}
                ]
                '''
        for col in ideal_group_stats:
            for val in ideal_group_stats[col]:
                ideal_group_stats[col][val] /= group_count # 그룹 수로 나누어 이상적인 빈도 계산
                '''
                예시: ideal_group_stats =
                [
                    'sex': {1: 15/그룹 수, 2: 20/그룹 수},
                    'counseling': {0: 30/그룹 수, 1: 5/그룹 수}
                ]
                '''
        for idx, row in df_sorted.iterrows():
            current_group = row['group'] # 지금 학생이 속한 그룹
            best_group = current_group # 더 좋은 그룹이 있다면 이 변수에 저장 (초기값은 현재 그룹)
            min_deviation = float('inf') # 가장 낮은 편차값을 기록할 변수 (초기값은 무한대)

            for g in range(group_count):
                if g == current_group: # 지금 학생이 속한 그룹과 비교해봐야 소용이 없으므로 스킵
                    continue  # 같은 그룹이면 의미 없음(건너뛰기) !continue 하단의 코드는 전부 생략됨.

                # projected group_stats 복사해서 시뮬레이션
                simulated_stats = copy.deepcopy(group_stats) # 학생이 해당 그룹으로 이동을 가정하기 위한 시뮬레이션을 위한 복사본 (아직 이동 안함)

                # (시뮬레이션을 위한) 현재 그룹에서 값 제거 -> 해당 학생이 다른 그룹으로 이동한다는 시뮬레이션
                for col in selected_discrete_variable:
                    val = row[col]
                    simulated_stats[current_group][col][val] -= 1 # 현재 그룹(current_group)에서 해당 값의 개수 감소
                    simulated_stats[g][col][val] += 1 # 새로운 그룹(g)에 해당 값의 개수 증가 (이동을 가짐)

                # (시뮬레이션을 위한) 전체 편차 계산 (예: max deviation 방식)
                total_dev = 0
                for gg in range(group_count): # 각 그룹에 대해
                    for col in selected_discrete_variable: # 각 범주형 변수에 대해
                        for val in ideal_group_stats[col]: # 각 범주형 변수의 값에 대해
                            current = simulated_stats[gg][col][val] # 현재 그룹에서 해당 값의 개수
                            ideal = ideal_group_stats[col][val] # 이상적인 그룹 통계에서 해당 값의 이상적인 개수
                            total_dev += abs(current - ideal) # 해당 그룹에서 해당 값의 개수와 이상적인 개수의 차이의 모든 변수의 절댓값을 더함
                            
                            # ! 여기서 의문인 점이 있음
                            # 범주형 모든 변수의 해당 학생의 값이 해당 그룹에 속했을 때의 편차를 계산하는 것인데
                            # 예를 들어 남자(1)이고 상담 유무(0)인 학생이 그룹 1으로 이동할 떄와
                            # 그룹 2로 이동할 때의 편차를 계산한다고 했을 때
                            # 그룹 1의 편차의 합은 15, 그룹 2의 편차의 합은 10이라고 가정했을 때
                            # 그룹 1으로 이동했을 때의 편차가 더 크므로 그룹 2로 이동하는 것이 더 좋다고 판단함.
                            # 그러나 그룹 2로 이동하면 편차의 합 입장에서는 이상적이겠지만, 각 범주형 변수의 차이에 대해선 이상적일까?

                            # ! 여기서 의문인 점이 있음
                            # 범주형 변수 중 예를 들어 상담 유무와 같이 1: 상담 있음, 0: 상담 없음
                            # 1이라는 값의 빈도 자체가 적은 경우
                            # 값 자체의 편차다 보니 절대적으로 값이 작은 상담 유무에서의 유(1)과 같은 경우
                            # 편차 역시 작게 나올 수 있어서 범주형 변수의 빈도 균형을 맞출때 영향을 덜 미칠 수 있지 않을까?


                if total_dev < min_deviation: # 현재 그룹에서 이동한 그룹의 편차가 최소 편차보다 작다면
                    min_deviation = total_dev # 최소 편차 갱신
                    best_group = g # 최소 편차의 그룹 번호 갱신
            # 최소 편차의 그룹 번호를 갱신 한 경우
            if best_group != current_group:
                df_sorted.at[idx, 'group'] = best_group
                for col in selected_discrete_variable:
                    val = row[col]
                    group_stats[current_group][col][val] -= 1 # 현재 그룹에서 해당 값의 개수 감소
                    group_stats[best_group][col][val] += 1 # 새로운 그룹에서 해당 값의 개수 증가

    elif not selected_discrete_variable:
        pass
    else:
        st.error("그룹화 중에 오류가 발생했습니다.")
    
    # 그룹 이름 설정
    if group_names:
        df_sorted['group'] = df_sorted['group'].apply(lambda x: group_names[x] if x < len(group_names) else f"Group {x + 1}")
    else:
        pass
    return df_sorted