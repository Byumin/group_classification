from collections import defaultdict


def run(context):
    # 동명이인의 경우 다른 반에 배정되도록하는 알고리즘
    # 그룹이 분류된 이후에
    # 각 그룹에 동명이인 학생이 있는지 확인
    # 있는 경우 다른 그룹에서 가장 유사한 학생과 스왑
    '''
    df_sorted: DataFrame, 그룹 분류된 DataFrame
    group_count: int, 그룹의 개수

    '''
    # context에서 필요한 데이터 가져오기
    df_sorted = context.get('df_sorted')
    group_count = context.get('group_count')
    selected_sort_variable_dict = context.get('selected_sort_variable_dict')
    selected_discrete_variable = conntext.get('selected_discrete_variable')
    classification_name_column = context.get('classification_name_column')
    classification_name_option = context.get('classification_name_option')

    # 이름만 같으면 동명이인으로 간주할 것인지? -> 원래 있는 이름 컬럼에서 끝 2자리로 판단 
    # 성과 이름 모두 같으면 동명이인으로 간주할 것인지? -> 원래 있는 이름 컬럼으로 판단

    # df_sorted에서 그룹별로 데이터 프레임 분리
    group_dfs = df_sorted.groupby('group')

    # 동명이인 확인
    # 성+이름 조건과 이름만 조건 분기
    for group_name, group_df in group_dfs :
        same_name_indices = defaultdict(lambda : defaultdict(list))
        group_name_list = group_df[classification_name_column]
        # 성+이름 조건
        if classification_name_option == '성+이름':
            # 그룹 내에 성+이름이 같은 학생이 있는지 확인
            for idx, name in enumerate(group_name_list) :