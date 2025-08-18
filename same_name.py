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
    # df_sorted에서 그룹별로 데이터 프레임 분리
    group_dfs = df_sorted.groupby('group')
    