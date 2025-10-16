def tuple_from_df(df, col_names):
    '''
    df : pandas dataframe
    col_names : 고려해야할 변수명, 리스트 형태로 입력 (예:['x1'], ['x1', 'x2'])
    * 오른쪽으로 갈수록 변수 정렬 우선순위가 더 높음
    '''
    import numpy as np

    tuples = tuple(df[col].to_numpy() for col in col_names)
    return tuples # 입력할 데이터

def suitable_bin_value(tuples, k): # 데이터를 살펴보고 적절한 bin_value를 찾는 함수
    '''
    tuples : 정렬하고자 하는 데이터, numpy array 타입이며 1차원이 기본
    고려해야할 변수가 1가지인 경우 튜플 형태로 입력 (예:(x1,))
    2가지 이상인 경우 튜플 형태로 입력 (예:(x1, x2))
    * 오른쪽으로 갈수록 변수 정렬 우선순위가 더 높음
    '''
    import numpy as np

    sorted_idx = np.lexsort(tuples) # ! 튜플의 마지막 원소를 기준으로 먼저 정렬
    sorted_x = np.asarray(tuples)[-1][sorted_idx] # 가장 우선순위가 높은 변수로 정렬 (배열 인덱싱)
    n = len(sorted_x)
    print(f"전체 데이터 개수: {n}")
    print(f"그룹 개수 : {k}")

    max_value = None # 혹시 추후에 max_value 설정할 일이 있을까봐 남겨둠
    if max_value is None:
        max_value = n // 2
    
    min_value = 2 # 초기값 설정
    for value in range(min_value, max_value + 1):
        print(f"현재 시도하는 bin_value: {value}")
        count, edges = np.histogram(sorted_x, bins=value)
        print(f"각 bin에 속한 데이터 개수: {count}")
        print(count[count < k].size)
        if count[count < k].size >= 1:
            print(f"적절한 bin_value: {value-1}")
            final_bin_value = value - 1
            break
    return final_bin_value

def init_group_assign(x, k, final_bin_value): # x: 전체 데이터, k: 그룹 수, final_bin_value: suitable_bin_value 함수에서 구한 최종 bin_value
    import numpy as np

    bin_edges = np.histogram_bin_edges(x, bins=final_bin_value)
    print(f"초기 bin_edges: {bin_edges}")

    bin_idx = np.digitize(x, bin_edges, right=True) -1
    bin_idx[x >= bin_edges[-1]] = len(bin_edges) - 2
    print(np.unique(bin_idx, return_counts=True)) # 각 bin별 데이터 수

    if len(x)
    origin_index = 

    final_bin_value = suitable_bin_value(x, k)
    print(f"최종 bin_value: {final_bin_value}")
    count, edges = np.histogram(x, bins=final_bin_value)
    print(f"각 bin에 속한 데이터 개수: {count}")

    group_assign = np.digitize(x, edges, right=True)
    print(f"각 데이터의 그룹 할당 결과: {group_assign}")

    unique, counts = np.unique(group_assign, return_counts=True)
    group_count = dict(zip(unique, counts))
    print(f"각 그룹별 데이터 개수: {group_count}")

    return group_assign