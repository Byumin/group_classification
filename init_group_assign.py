def tuple_from_df(df, col_names): # lexsort에 넣을 튜플 생성 함수 (조건부 정렬용)
    '''
    df : pandas dataframe
    col_names : 고려해야할 변수명, 리스트 형태로 입력 (예:['x1'], ['x1', 'x2'])
    * 오른쪽으로 갈수록 변수 정렬 우선순위가 더 높음
    '''
    import pandas as pd
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

    if k <= 1:
        print("그룹 수가 1개이므로 bin_value는 1로 설정합니다.")
        return sorted_idx, sorted_x, 1

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
    return sorted_idx, sorted_x, final_bin_value
'''
x1와 x2를 고려해서 정렬을 했을때, x1이 같다면 x2를 기준으로 정렬
결국 히스토그램 대상은 1차원인 변수 하나이기 때문에
해당 변수가 동일하다면 다른 bin으로 나누는 것이 불가능
따라서 bin 내부에서 교차 배정되는 그룹 규칙에서는
x1과 x2 모두 고려하든 x1만 고려하든 동일한 결과가 나옴.
'''

def init_group_assign(tuples, k, final_bin_value): # x: 전체 데이터, k: 그룹 수, final_bin_value: suitable_bin_value 함수에서 구한 최종 bin_value
    import numpy as np

    x = np.asarray(tuples)[-1] # 가장 우선순위가 높은 변수로 정렬 (배열 인덱싱)
    bin_edges = np.histogram_bin_edges(x, bins=final_bin_value)
    print(f"초기 bin_edges: {bin_edges}")

    bin_idx = np.digitize(x, bin_edges, right=True) -1
    bin_idx[x >= bin_edges[-1]] = len(bin_edges) - 2
    print(np.unique(bin_idx, return_counts=True)) # 각 bin별 데이터 수

    origin_idx = np.argsort(x) # 원래 데이터 순서대로 정렬했을 때의 인덱스
    print(f"원래 데이터 순서대로 정렬했을 때의 인덱스: {origin_idx}")
    group_assign = np.zeros(len(x), dtype=int)

    current_group = 0
    for b in range(final_bin_value): # 각 bin마다
        print(f"현재 bin: {b}") # x와 bin_idx의 인덱스는 동일
        idx_in_bin = np.where(bin_idx == b)[0] # 현재 bin에 속한 데이터 원본 인덱스
        print(f"현재 bin에 속한 데이터 원본 인덱스: {idx_in_bin}")

        sorted_idx_in_bin = idx_in_bin[np.argsort(x[idx_in_bin])]
        # x[idx_in_bin] bin에 속하는 데이터 값 뽑고
        # np.argsort()로 정렬된 인덱스 뽑고
        # idx_in_bin[정렬된 인덱스]로 원본 인덱스 매핑 / 헷갈리네
        print(f"현재 bin에 속한 데이터 원본 인덱스(값 기준 정렬됨): {sorted_idx_in_bin}")

        for idx in sorted_idx_in_bin: # bin 내부에서 값 기준으로 정렬된 순서대로
            group_assign[idx] = current_group
            current_group = (current_group + 1) % k
        print(f"현재까지의 그룹 배정: {group_assign}")
    print(f"최종 그룹 배정: {group_assign}")
    return group_assign # 원본 데이터 전체 길이와 같은 길이를 가진 1차원 numpy 배열 / 각 데이터가 속한 그룹 번호만 담긴 리스트(또는 벡터)